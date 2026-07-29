"""Compute CSD / CLIP-I(content) for USO I2I baseline. Mirrors eval_all.py exactly."""
import csv, os, sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import open_clip

CWD = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
OUT_DIR = f"{CWD}/d_grand/ext_uso_i2i"
STY_DIR = f"{CWD}/sty"
CNT_DIR = f"{CWD}/cnt"
RESULTS = f"{CWD}/results/ext_uso_i2i.csv"

CNTS = ["00", "01", "02", "03", "04", "05", "06", "07"]
STYLES = ["00", "10", "20", "27", "33"]

sys.path.insert(0, '../../CSD_Score')
from model import CSD_CLIP, convert_state_dict
csd_dir = '../../CSD_Score'
csd_model = CSD_CLIP('vit_large', 'default', model_path=f'{csd_dir}/models/ViT-L-14.pt')
ck = torch.load(f'{csd_dir}/models/checkpoint.pth', map_location='cpu', weights_only=False)
csd_model.load_state_dict(convert_state_dict(ck['model_state_dict']), strict=False)
csd_model = csd_model.to('cuda').eval()

clip_model_vit, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model_vit = clip_model_vit.to('cuda').eval()

_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1,3,1,1)
_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1,3,1,1)
_preproc = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])

def _csd_tensor(path):
    img = Image.open(path).convert('RGB')
    t = transforms.ToTensor()(img).unsqueeze(0).to('cuda')
    t = F.interpolate(t, size=(224,224), mode='bilinear', align_corners=False)
    return (t - _mean) / _std

def compute_csd(output_path, style_path):
    with torch.no_grad():
        _, _, s_out = csd_model(_csd_tensor(output_path).float())
        _, _, s_sty = csd_model(_csd_tensor(style_path).float())
    return (F.normalize(s_out, dim=-1) * F.normalize(s_sty, dim=-1)).sum().item()

def compute_clip_i(output_path, ref_path):
    t1 = _preproc(Image.open(output_path).convert('RGB')).unsqueeze(0).to('cuda')
    t2 = _preproc(Image.open(ref_path).convert('RGB')).unsqueeze(0).to('cuda')
    with torch.no_grad():
        f1 = clip_model_vit.encode_image(t1)
        f2 = clip_model_vit.encode_image(t2)
    return (F.normalize(f1, dim=-1) * F.normalize(f2, dim=-1)).sum().item()

rows, csds, clipis = [], [], []
for cid in CNTS:
    cnt_path = f"{CNT_DIR}/{cid}.png"
    for sid in STYLES:
        sty_path = f"{STY_DIR}/{sid}.png"
        out = f"{OUT_DIR}/c{cid}_s{sid}_0.png"
        if not os.path.exists(out):
            out = f"{OUT_DIR}/c{cid}_s{sid}.png"
        if not os.path.exists(out):
            print("MISSING", out); continue
        csd = compute_csd(out, sty_path)
        clipi = compute_clip_i(out, cnt_path)
        rows.append(["uso", cid, sid, f"{csd:.4f}", f"{clipi:.4f}"])
        csds.append(csd); clipis.append(clipi)

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
with open(RESULTS, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "cnt_id", "sty_id", "csd", "clip_i"])
    w.writerows(rows)

n = len(csds)
print(f"n={n}  mean CSD={sum(csds)/n:.4f}  mean CLIP-I(cnt)={sum(clipis)/n:.4f}")
print("CSV ->", RESULTS)
