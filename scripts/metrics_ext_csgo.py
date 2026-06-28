"""Compute CSD / CLIP-T for CSGO external baseline. Mirrors eval_all.py."""
import csv, os, sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import open_clip

CWD = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
OUT_DIR = f"{CWD}/d_grand/ext_csgo"
STY_DIR = f"{CWD}/sty"
RESULTS = f"{CWD}/results/ext_csgo.csv"

STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]
PROMPTS = [
    ("p0", "A panda eat bamboo"),
    ("p1", "A castle on a cliff by the sea"),
    ("p2", "A cat sitting on a windowsill"),
    ("p3", "A field of sunflowers at sunset"),
    ("p4", "A portrait of an old fisherman"),
]

sys.path.insert(0, '/home/ubuntu/chengyanli/items/CSD_Score')
from model import CSD_CLIP, convert_state_dict

csd_dir = '/home/ubuntu/chengyanli/items/CSD_Score'
csd_model = CSD_CLIP('vit_large', 'default', model_path=f'{csd_dir}/models/ViT-L-14.pt')
ck = torch.load(f'{csd_dir}/models/checkpoint.pth', map_location='cpu', weights_only=False)
csd_model.load_state_dict(convert_state_dict(ck['model_state_dict']), strict=False)
csd_model = csd_model.to('cuda').eval()

clip_model_vit, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model_vit = clip_model_vit.to('cuda').eval()
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1,3,1,1)
_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1,3,1,1)

def _img_to_clip_tensor(path):
    img = Image.open(path).convert('RGB')
    t = transforms.ToTensor()(img).unsqueeze(0).to('cuda')
    t = F.interpolate(t, size=(224,224), mode='bilinear', align_corners=False)
    return (t - _mean) / _std

def compute_csd(output_path, style_path):
    t_out = _img_to_clip_tensor(output_path)
    t_sty = _img_to_clip_tensor(style_path)
    with torch.no_grad():
        _, _, s_out = csd_model(t_out.float())
        _, _, s_sty = csd_model(t_sty.float())
    return (F.normalize(s_out, dim=-1) * F.normalize(s_sty, dim=-1)).sum().item()

def compute_clip_t(output_path, text):
    img = Image.open(output_path).convert('RGB')
    img_t = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])(img).unsqueeze(0).to('cuda')
    text_t = clip_tokenizer([text]).to('cuda')
    with torch.no_grad():
        img_feat = clip_model_vit.encode_image(img_t)
        txt_feat = clip_model_vit.encode_text(text_t)
    return (F.normalize(img_feat, dim=-1) * F.normalize(txt_feat, dim=-1)).sum().item()

rows = []
csds, clipts = [], []
for sid in STYLES:
    sty_path = f"{STY_DIR}/{sid}.png"
    for cid, prompt in PROMPTS:
        out = f"{OUT_DIR}/{sid}_{cid}.png"
        if not os.path.exists(out):
            print("MISSING", out); continue
        csd = compute_csd(out, sty_path)
        clipt = compute_clip_t(out, prompt)
        rows.append(["csgo", sid, cid, f"{csd:.4f}", f"{clipt:.4f}"])
        csds.append(csd); clipts.append(clipt)

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
with open(RESULTS, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "sty_id", "content_id", "csd", "clip_t"])
    w.writerows(rows)

n = len(csds)
print(f"n={n}  mean CSD={sum(csds)/n:.4f}  mean CLIP-T={sum(clipts)/n:.4f}")
print("CSV ->", RESULTS)
