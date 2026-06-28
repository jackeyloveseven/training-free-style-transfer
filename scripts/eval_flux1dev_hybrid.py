"""Evaluate FLUX.1-dev Hybrid outputs with our exact metrics.

CSD + CLIP-T replicated verbatim from scripts/eval_all.py (~lines 83-130):
  - CSD: CSD_CLIP('vit_large') ViT-L-14.pt + checkpoint.pth
  - CLIP-T: open_clip ViT-B-32 pretrained 'openai'

Run in sensenova env:
  /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/eval_flux1dev_hybrid.py
"""
import csv, glob, os, re, sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import open_clip

ROOT = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
OUTDIR = f"{ROOT}/d_grand/flux1dev_hybrid"
STYDIR = f"{ROOT}/sty"
CSV_OUT = f"{ROOT}/results/flux1dev_hybrid.csv"
METHOD = "hybrid"

# prompt index -> exact prompt text used at generation time
PROMPTS = {
    0: "A panda eat bamboo",
    1: "A castle on a cliff by the sea",
    2: "A cat sitting on a windowsill",
    3: "A field of sunflowers at sunset",
    4: "A portrait of an old fisherman",
}

# ============== Load CSD model (verbatim from eval_all.py) ==============
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

def _img_to_clip_tensor(path_or_pil):
    if isinstance(path_or_pil, str):
        img = Image.open(path_or_pil).convert('RGB')
    else:
        img = path_or_pil.convert('RGB')
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

# ============== Evaluate grid ==============
paths = sorted(glob.glob(f"{OUTDIR}/sty*_p*.png"))
pat = re.compile(r"sty(\d+)_p(\d+)\.png$")
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)

rows = []
csds, clipts = [], []
for p in paths:
    m = pat.search(os.path.basename(p))
    if not m:
        continue
    sid, pidx = m.group(1), int(m.group(2))
    style_path = f"{STYDIR}/{sid}.png"
    text = PROMPTS[pidx]
    csd = compute_csd(p, style_path)
    clip_t = compute_clip_t(p, text)
    rows.append((METHOD, sid, f"p{pidx}", f"{csd:.4f}", f"{clip_t:.4f}"))
    csds.append(csd); clipts.append(clip_t)
    print(f"sty{sid}_p{pidx}: CSD={csd:.4f} CLIP-T={clip_t:.4f}")

with open(CSV_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "sty_id", "content_id", "csd", "clip_t"])
    w.writerows(rows)

n = len(csds)
mean_csd = sum(csds)/n if n else float('nan')
mean_clipt = sum(clipts)/n if n else float('nan')
print("="*50)
print(f"n={n}")
print(f"mean CSD    = {mean_csd:.4f}")
print(f"mean CLIP-T = {mean_clipt:.4f}")
print(f"CSV -> {CSV_OUT}")
