"""
Standalone CSD + CLIP-T evaluation for the external USO T2I baseline.
Metric functions copied verbatim from eval_all.py (compute_csd / compute_clip_t)
so numbers are directly comparable to our pipeline.

Run with sensenova env:
  /home/ubuntu/miniconda3/envs/sensenova/bin/python eval_ext_uso_t2i.py
"""
import csv, glob, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import open_clip

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

CWD = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
IMG_DIR = Path(CWD) / "d_grand" / "ext_uso_t2i"
STY_DIR = Path(CWD) / "sty"
RESULTS_DIR = Path(CWD) / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "ext_uso_t2i.csv"

STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]
PROMPTS = {
    0: "A panda eat bamboo",
    1: "A castle on a cliff by the sea",
    2: "A cat sitting on a windowsill",
    3: "A field of sunflowers at sunset",
    4: "A portrait of an old fisherman",
}

# ============== CSD + CLIP metrics (copied from eval_all.py) ==============
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

# ============== Evaluate ==============
rows = []
per_style = {s: [] for s in STYLES}
all_csd, all_clip = [], []

for s in STYLES:
    sty_path = str(STY_DIR / f"{s}.png")
    for ci, prompt in PROMPTS.items():
        out_path = str(IMG_DIR / f"{s}_{ci}_0.png")
        if not os.path.exists(out_path):
            print(f"MISSING: {out_path}")
            continue
        csd = compute_csd(out_path, sty_path)
        clip_t = compute_clip_t(out_path, prompt)
        rows.append(["ext_uso_t2i", s, ci, f"{csd:.4f}", f"{clip_t:.4f}"])
        per_style[s].append((csd, clip_t))
        all_csd.append(csd)
        all_clip.append(clip_t)

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "sty_id", "content_id", "csd", "clip_t"])
    w.writerows(rows)

print(f"\nWrote {OUT_CSV}  (n={len(rows)})\n")
print("Per-style means:")
print(f"{'sty':>5} {'CSD':>8} {'CLIP-T':>8}  n")
for s in STYLES:
    vals = per_style[s]
    if not vals:
        continue
    c = np.mean([v[0] for v in vals])
    t = np.mean([v[1] for v in vals])
    print(f"{s:>5} {c:8.4f} {t:8.4f}  {len(vals)}")
print("-" * 32)
print(f"{'ALL':>5} {np.mean(all_csd):8.4f} {np.mean(all_clip):8.4f}  {len(all_csd)}")
