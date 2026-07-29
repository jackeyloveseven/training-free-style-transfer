"""Compute CSD + CLIP-T metrics for all 5 methods on the 6 new styles.
Appends to results/ext_*.csv and results/t2i_new6styles.csv (if not already there).
Also prints a per-method summary table.
"""
import csv, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import open_clip

ROOT = Path("/home/ubuntu/chengyanli/items/save/attention_20260426_090948")
RESULTS_DIR = ROOT / "results"
CSD_DIR = "../../CSD_Score"
sys.path.insert(0, CSD_DIR)
from model import CSD_CLIP, convert_state_dict

STYLES = os.environ.get("STYLES_OVERRIDE", "01,07,09,22,23,31").split(",")
PROMPTS = [
    ("kp0", 0, "A panda eat bamboo"),
    ("kp1", 1, "A castle on a cliff by the sea"),
    ("kp2", 2, "A cat sitting on a windowsill"),
    ("kp3", 3, "A field of sunflowers at sunset"),
    ("kp4", 4, "A portrait of an old fisherman"),
]

print("Loading CSD …")
csd_model = CSD_CLIP('vit_large', 'default', model_path=f'{CSD_DIR}/models/ViT-L-14.pt')
ck = torch.load(f'{CSD_DIR}/models/checkpoint.pth', map_location='cpu', weights_only=False)
csd_model.load_state_dict(convert_state_dict(ck['model_state_dict']), strict=False)
csd_model = csd_model.to('cuda').eval()

clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to('cuda').eval()
clip_tok = open_clip.get_tokenizer('ViT-B-32')

_mn = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1,3,1,1)
_sd = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1,3,1,1)

def _to_clip(path):
    t = transforms.ToTensor()(Image.open(path).convert('RGB')).unsqueeze(0).to('cuda')
    t = F.interpolate(t, size=(224,224), mode='bilinear', align_corners=False)
    return (t - _mn) / _sd

def compute_csd(out_path, sty_path):
    with torch.no_grad():
        _, _, s_out = csd_model(_to_clip(out_path).float())
        _, _, s_sty = csd_model(_to_clip(sty_path).float())
    return (F.normalize(s_out, dim=-1) * F.normalize(s_sty, dim=-1)).sum().item()

def compute_clip_t(out_path, text):
    t = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711)),
    ])(Image.open(out_path).convert('RGB')).unsqueeze(0).to('cuda')
    tok = clip_tok([text]).to('cuda')
    with torch.no_grad():
        i = clip_model.encode_image(t)
        tx = clip_model.encode_text(tok)
    return (F.normalize(i, dim=-1) * F.normalize(tx, dim=-1)).sum().item()

METHODS = {
    "ours_hybrid2": {
        "csv": RESULTS_DIR / "t2i_new6styles.csv",
        "csv_method": "hybrid2",
        "path": lambda sid, kp, idx: ROOT / "d_grand" / "t2i_ablation" / f"hybrid2_{kp}_s{sid}.png",
        "content_id": lambda kp, idx: kp,
    },
    "styleshot": {
        "csv": RESULTS_DIR / "ext_styleshot.csv",
        "csv_method": "styleshot",
        "path": lambda sid, kp, idx: ROOT / "d_grand" / "ext_styleshot" / f"sty{sid}_{kp}.png",
        "content_id": lambda kp, idx: kp,
    },
    "csgo": {
        "csv": RESULTS_DIR / "ext_csgo.csv",
        "csv_method": "csgo",
        "path": lambda sid, kp, idx: ROOT / "d_grand" / "ext_csgo" / f"{sid}_p{idx}.png",
        "content_id": lambda kp, idx: f"p{idx}",
    },
    "uso": {
        "csv": RESULTS_DIR / "ext_uso_t2i.csv",
        "csv_method": "ext_uso_t2i",
        "path": lambda sid, kp, idx: ROOT / "d_grand" / "ext_uso_t2i" / f"{sid}_{idx}_0.png",
        "content_id": lambda kp, idx: str(idx),
    },
    "ipadapter": {
        "csv": RESULTS_DIR / "ext_ipadapter.csv",
        "csv_method": "ext_ipadapter",
        "path": lambda sid, kp, idx: ROOT / "d_grand" / "ext_ipadapter" / f"{sid}_{idx}.png",
        "content_id": lambda kp, idx: str(idx),
    },
}

summary = {}
for m_name, cfg in METHODS.items():
    print(f"\n{'='*60}\n{m_name}\n{'='*60}")
    existing_csv_rows = []
    if cfg["csv"].exists():
        with open(cfg["csv"]) as f:
            reader = csv.DictReader(f)
            existing_csv_rows = list(reader)
    existing_keys = {(r["sty_id"], r["content_id"]) for r in existing_csv_rows}
    new_rows, csds, clipts = [], [], []
    for kp, idx, prompt in PROMPTS:
        for sid in STYLES:
            img_path = cfg["path"](sid, kp, idx)
            cid = cfg["content_id"](kp, idx)
            if not img_path.exists():
                print(f"  MISSING: {img_path.name}")
                continue
            if (sid, cid) in existing_keys:
                print(f"  skip (already in CSV): sty{sid} {cid}")
                continue
            csd = compute_csd(str(img_path), str(ROOT / "sty" / f"{sid}.png"))
            clip_t = compute_clip_t(str(img_path), prompt)
            new_rows.append({"method": cfg["csv_method"], "sty_id": sid, "content_id": cid,
                             "csd": f"{csd:.4f}", "clip_t": f"{clip_t:.4f}"})
            csds.append(csd); clipts.append(clip_t)
            print(f"  sty{sid} {cid}: CSD={csd:.4f}  CLIP-T={clip_t:.4f}", flush=True)
    if new_rows:
        all_rows = existing_csv_rows + new_rows
        with open(cfg["csv"], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method","sty_id","content_id","csd","clip_t"])
            writer.writeheader(); writer.writerows(all_rows)
        print(f"  appended {len(new_rows)} rows → {cfg['csv'].name}")
    if csds:
        summary[m_name] = (np.mean(csds), np.mean(clipts), len(csds))

print(f"\n{'='*60}")
print("SUMMARY — new6 styles metrics")
print(f"{'Method':<15} {'n':>3} {'CSD':>7} {'CLIP-T':>8}")
print("-" * 40)
for m_name, (csd, clipt, n) in summary.items():
    print(f"{m_name:<15} {n:>3} {csd:>7.4f} {clipt:>8.4f}")
print("METRICS_NEW6_DONE")
