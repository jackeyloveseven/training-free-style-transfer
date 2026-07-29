"""
Metrics for CSGO baseline. Replicates compute_csd / compute_clip_t from eval_all.py EXACTLY.
Run with: /home/ubuntu/miniconda3/envs/sensenova/bin/python eval_ext_csgo.py [--smoke PATH STY]
"""
import sys, csv, glob, os, argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

CWD = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
STY_DIR = f"{CWD}/sty"
OUT_DIR = f"{CWD}/d_grand/ext_csgo"
CSV_PATH = f"{CWD}/results/ext_csgo.csv"

PROMPTS = {
    "p0": "A panda eat bamboo",
    "p1": "A castle on a cliff by the sea",
    "p2": "A cat sitting on a windowsill",
    "p3": "A field of sunflowers at sunset",
    "p4": "A portrait of an old fisherman",
}

# ---- CSD + CLIP setup (verbatim from eval_all.py) ----
sys.path.insert(0, '../../CSD_Score')
from model import CSD_CLIP, convert_state_dict
import open_clip

csd_dir = '../../CSD_Score'
csd_model = CSD_CLIP('vit_large', 'default', model_path=f'{csd_dir}/models/ViT-L-14.pt')
ck = torch.load(f'{csd_dir}/models/checkpoint.pth', map_location='cpu', weights_only=False)
csd_model.load_state_dict(convert_state_dict(ck['model_state_dict']), strict=False)
csd_model = csd_model.to('cuda').eval()

clip_model_vit, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model_vit = clip_model_vit.to('cuda').eval()
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1, 3, 1, 1)
_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1, 3, 1, 1)


def _img_to_clip_tensor(path_or_pil):
    if isinstance(path_or_pil, str):
        img = Image.open(path_or_pil).convert('RGB')
    else:
        img = path_or_pil.convert('RGB')
    t = transforms.ToTensor()(img).unsqueeze(0).to('cuda')
    t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", nargs=2, metavar=("PATH", "STY"))
    args = ap.parse_args()
    if args.smoke:
        p, s = args.smoke
        print("CSD", compute_csd(p, f"{STY_DIR}/{s}.png"))
        print("CLIP-T", compute_clip_t(p, "A cat sitting on a windowsill"))
        sys.exit(0)

    rows = []
    csds, clips = [], []
    for f in sorted(glob.glob(f"{OUT_DIR}/*_p*.png")):
        name = os.path.basename(f)[:-4]  # e.g. 27_p2
        if name.startswith("smoke"):
            continue
        sid, pk = name.split("_")
        text = PROMPTS[pk]
        csd = compute_csd(f, f"{STY_DIR}/{sid}.png")
        clip_t = compute_clip_t(f, text)
        rows.append(["csgo", sid, pk, f"{csd:.4f}", f"{clip_t:.4f}"])
        csds.append(csd)
        clips.append(clip_t)
        print(f"{name}: CSD={csd:.4f} CLIP-T={clip_t:.4f}")

    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "sty_id", "content_id", "csd", "clip_t"])
        w.writerows(rows)

    n = len(csds)
    print(f"\n=== CSGO baseline: n={n} ===")
    print(f"mean CSD    = {sum(csds)/n:.4f}")
    print(f"mean CLIP-T = {sum(clips)/n:.4f}")
    print(f"CSV: {CSV_PATH}")
