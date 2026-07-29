"""StyleShot I2I style transfer on the 40-pair benchmark.

StyleShot's official API supports content-image-conditioned I2I: the StyleShot class,
when constructed with a contour-detection transform (Canny/HED) + a Contour ControlNet,
exposes generate(style_image=..., content_image=...) -- the content image's contour map
drives a ControlNet so structure is preserved while the style encoder injects style.
(In the T2I runner scripts/run_new6_styleshot.py only style_image + prompt were used.)

==================== BLOCKER (verify before launch) ====================
BOTH the StyleShot repo code AND its weights are CURRENTLY MISSING on disk:
  - repo models/StyleShot  (provides `from ip_adapter import StyleShot`,
    StyleContentStableDiffusionControlNetPipeline): GONE -- no copy found anywhere on disk.
  - weights models/StyleShot/pretrained_weight/{ip.bin,
    style_aware_encoder.bin}: GONE.
  - SD1.5 base models/stable-diffusion-v1-5: GONE.
  - hub models--Gaojunyao--StyleShot: metadata only (0 blobs).
The existing d_grand/ext_styleshot/* (T2I) were made before these were deleted.

To run, restore all three:
  git clone https://github.com/open-mmlab/StyleShot models/StyleShot
  hf download Gaojunyao/StyleShot --local-dir models/StyleShot
  hf download Gaojunyao/StyleShot_lineart ...   # contour ControlNet for I2I
  hf download stable-diffusion-v1-5/stable-diffusion-v1-5 \
      --local-dir models/stable-diffusion-v1-5
Then verify imports and the exact ControlNet/transform names against the restored repo's
README before launching (the content-I2I constructor signature differs across StyleShot
releases -- the call below follows the upstream `styleshot_image_driven_demo.py`).
========================================================================

BENCHMARK: 8 styles x 5 contents = 40 pairs.
Output: d_grand/i2i_ext_styleshot/styleshot_cnt{CC}_sty{SS}.png  (512x512)

Run (in the env that has the StyleShot deps; T2I used the longcat-video conda env):
  /home/ubuntu/miniconda3/envs/longcat-video/bin/python scripts/i2i_run_styleshot.py
"""
import os, sys
import torch
from PIL import Image

# --- paths (mirror scripts/run_new6_styleshot.py; UPDATE after restoring weights) ---
SS_DIR = "models/StyleShot"
sys.path.insert(0, SS_DIR)
from transformers import CLIPVisionModelWithProjection
from diffusers import ControlNetModel
from annotator.hed import SOFT_HEDdetector          # contour detector shipped with StyleShot
from ip_adapter import StyleShot, StyleContentStableDiffusionControlNetPipeline

BASE = "models/stable-diffusion-v1-5"
CLIP_H = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
SS = "models/StyleShot"
CONTOUR_CN = "models/StyleShot_lineart"  # HED/lineart ControlNet weights

CWD = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
STY_DIR = f"{CWD}/sty"
CNT_DIR = f"{CWD}/cnt"
OUT_DIR = f"{CWD}/d_grand/i2i_ext_styleshot"
os.makedirs(OUT_DIR, exist_ok=True)

STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]
CNTS = ["00", "04", "08", "12", "17"]
RES = 512
SEED = 42


def build():
    device = "cuda"
    ip_ckpt = os.path.join(SS, "pretrained_weight/ip.bin")
    enc = os.path.join(SS, "pretrained_weight/style_aware_encoder.bin")
    # I2I path: contour detector + Contour ControlNet so content structure is preserved.
    detector = SOFT_HEDdetector()
    controlnet = ControlNetModel.from_pretrained(CONTOUR_CN, torch_dtype=torch.float32)
    pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(
        BASE, controlnet=controlnet)
    return StyleShot(device, pipe, ip_ckpt, enc, CLIP_H), detector


def gen(ss, detector, style_path, content_path):
    style = Image.open(style_path).convert("RGB")
    content = Image.open(content_path).convert("RGB").resize((RES, RES))
    contour = detector(content)                       # HED/lineart contour of content
    if not isinstance(contour, Image.Image):
        contour = Image.fromarray(contour)
    g = ss.generate(style_image=style, prompt=[[""]], content_image=contour,
                    seed=SEED, num_inference_steps=50, guidance_scale=7.5)
    return g[0][0]


ss, detector = build()
print("StyleShot (I2I) loaded.", flush=True)
total = len(STYLES) * len(CNTS)
n = 0
for cid in CNTS:
    for sid in STYLES:
        out = f"{OUT_DIR}/styleshot_cnt{cid}_sty{sid}.png"
        if os.path.exists(out):
            print(f"skip (exists) {out}", flush=True); n += 1; continue
        img = gen(ss, detector, f"{STY_DIR}/{sid}.png", f"{CNT_DIR}/{cid}.png")
        img.resize((RES, RES), Image.LANCZOS).save(out)
        n += 1
        print(f"[{n}/{total}] saved {out}", flush=True)
print("STYLESHOT_I2I_DONE", n)
