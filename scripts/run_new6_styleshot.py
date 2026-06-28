"""StyleShot on new 6 diverse styles (01,07,09,22,23,31) × 5 prompts.
Supplements existing ext_styleshot (original 8 styles).
Output: d_grand/ext_styleshot/sty{sid}_kp{N}.png
"""
import os, sys
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

SS_DIR = "/mnt/local/baselines/StyleShot"
sys.path.insert(0, SS_DIR)
from ip_adapter import StyleShot

BASE = "/mnt/local/baselines/models/stable-diffusion-v1-5"
CLIP_H = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
SS = "/mnt/local/baselines/models/StyleShot"
STY_DIR = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty"
OUT_DIR = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948/d_grand/ext_styleshot"

STYLES = os.environ.get("STYLES_OVERRIDE", "01,07,09,22,23,31").split(",")
PROMPTS = [
    ("kp0", "A panda eat bamboo"),
    ("kp1", "A castle on a cliff by the sea"),
    ("kp2", "A cat sitting on a windowsill"),
    ("kp3", "A field of sunflowers at sunset"),
    ("kp4", "A portrait of an old fisherman"),
]

def build():
    device = "cuda"
    ip_ckpt = os.path.join(SS, "pretrained_weight/ip.bin")
    enc = os.path.join(SS, "pretrained_weight/style_aware_encoder.bin")
    pipe = StableDiffusionPipeline.from_pretrained(BASE, safety_checker=None)
    return StyleShot(device, pipe, ip_ckpt, enc, CLIP_H)

def gen_one(ss, style_path, prompt, out_path, seed=42):
    img = Image.open(style_path).convert("RGB")
    g = ss.generate(style_image=img, prompt=[[prompt]], seed=seed,
                    num_inference_steps=50, guidance_scale=7.5)
    g[0][0].save(out_path)

os.makedirs(OUT_DIR, exist_ok=True)
ss = build()
n = 0
for sid in STYLES:
    for cid, prompt in PROMPTS:
        out = os.path.join(OUT_DIR, f"sty{sid}_{cid}.png")
        gen_one(ss, os.path.join(STY_DIR, f"{sid}.png"), prompt, out)
        n += 1
        print(f"[{n}/{len(STYLES)*len(PROMPTS)}] saved {out}", flush=True)
print("STYLESHOT_NEW6_DONE", n)
