"""CSGO (bug-fixed) on new 6 diverse styles (01,07,09,22,23,31) × 5 prompts.
Bug: content_scale/controlnet_conditioning_scale were swapped; fix is already applied
to /mnt/local/baselines/CSGO/infer/run_baseline.py.
Output: d_grand/ext_csgo/{sid}_p{N}.png
"""
import os, sys
import numpy as np
import torch
from PIL import Image

REPO = "/mnt/local/baselines/CSGO"
sys.path.insert(0, REPO)
from ip_adapter.utils import BLOCKS, controlnet_BLOCKS
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline
from ip_adapter import CSGO

BASE = f"{REPO}/base_models"
device = torch.device("cuda:0")

STY_DIR = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty"
OUT_DIR = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948/d_grand/ext_csgo"
os.makedirs(OUT_DIR, exist_ok=True)

STYLES = os.environ.get("STYLES_OVERRIDE", "01,07,09,22,23,31").split(",")
PROMPTS = [
    ("p0", "A panda eat bamboo"),
    ("p1", "A castle on a cliff by the sea"),
    ("p2", "A cat sitting on a windowsill"),
    ("p3", "A field of sunflowers at sunset"),
    ("p4", "A portrait of an old fisherman"),
]
NEG = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"
RES = 1024
SEED = 42

def build():
    vae = AutoencoderKL.from_pretrained(f"{BASE}/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained(
        f"{BASE}/TTPLanet_SDXL_Controlnet_Tile_Realistic", torch_dtype=torch.float16, use_safetensors=True)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        f"{BASE}/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16,
        add_watermarker=False, vae=vae)
    pipe.enable_vae_tiling()
    csgo = CSGO(pipe, f"{BASE}/IP-Adapter/sdxl_models/image_encoder",
                f"{BASE}/CSGO/csgo_4_32.bin", device,
                num_content_tokens=4, num_style_tokens=32,
                target_content_blocks=BLOCKS['content'], target_style_blocks=BLOCKS['style'],
                controlnet_adapter=True,
                controlnet_target_content_blocks=controlnet_BLOCKS['content'],
                controlnet_target_style_blocks=controlnet_BLOCKS['style'],
                content_model_resampler=True, style_model_resampler=True)
    csgo.use_CSD = None
    return csgo

def gen(csgo, style_img, prompt):
    black = Image.fromarray(np.zeros((RES, RES, 3), dtype=np.uint8)).convert("RGB")
    imgs = csgo.generate(
        pil_content_image=black, pil_style_image=style_img,
        prompt=prompt, negative_prompt=NEG,
        height=RES, width=RES,
        content_scale=0.01, style_scale=1.0,
        guidance_scale=7, num_images_per_prompt=1, num_samples=1,
        num_inference_steps=50, seed=SEED,
        image=black, controlnet_conditioning_scale=0.0)  # tile ControlNet OFF:
        # cc=1.0 on the black tile forced near-gray outputs (text-driven protocol = controlnet off)
    return imgs[0]

csgo = build()
print("CSGO loaded.", flush=True)
n = 0
for sid in STYLES:
    sty = Image.open(f"{STY_DIR}/{sid}.png").convert("RGB")
    for pk, ptxt in PROMPTS:
        out = f"{OUT_DIR}/{sid}_{pk}.png"
        img = gen(csgo, sty, ptxt)
        img.save(out)
        n += 1
        print(f"[{n}/{len(STYLES)*len(PROMPTS)}] saved {out}", flush=True)
print("CSGO_NEW6_DONE", n)
