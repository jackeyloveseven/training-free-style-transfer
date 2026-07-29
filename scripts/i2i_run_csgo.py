"""CSGO I2I style transfer on the 40-pair benchmark.
Content image + style image -> stylized output preserving content.

CSGO natively supports content+style I2I (official infer/infer_CSGO.py protocol):
  - pil_content_image = content   (content IP tokens, content_scale)
  - pil_style_image   = style     (style IP tokens, style_scale)
  - image             = content   (Tile ControlNet preserves structure)

Reuses the model-loading code from scripts/run_new6_csgo.py (build()) verbatim,
switched from text-driven (black content, content_scale=0.01, ctrlnet off) to the
official I2I config (content image, content_scale=0.5, controlnet_conditioning_scale=0.6).

BENCHMARK: 8 styles x 5 contents = 40 pairs.
Output: d_grand/i2i_ext_csgo/csgo_cnt{CC}_sty{SS}.png  (512x512)

Run: /home/ubuntu/chengyanli/flux/bin/python3 scripts/i2i_run_csgo.py

PREREQUISITE (verify before launch): the CSGO adapter checkpoint
  models/CSGO/csgo_4_32.bin
must exist. It is currently MISSING on disk (only InstantX/CSGO metadata is cached,
no blobs). Re-download with:
  hf download InstantX/CSGO csgo_4_32.bin \
      --local-dir models/CSGO
"""
import os, sys
import torch
from PIL import Image

REPO = "models/CSGO-code"
sys.path.insert(0, REPO)
from ip_adapter.utils import BLOCKS, controlnet_BLOCKS, resize_content
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline
from ip_adapter import CSGO

BASE = f"{REPO}/base_models"
device = torch.device("cuda:0")

CWD = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
STY_DIR = f"{CWD}/sty"
CNT_DIR = f"{CWD}/cnt"
OUT_DIR = f"{CWD}/d_grand/i2i_ext_csgo"
os.makedirs(OUT_DIR, exist_ok=True)

STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]
CNTS = ["00", "04", "08", "12", "17"]
RES = 512
SEED = 42
NEG = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"
PROMPT = "best quality, high quality"  # generic; avoids the BLIP caption dependency


def build():
    # --- verbatim from scripts/run_new6_csgo.py:build() ---
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


def gen(csgo, content_img, style_img):
    # Official I2I config (infer/infer_CSGO.py): content drives both the content IP
    # tokens AND the Tile ControlNet; style drives the style IP tokens.
    # resize_content snaps to /16 and up to 1024 long side (CSGO operates at SDXL res).
    w, h, content_resized = resize_content(content_img)
    imgs = csgo.generate(
        pil_content_image=content_resized, pil_style_image=style_img,
        prompt=PROMPT, negative_prompt=NEG,
        height=h, width=w,
        content_scale=0.5, style_scale=1.0,
        guidance_scale=10, num_images_per_prompt=1, num_samples=1,
        num_inference_steps=50, seed=SEED,
        image=content_resized.convert("RGB"), controlnet_conditioning_scale=0.6)
    return imgs[0]


csgo = build()
print("CSGO loaded.", flush=True)
total = len(STYLES) * len(CNTS)
n = 0
for cid in CNTS:
    content = Image.open(f"{CNT_DIR}/{cid}.png").convert("RGB")
    for sid in STYLES:
        out = f"{OUT_DIR}/csgo_cnt{cid}_sty{sid}.png"
        if os.path.exists(out):
            print(f"skip (exists) {out}", flush=True); n += 1; continue
        style = Image.open(f"{STY_DIR}/{sid}.png").convert("RGB")
        img = gen(csgo, content, style)
        img.resize((RES, RES), Image.LANCZOS).save(out)
        n += 1
        print(f"[{n}/{total}] saved {out}", flush=True)
print("CSGO_I2I_DONE", n)
