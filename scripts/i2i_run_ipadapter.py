"""IP-Adapter (FLUX.1-dev) I2I style transfer on the 40-pair benchmark.

The T2I runner scripts/run_new6_ipadapter.py / baselines/ip_adapter_flux/gen.py use
FluxPipeline (text -> image, style via ip_adapter_image) and CANNOT preserve a content
image. For I2I we swap to FluxImg2ImgPipeline:
  - image           = content   (SDEdit init; strength<1 preserves structure)
  - ip_adapter_image= style      (XLabs FLUX IP-Adapter injects style)
  - strength        = 0.6        (lower = more content kept; tune if needed)

All weights are local and verified present:
  FLUX.1-dev          /mnt/local/cache/huggingface/FLUX.1-dev
  XLabs IP-Adapter    hub models--XLabs-AI--flux-ip-adapter/.../ip_adapter.safetensors
  CLIP image encoder  hub models--openai--clip-vit-large-patch14

BENCHMARK: 8 styles x 5 contents = 40 pairs.
Output: d_grand/i2i_ext_ipadapter/ipadapter_cnt{CC}_sty{SS}.png  (512x512)

Run: /home/ubuntu/chengyanli/flux/bin/python3 scripts/i2i_run_ipadapter.py
"""
import os, glob
os.environ.setdefault("HF_HOME", "/mnt/local/huggingface_cache")
import torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline

CWD = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
FLUX = "/mnt/local/cache/huggingface/FLUX.1-dev"
IPDIR = glob.glob("/mnt/local/huggingface_cache/hub/models--XLabs-AI--flux-ip-adapter/snapshots/*")[0]
CLIP = glob.glob("/mnt/local/huggingface_cache/hub/models--openai--clip-vit-large-patch14/snapshots/*")[0]
STY_DIR = f"{CWD}/sty"
CNT_DIR = f"{CWD}/cnt"
OUT_DIR = f"{CWD}/d_grand/i2i_ext_ipadapter"
os.makedirs(OUT_DIR, exist_ok=True)

STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]
CNTS = ["00", "04", "08", "12", "17"]
RES = 512
SEED = 43
IP_SCALE = 1.5
STRENGTH = 0.8           # 0.6 returned ~content (style ignored); 0.8 keeps structure + applies painterly style
STEPS = 28
GUIDANCE = 3.5


def build():
    pipe = FluxImg2ImgPipeline.from_pretrained(FLUX, torch_dtype=torch.bfloat16)
    pipe.load_ip_adapter(
        IPDIR, weight_name="ip_adapter.safetensors",
        image_encoder_pretrained_model_name_or_path=CLIP,
    )
    pipe.set_ip_adapter_scale(IP_SCALE)
    return pipe.to("cuda")


def gen(pipe, content_img, style_img):
    g = torch.Generator("cuda").manual_seed(SEED)
    img = pipe(
        prompt="",
        image=content_img,
        ip_adapter_image=style_img,
        strength=STRENGTH,
        width=RES, height=RES,
        num_inference_steps=STEPS, guidance_scale=GUIDANCE,
        generator=g,
    ).images[0]
    return img


pipe = build()
print("IP-Adapter(FLUX img2img) loaded.", flush=True)
total = len(STYLES) * len(CNTS)
n = 0
for cid in CNTS:
    content = Image.open(f"{CNT_DIR}/{cid}.png").convert("RGB").resize((RES, RES), Image.LANCZOS)
    for sid in STYLES:
        out = f"{OUT_DIR}/ipadapter_cnt{cid}_sty{sid}.png"
        style = Image.open(f"{STY_DIR}/{sid}.png").convert("RGB")
        img = gen(pipe, content, style)
        img.resize((RES, RES), Image.LANCZOS).save(out)
        n += 1
        print(f"[{n}/{total}] saved {out}", flush=True)
print("IPADAPTER_I2I_DONE", n)
