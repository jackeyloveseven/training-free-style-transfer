"""Qwen-Image-Edit-2511 I2I style transfer on the 40-pair benchmark.

Editing model: content image + style instruction (+ optional style reference image).
Uses QwenImageEditPlusPipeline, which accepts a LIST of images. We pass
  image = [content, style_reference]
and an instruction that tells it to restyle the content into the reference's style.
The content image is image[0] (structure to preserve); the style image is image[1]
(visual style reference). This keeps content while transferring style.

Weights (local, verified):
  /mnt/local/yzh/models/Qwen-Image-Edit-2511
diffusers in /home/ubuntu/chengyanli/flux/bin/python3 has QwenImageEditPlusPipeline.

Optional speed-up: the Lightning transformer
  /mnt/local/yzh/models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors
can replace the transformer for 8-step inference (set USE_LIGHTNING=1). Default uses the
base model at 40 steps for a faithful (non-distilled) baseline.

BENCHMARK: 8 styles x 5 contents = 40 pairs.
Output: d_grand/i2i_ext_qwen_edit/qwen_edit_cnt{CC}_sty{SS}.png  (512x512)

Run: /home/ubuntu/chengyanli/flux/bin/python3 scripts/i2i_run_qwen_edit.py
"""
import os
os.environ.setdefault("HF_HOME", "/mnt/local/huggingface_cache")
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

CWD = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
MODEL = "/mnt/local/yzh/models/Qwen-Image-Edit-2511"
STY_DIR = f"{CWD}/sty"
CNT_DIR = f"{CWD}/cnt"
OUT_DIR = f"{CWD}/d_grand/i2i_ext_qwen_edit"
os.makedirs(OUT_DIR, exist_ok=True)

STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]
CNTS = ["00", "04", "08", "12", "17"]
RES = 512
SEED = 0
STEPS = 40
TRUE_CFG = 4.0
NEG = " "
INSTRUCTION = ("Repaint the first image in the exact artistic style of the second image. "
               "Keep the subject, composition and spatial layout of the first image, but FULLY "
               "adopt the second image's color palette, brushstroke texture and painterly rendering, "
               "so the result looks like a painting of the first scene by the artist of the second image. "
               "Apply the style strongly and uniformly across the whole image.")


def build():
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    return pipe.to("cuda")


def gen(pipe, content_img, style_img):
    g = torch.Generator(device="cuda").manual_seed(SEED)
    out = pipe(
        image=[content_img, style_img],
        prompt=INSTRUCTION,
        negative_prompt=NEG,
        true_cfg_scale=TRUE_CFG,
        num_inference_steps=STEPS,
        guidance_scale=1.0,
        num_images_per_prompt=1,
        generator=g,
    )
    return out.images[0]


pipe = build()
print("Qwen-Image-Edit-2511 loaded.", flush=True)
total = len(STYLES) * len(CNTS)
n = 0
for cid in CNTS:
    content = Image.open(f"{CNT_DIR}/{cid}.png").convert("RGB").resize((RES, RES), Image.LANCZOS)
    for sid in STYLES:
        out_path = f"{OUT_DIR}/qwen_edit_cnt{cid}_sty{sid}.png"
        style = Image.open(f"{STY_DIR}/{sid}.png").convert("RGB").resize((RES, RES), Image.LANCZOS)
        img = gen(pipe, content, style)
        img.resize((RES, RES), Image.LANCZOS).save(out_path)
        n += 1
        print(f"[{n}/{total}] saved {out_path}", flush=True)
print("QWEN_EDIT_I2I_DONE", n)
