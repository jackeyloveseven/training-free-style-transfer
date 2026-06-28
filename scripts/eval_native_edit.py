"""
Paper experiment #19 — Klein 9B NATIVE in-context edit as a style-transfer baseline.

NO SLT / NO Adam / NO style forward. We feed the STYLE image as Klein's native
in-context EDIT reference (DiffSynth Flux2ImagePipeline `edit_image=...`) plus a
content text prompt, and run the plain native denoise loop. This measures whether
the raw FLUX.2 backbone's own edit capability achieves style transfer on its own.

We import eval_all (everything heavy is under `if __name__ == '__main__'`, so the
import only loads the Klein pipeline + CSD/CLIP metric models and registers the
attention hooks — it does NOT run any benchmark). We then call the pipeline's
NATIVE __call__ with edit_image=style. The eval_all hooks are made transparent by
emptying controller.self_layers (cache mode records nothing, output unchanged).

Usage:
  /home/ubuntu/miniconda3/envs/sensenova/bin/python eval_native_edit.py --task smoke
  /home/ubuntu/miniconda3/envs/sensenova/bin/python eval_native_edit.py --task full
"""
import argparse, csv, os
from pathlib import Path
from PIL import Image

# Import the already-built pipeline + metrics from eval_all (no benchmark runs on import).
import eval_all
from eval_all import pipe, controller, compute_csd, compute_clip_t, CWD

# Make the eval_all attention hooks fully transparent: with an empty self_layers
# list the hooked processors never cache nor modify attention -> identical to the
# stock FLUX.2 forward, and no per-step tensor accumulation.
controller.self_layers = []
controller.reset_mode()

OUT_DIR = Path(CWD) / "d_grand" / "native_edit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = Path(CWD) / "results" / "native_edit.csv"

STY_DIR = Path(CWD) / "sty"

# Native-edit config (Klein defaults — faithful raw backbone, no style tuning):
SEED = 43
STEPS = 25            # match our pipeline
SIZE = 512            # 512^2, match our pipeline
CFG_SCALE = 1.0       # Klein default (single forward, no negative)
EMBED_GUIDANCE = 4.0  # Klein default embedded guidance
AUTO_RESIZE = True    # native edit-image handling

PROMPTS = [
    ("kp0", "A panda eat bamboo"),
    ("kp1", "A castle on a cliff by the sea"),
    ("kp2", "A cat sitting on a windowsill"),
    ("kp3", "A field of sunflowers at sunset"),
    ("kp4", "A portrait of an old fisherman"),
]
STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]


def native_edit(style_path, prompt):
    """Klein 9B native in-context edit: style image as edit reference + content prompt."""
    style_img = Image.open(style_path).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    image = pipe(
        prompt=prompt,
        edit_image=style_img,
        edit_image_auto_resize=AUTO_RESIZE,
        cfg_scale=CFG_SCALE,
        embedded_guidance=EMBED_GUIDANCE,
        height=SIZE, width=SIZE,
        num_inference_steps=STEPS,
        seed=SEED,
    )
    return image


def run_smoke():
    style_path = str(STY_DIR / "27.png")
    prompt = "A cat sitting on a wooden table"
    print(f"[smoke] native edit  style=sty/27.png  prompt={prompt!r}", flush=True)
    img = native_edit(style_path, prompt)
    out_png = OUT_DIR / "smoke_sty27_cat.png"
    img.save(out_png)
    csd = compute_csd(str(out_png), style_path)
    clip_t = compute_clip_t(str(out_png), prompt)
    print(f"[smoke] saved {out_png}")
    print(f"[smoke] CSD(vs style)={csd:.4f}  CLIP-T={clip_t:.4f}", flush=True)
    return csd, clip_t


def run_full():
    rows = []
    for sid in STYLES:
        style_path = str(STY_DIR / f"{sid}.png")
        for cid, prompt in PROMPTS:
            img = native_edit(style_path, prompt)
            out_png = OUT_DIR / f"sty{sid}_{cid}.png"
            img.save(out_png)
            csd = compute_csd(str(out_png), style_path)
            clip_t = compute_clip_t(str(out_png), prompt)
            rows.append(("native_edit", sid, cid, csd, clip_t))
            print(f"sty{sid} {cid}: CSD={csd:.4f} CLIP-T={clip_t:.4f}  -> {out_png.name}", flush=True)

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "sty_id", "content_id", "csd", "clip_t"])
        w.writerows(rows)

    mean_csd = sum(r[3] for r in rows) / len(rows)
    mean_clip = sum(r[4] for r in rows) / len(rows)
    print(f"\n{'='*60}")
    print(f"NATIVE EDIT (n={len(rows)})  mean CSD={mean_csd:.4f}  mean CLIP-T={mean_clip:.4f}")
    print(f"Saved CSV: {RESULTS_CSV}")
    print(f"{'='*60}", flush=True)
    return mean_csd, mean_clip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="smoke", choices=["smoke", "full"])
    args = ap.parse_args()
    if args.task == "smoke":
        run_smoke()
    else:
        run_full()
