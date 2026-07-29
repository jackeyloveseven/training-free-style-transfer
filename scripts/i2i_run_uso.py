"""USO (FLUX.1-dev) I2I style transfer on the 40-pair benchmark.

USO inference.py does content+style I2I via image_paths = [content, style]:
  - ref_imgs[:1] (content) -> preprocess_ref -> structural conditioning
  - ref_imgs[1:] (style)   -> SigLIP -> style conditioning
This is the SAME mechanism that produced the existing d_grand/ext_uso_i2i/c*_s*.png,
but that set used styles {00,10,20,27,33} x contents {c00..c07} -- NOT this benchmark.
This script regenerates on the exact 40 (content,style) pairs.

Reuses the subprocess/env wiring from scripts/run_new6_uso.py verbatim, changing only:
  - image_paths from ["", style]  ->  [content, style]   (adds content conditioning)
  - the style/content loops to the benchmark set
  - save_dir naming to the i2i convention

==================== BLOCKER (verify before launch) ====================
The USO weights are CURRENTLY MISSING on disk:
  - models--bytedance-research--USO (dit_lora.safetensors / projector.safetensors): NOT cached
  - baseline_weights/USO: empty
The USO repo code + FLUX/SigLIP/T5 base weights survive only in the BACKUP:
  /home/ubuntu/chengyanli/items/save/attention_20260426_090948_backup_20260612
Before running, restore the USO adapter weights, e.g.:
  hf download bytedance-research/USO --include "uso_flux_v1.0/*" \
      --local-dir /mnt/local/huggingface_cache/... (then update LORA/PROJECTION_MODEL below)
========================================================================

Output: d_grand/i2i_ext_uso/uso_cnt{CC}_sty{SS}_0.png  (1024x1024; downsample at metrics time)

Run: /home/ubuntu/chengyanli/flux/bin/python3 scripts/i2i_run_uso.py
"""
import json, os, subprocess
from pathlib import Path

ROOT = Path("/home/ubuntu/chengyanli/items/save/attention_20260426_090948")
BACKUP = Path("/home/ubuntu/chengyanli/items/save/attention_20260426_090948_backup_20260612")
PYTHON = Path("/home/ubuntu/chengyanli/flux/bin/python3")
USO_DIR = BACKUP / "experiments/paper_run_20260519/baselines/USO"
WEIGHTS = BACKUP / "experiments/paper_run_20260519/baseline_weights"
OUT_DIR = ROOT / "d_grand" / "i2i_ext_uso"
TMP_JSON = ROOT / "scripts" / "_tmp_uso_i2i.json"

STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]
CNTS = ["00", "04", "08", "12", "17"]

# USO adapter weights -- UPDATE these to wherever you restore them.
USO_LORA = "models/USO/uso_flux_v1.0/dit_lora.safetensors"
USO_PROJ = "models/USO/uso_flux_v1.0/projector.safetensors"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Build eval json: image_paths = [content, style] (content first -> structure preserved).
data = []
for cid in CNTS:
    for sid in STYLES:
        data.append({
            "prompt": "",
            "image_paths": [str(ROOT / "cnt" / f"{cid}.png"), str(ROOT / "sty" / f"{sid}.png")],
            "save_dir": f"uso_cnt{cid}_sty{sid}",
        })
with open(TMP_JSON, "w") as f:
    json.dump(data, f, indent=2)

env = os.environ.copy()
env.update({
    "FLUX_DEV": str(WEIGHTS / "FLUX.1-dev/flux1-dev.safetensors"),
    "AE": str(WEIGHTS / "FLUX.1-dev/ae.safetensors"),
    "CLIP": str(WEIGHTS / "text/clip-vit-large-patch14"),
    "T5": str(WEIGHTS / "text/t5-xxl"),
    "LORA": USO_LORA,
    "PROJECTION_MODEL": USO_PROJ,
    "SIGLIP_PATH": str(WEIGHTS / "siglip-so400m-patch14-384"),
    "PYTHONPATH": str(USO_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""),
})

cmd = [
    str(PYTHON), str(USO_DIR / "inference.py"),
    "--eval_json_path", str(TMP_JSON),
    "--save_path", str(OUT_DIR),
    "--model_type", "flux-dev",
    "--width", "1024", "--height", "1024",
    "--num_steps", "25", "--guidance", "4.0",
    "--seed", "3407", "--content_ref", "512", "--lora_rank", "128",
]
print("[uso_i2i] running inference ...")
print(" ".join(cmd))
proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True)
if proc.returncode != 0:
    raise SystemExit(proc.returncode)

TMP_JSON.unlink(missing_ok=True)
n = sum(1 for cid in CNTS for sid in STYLES
        if (OUT_DIR / f"uso_cnt{cid}_sty{sid}_0.png").exists())
print(f"USO_I2I_DONE  {n}/{len(STYLES)*len(CNTS)} files present")
