"""USO (FLUX.1-dev) on new 6 diverse styles (01,07,09,22,23,31) × 5 prompts.
Output: d_grand/ext_uso_t2i/{sid}_{N}_0.png  (same naming as existing 8-style results)
"""
import json, os, subprocess
from pathlib import Path

ROOT = Path("/home/ubuntu/chengyanli/items/save/attention_20260426_090948")
PYTHON = Path("/home/ubuntu/chengyanli/flux/bin/python3")
USO_DIR = ROOT / "experiments/paper_run_20260519/baselines/USO"
WEIGHTS = ROOT / "experiments/paper_run_20260519/baseline_weights"
OUT_DIR = ROOT / "d_grand" / "ext_uso_t2i"
TMP_JSON = ROOT / "scripts" / "_tmp_uso_new6.json"

STYLES = os.environ.get("STYLES_OVERRIDE", "01,07,09,22,23,31").split(",")
PROMPTS = [
    (0, "A panda eat bamboo"),
    (1, "A castle on a cliff by the sea"),
    (2, "A cat sitting on a windowsill"),
    (3, "A field of sunflowers at sunset"),
    (4, "A portrait of an old fisherman"),
]

OUT_DIR.mkdir(parents=True, exist_ok=True)

data = []
for sid in STYLES:
    for idx, prompt in PROMPTS:
        data.append({
            "prompt": prompt,
            "image_paths": ["", str(ROOT / "sty" / f"{sid}.png")],
            "save_dir": f"{sid}_{idx}",
        })

with open(TMP_JSON, "w") as f:
    json.dump(data, f, indent=2)

env = os.environ.copy()
env.update({
    "FLUX_DEV": str(WEIGHTS / "FLUX.1-dev/flux1-dev.safetensors"),
    "AE": str(WEIGHTS / "FLUX.1-dev/ae.safetensors"),
    "CLIP": str(WEIGHTS / "text/clip-vit-large-patch14"),
    "T5": str(WEIGHTS / "text/t5-xxl"),
    "LORA": "/mnt/local/huggingface_cache/hub/models--bytedance-research--USO/snapshots/18feb32b915cd0c577eb43fbcaf28a593bb629f5/uso_flux_v1.0/dit_lora.safetensors",
    "PROJECTION_MODEL": "/mnt/local/huggingface_cache/hub/models--bytedance-research--USO/snapshots/18feb32b915cd0c577eb43fbcaf28a593bb629f5/uso_flux_v1.0/projector.safetensors",
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
print("[uso_new6] running inference …")
print(" ".join(cmd))
proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True)
if proc.returncode != 0:
    raise SystemExit(proc.returncode)

TMP_JSON.unlink(missing_ok=True)

n = sum(1 for sid in STYLES for idx, _ in PROMPTS
        if (OUT_DIR / f"{sid}_{idx}_0.png").exists())
print(f"USO_NEW6_DONE  {n}/{len(STYLES)*len(PROMPTS)} files present")
