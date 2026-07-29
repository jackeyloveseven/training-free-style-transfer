"""X-Flux IP-Adapter on new 6 diverse styles (01,07,09,22,23,31) × 5 prompts.
Output: d_grand/ext_ipadapter/{sid}_{N}.png  (same naming as existing 8-style results)
"""
import os, shutil, subprocess
from pathlib import Path

ROOT = Path("/home/ubuntu/chengyanli/items/save/attention_20260426_090948")
PYTHON = Path("/home/ubuntu/chengyanli/flux/bin/python3")
XFLUX_DIR = ROOT / "experiments/paper_run_20260519/baselines/x-flux"
WEIGHTS = ROOT / "experiments/paper_run_20260519/baseline_weights"
OUT_DIR = ROOT / "d_grand" / "ext_ipadapter"
TMP_DIR = ROOT / "d_grand" / ".tmp_ipadapter_new6"

STYLES = os.environ.get("STYLES_OVERRIDE", "01,07,09,22,23,31").split(",")
PROMPTS = [
    (0, "A panda eat bamboo"),
    (1, "A castle on a cliff by the sea"),
    (2, "A cat sitting on a windowsill"),
    (3, "A field of sunflowers at sunset"),
    (4, "A portrait of an old fisherman"),
]

OUT_DIR.mkdir(parents=True, exist_ok=True)
if TMP_DIR.exists():
    shutil.rmtree(TMP_DIR)
TMP_DIR.mkdir(parents=True)

env = os.environ.copy()
env.update({
    "FLUX_DEV": str(WEIGHTS / "FLUX.1-dev/flux1-dev.safetensors"),
    "AE": str(WEIGHTS / "FLUX.1-dev/ae.safetensors"),
    "CLIP": str(WEIGHTS / "text/clip-vit-large-patch14"),
    "T5": str(WEIGHTS / "text/t5-xxl"),
    "PYTHONPATH": str(XFLUX_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""),
})

ip_path = Path("models/flux-ip-adapter/ip_adapter.safetensors")
n = 0
for sid in STYLES:
    style_path = ROOT / "sty" / f"{sid}.png"
    for idx, prompt in PROMPTS:
        target = OUT_DIR / f"{sid}_{idx}.png"
        if target.exists():
            print(f"skip (exists): {target.name}")
            n += 1
            continue
        before = set(TMP_DIR.glob("*.png"))
        cmd = [
            str(PYTHON), str(XFLUX_DIR / "main.py"),
            "--prompt", prompt,
            "--img_prompt", str(style_path),
            "--ip_local_path", str(ip_path),
            "--device", "cuda",
            "--use_ip",
            "--width", "1024", "--height", "1024",
            "--num_steps", "25",
            "--guidance", "4.0",
            "--true_gs", "3.5",
            "--ip_scale", "1.0",
            "--timestep_to_start_cfg", "1",
            "--seed", "3407",
            "--save_path", str(TMP_DIR),
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"x-flux failed for sty{sid} p{idx}")
        after = sorted(set(TMP_DIR.glob("*.png")) - before, key=lambda p: p.stat().st_mtime)
        if not after:
            raise RuntimeError(f"No x-flux output for sty{sid}_p{idx}")
        after[-1].rename(target)
        n += 1
        print(f"[{n}/{len(STYLES)*len(PROMPTS)}] saved {target.name}", flush=True)

shutil.rmtree(TMP_DIR)
print(f"IPADAPTER_NEW6_DONE  {n}/{len(STYLES)*len(PROMPTS)}")
