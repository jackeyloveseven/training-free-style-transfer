"""V-Shuffle (our MM-DiT adaptation) baseline runner.
Validation mode (default): 3 (style,prompt) pairs, print CSD/CLIP-T, save imgs.
Full mode (--full): 8-style x 5-prompt subset -> results/ext_vshuffle.csv.
Run: /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/run_vshuffle.py [--full]
"""
import os, sys, csv
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import eval_all as E

CFG = E.T2I_BASELINES["vshuffle"]
OUT = Path(E.CWD) / "d_grand" / "ext_vshuffle"; OUT.mkdir(parents=True, exist_ok=True)
PROMPTS = E.T2I_PROMPTS_30   # kp0..kp4 (panda/castle/cat/sunflowers/fisherman)

def gen(sty, pid, prompt):
    sty_path = f"{E.CWD}/sty/{sty}.png"
    img = E.generate_t2i(prompt, sty_path, CFG)
    fp = OUT / f"{sty}_{pid}.png"; img.save(fp)
    return E.compute_csd(str(fp), sty_path), E.compute_clip_t(str(fp), prompt)

full = "--full" in sys.argv
if not full:
    print("=== V-Shuffle VALIDATION (3 pairs) ===", flush=True)
    for sty, (pid, prompt) in [("27", PROMPTS[0]), ("20", PROMPTS[0]), ("12", PROMPTS[2])]:
        csd, ct = gen(sty, pid, prompt)
        print(f"  sty{sty} {pid} '{prompt[:24]}': CSD={csd:.4f} CLIP-T={ct:.4f}", flush=True)
    print("VALIDATION_DONE", flush=True)
else:
    STYLES = E.STYLES_6  # 8 styles
    rows = []
    for sty in STYLES:
        for pid, prompt in PROMPTS:
            csd, ct = gen(sty, pid, prompt)
            rows.append({"method":"vshuffle","sty_id":sty,"content_id":pid,"csd":f"{csd:.4f}","clip_t":f"{ct:.4f}"})
            print(f"  sty{sty} {pid}: CSD={csd:.4f} CLIP-T={ct:.4f}", flush=True)
    import numpy as np
    with open(E.RESULTS_DIR/"ext_vshuffle.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["method","sty_id","content_id","csd","clip_t"]); w.writeheader(); w.writerows(rows)
    print(f"\nVSHUFFLE mean CSD={np.mean([float(r['csd']) for r in rows]):.4f} "
          f"CLIP-T={np.mean([float(r['clip_t']) for r in rows]):.4f} (n={len(rows)})")
    print("VSHUFFLE_FULL_DONE", flush=True)
