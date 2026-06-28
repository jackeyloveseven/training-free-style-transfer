#!/usr/bin/env python3
"""Generate OUR Klein-9B hybrid I2I outputs at multiple seeds for the 40-pair
benchmark (8 styles x 5 content). Seed 43 already exists in d_grand/i2i_ablation;
this adds seeds 44,45,46 -> d_grand/i2i_ours_seed{N}/hybrid2_cnt{CC}_sty{SS}.png.
Run: /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/ours_i2i_multiseed.py
"""
import os, sys, time
sys.argv = [sys.argv[0]]
SEEDS = [44, 45, 46]
STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]
CNTS = ["00", "04", "08", "12", "17"]
import eval_all as ea
from pathlib import Path
cfg = ea.I2I_HYBRID
t0 = time.time(); n = 0; total = len(SEEDS)*len(STYLES)*len(CNTS)
for seed in SEEDS:
    os.environ["EVAL_SEED"] = str(seed)
    out_dir = Path(ea.CWD)/"d_grand"/f"i2i_ours_seed{seed}"; out_dir.mkdir(parents=True, exist_ok=True)
    for cid in CNTS:
        cnt_path = f"{ea.CWD}/cnt/{cid}.png"
        for sid in STYLES:
            sty_path = f"{ea.CWD}/sty/{sid}.png"
            op = str(out_dir/f"hybrid2_cnt{cid}_sty{sid}.png")
            n += 1
            if os.path.exists(op): continue
            img = ea.generate_i2i(cnt_path, sty_path, "masterpiece", cfg)
            img.save(op)
            if n % 8 == 0:
                print(f"[{n}/{total}] seed{seed} cnt{cid} sty{sid} ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE {total} imgs ({time.time()-t0:.0f}s)", flush=True)
