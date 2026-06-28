#!/usr/bin/env python3
"""Run Klein-9B V-Shuffle on the FULL T2I benchmark (39 styles x 6 prompts = 234)
to match the Klein hybrid full benchmark (d_grand/t2i_benchmark, t2i_full_benchmark.csv).
Mirrors run_t2i_full() but for the vshuffle baseline only, using eval_all's own
Klein set_vshuffle (NOT the old FLUX core). Resumable (skips existing tiles).
Run: /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/run_vshuffle_klein_full.py
"""
import os, sys, csv, time
sys.argv = [sys.argv[0]]          # keep eval_all's argparse from seeing our args
import eval_all as ea             # module-level: loads Klein model + defs
from pathlib import Path

cfg = {"mode": "vshuffle", "inject_beta": 1.0}   # faithful full injection (beta=1.0)
out_dir = Path(ea.CWD) / "d_grand" / "t2i_vshuffle_b10_full"
out_dir.mkdir(parents=True, exist_ok=True)
csv_path = Path(ea.RESULTS_DIR) / "t2i_vshuffle_b10_full.csv"

prompts = ea.T2I_PROMPTS_FULL
styles = ea.STYLES_ALL
total = len(prompts) * len(styles)
rows = []
count = 0
t0 = time.time()
print(f"KLEIN V-SHUFFLE FULL: {len(prompts)} prompts x {len(styles)} styles = {total}", flush=True)
for pid, prompt in prompts:
    for sty in styles:
        sty_path = f"{ea.CWD}/sty/{sty}.png"
        out_path = str(out_dir / f"vshuffle_{pid}_s{sty}.png")
        if not os.path.exists(out_path):
            img = ea.generate_t2i(prompt, sty_path, cfg)
            img.save(out_path)
        csd = ea.compute_csd(out_path, sty_path)
        clip_t = ea.compute_clip_t(out_path, prompt)
        rows.append({"method": "vshuffle", "sty_id": sty, "content_id": pid,
                     "csd": f"{csd:.4f}", "clip_t": f"{clip_t:.4f}"})
        count += 1
        if count % 10 == 0:
            print(f"[{count}/{total}] {pid} s{sty} csd={csd:.3f} clipt={clip_t:.3f} ({time.time()-t0:.0f}s)", flush=True)
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "sty_id", "content_id", "csd", "clip_t"])
    w.writeheader(); w.writerows(rows)
import statistics as st
print(f"DONE {csv_path} n={len(rows)} mean_csd={st.mean(float(r['csd']) for r in rows):.4f} "
      f"mean_clipt={st.mean(float(r['clip_t']) for r in rows):.4f} ({time.time()-t0:.0f}s)", flush=True)
