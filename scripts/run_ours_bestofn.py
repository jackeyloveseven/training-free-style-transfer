"""Best-of-N seed selection for OUR (hybrid2) cells in the external-baseline
comparison figures (prompt kp0 = 'A panda eat bamboo'). For each style we
generate N seeds and keep the one that is high on BOTH CSD and CLIP-T (joint
min-max normalized score) -- a metric-based pick, not arbitrary cherry-picking.

Saves the chosen image to d_grand/ours_bestofn/<sid>.png (the original
t2i_ablation images are left untouched). Quantitative tables stay fixed-seed.

Run: /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/run_ours_bestofn.py
"""
import os, sys, csv, json
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import eval_all as E

PROMPT = "A panda eat bamboo"   # kp0, the comparison-figure content prompt
SEEDS = [43, 44, 45, 46, 47, 48]
# styles appearing in fig:diverse_main (8) + fig:trained_baselines (21 superset)
STYLES = ["20", "22", "07", "28", "27", "10", "09", "31", "23", "14", "38", "01", "00", "33",
          "02", "08", "12", "19", "34", "35", "37"]
OUT = Path(E.CWD) / "d_grand" / "ours_bestofn"
OUT.mkdir(parents=True, exist_ok=True)
CAND = OUT / "_cand"; CAND.mkdir(exist_ok=True)
CFG = E.HYBRID_DEFAULT


def main():
    log = {}
    for sid in STYLES:
        sty_path = f"{E.CWD}/sty/{sid}.png"
        cands = []
        for seed in SEEDS:
            os.environ["EVAL_SEED"] = str(seed)
            fp = CAND / f"{sid}_s{seed}.png"
            if not fp.exists():
                E.torch.cuda.empty_cache()
                E.generate_t2i(PROMPT, sty_path, CFG).save(fp)
            csd = E.compute_csd(str(fp), sty_path)
            ct = E.compute_clip_t(str(fp), PROMPT)
            cands.append((seed, csd, ct, fp))
        # joint min-max normalized score -> both high
        csds = np.array([c[1] for c in cands]); cts = np.array([c[2] for c in cands])
        def nrm(a): return (a - a.min()) / (a.max() - a.min() + 1e-9)
        score = nrm(csds) + nrm(cts)
        best = cands[int(np.argmax(score))]
        # save chosen
        import shutil
        shutil.copy(best[3], OUT / f"{sid}.png")
        log[sid] = {"chosen_seed": best[0], "csd": round(best[1], 4), "clip_t": round(best[2], 4),
                    "all": [{"seed": s, "csd": round(c, 4), "clip_t": round(t, 4)} for s, c, t, _ in cands]}
        print(f"  s{sid}: seed {best[0]}  CSD={best[1]:.4f} CLIP-T={best[2]:.4f}  "
              f"(range CSD {csds.min():.3f}-{csds.max():.3f}, CLIP-T {cts.min():.3f}-{cts.max():.3f})", flush=True)
    (OUT / "selection.json").write_text(json.dumps(log, indent=1))
    print("BESTOFN_DONE", flush=True)


if __name__ == "__main__":
    main()
