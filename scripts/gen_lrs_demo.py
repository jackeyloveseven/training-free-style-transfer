"""LRS demonstration: w/ LRS (no layout leak) vs w/o LRS (style layout leaks).

Generates hybrid2 (GRAG on, grag 0.5/0.7) and lrs_off (grag 1.0/1.0) for
strong-layout style references at a FIXED seed, so the only difference between
the pair is LRS on/off. Used to curate a clear w/ vs w/o LRS contrast for the
ablation figure (does NOT touch the original pipeline; copies-only convention).

Run: /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/gen_lrs_demo.py
"""
import os, sys
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import eval_all as E

PROMPT = "A panda eat bamboo"
SEED = 43
# Styles whose reference image has a strong, recognizable spatial layout
# (central figure / distinctive composition) so layout leakage is visible.
STYLES = ["23", "01", "38", "22", "27", "33", "08", "19", "00", "02"]
OUT = Path(E.CWD) / "d_grand" / "lrs_demo"
OUT.mkdir(parents=True, exist_ok=True)

W_LRS = E.HYBRID_DEFAULT
WO_LRS = {**E.HYBRID_DEFAULT, "grag_deep": 1.0, "grag_shallow": 1.0}

os.environ["EVAL_SEED"] = str(SEED)
for sid in STYLES:
    sty = f"{E.CWD}/sty/{sid}.png"
    for tag, cfg in [("wLRS", W_LRS), ("woLRS", WO_LRS)]:
        fp = OUT / f"{sid}_{tag}.png"
        if fp.exists():
            continue
        E.torch.cuda.empty_cache()
        E.generate_t2i(PROMPT, sty, cfg).save(fp)
        print(f"saved {fp}", flush=True)
print("LRS_DEMO_DONE", flush=True)
