"""Fresh last-attempt probe: does the OT layer-routing split (OT_LAYER_FRAC)
Pareto-improve BOTH CSD and CLIP-T at fixed inference cost?

Deep layers run Sinkhorn-OT (style->CSD), shallow layers run standard attn
(structure->CLIP-T). The split fraction was never swept in prior experiments.
This adds NO inference time (same #forwards) and changes no weights -- pure
layer allocation. Allowed to find nothing (current 0.5 may be optimal).

Run: /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/probe_ot_layer_frac.py
"""
import os, sys
from pathlib import Path
from collections import defaultdict
import statistics as st
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import eval_all as E

PROMPTS = {"kp0": "A panda eat bamboo", "kp2": "A sports car on a mountain road"}
STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]   # ablation-8 subset
FRACS = ["0.35", "0.5", "0.65"]                              # 0.5 = current default
OUT = Path(E.CWD) / "d_grand" / "olf_probe"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["EVAL_SEED"] = "43"

rows = []
for frac in FRACS:
    os.environ["OT_LAYER_FRAC"] = frac
    for sid in STYLES:
        sty = f"{E.CWD}/sty/{sid}.png"
        for pk, prompt in PROMPTS.items():
            fp = OUT / f"f{frac}_{sid}_{pk}.png"
            E.torch.cuda.empty_cache()
            E.generate_t2i(prompt, sty, E.HYBRID_DEFAULT).save(fp)
            csd = E.compute_csd(str(fp), sty)
            ct = E.compute_clip_t(str(fp), prompt)
            rows.append((frac, csd, ct))
            print(f"frac={frac} s{sid} {pk}: CSD={csd:.4f} CLIP-T={ct:.4f}", flush=True)

ag = defaultdict(lambda: [[], []])
for frac, c, t in rows:
    ag[frac][0].append(c); ag[frac][1].append(t)
print("\n=== SUMMARY (vs default 0.5) ===", flush=True)
for f in FRACS:
    c, t = ag[f]
    print(f"OT_LAYER_FRAC={f}: CSD={st.mean(c):.4f}  CLIP-T={st.mean(t):.4f}  n={len(c)}", flush=True)
print("OLF_PROBE_DONE", flush=True)
