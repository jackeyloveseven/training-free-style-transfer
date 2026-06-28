"""Full-benchmark multi-seed (T2I): hybrid2 + adam8 over STYLES_ALL x T2I_PROMPTS_FULL
at seeds 44, 45 (seed 43 = existing results/t2i_full_benchmark.csv).
Writes results/t2i_full_seed<S>.csv and results/multiseed_full_summary.json.
Does NOT touch t2i_full_benchmark.csv (canonical seed-43).

Run with: /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/run_seed_full.py [seed ...]
"""
import os, sys, csv, json, time
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import eval_all as E   # triggers single Klein-9B load

METHODS = {"hybrid2": E.HYBRID_DEFAULT, "adam8": E.ADAM8_DEFAULT}
SEED_CSV = lambda s: E.RESULTS_DIR / f"t2i_full_seed{s}.csv"
SUMMARY_JSON = E.RESULTS_DIR / "multiseed_full_summary.json"


def run_seed(seed):
    out_dir = Path(E.CWD) / "d_grand" / f"t2i_full_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EVAL_SEED"] = str(seed)
    rows = []
    n_total = len(METHODS) * len(E.T2I_PROMPTS_FULL) * len(E.STYLES_ALL)
    print(f"\n{'='*80}\nFULL SEED {seed}: {n_total} images", flush=True)
    done = 0
    for method_name, config in METHODS.items():
        for pid, prompt in E.T2I_PROMPTS_FULL:
            for sty_id in E.STYLES_ALL:
                sty_path = f"{E.CWD}/sty/{sty_id}.png"
                E.torch.cuda.empty_cache()
                fname = f"{method_name}_{pid}_s{sty_id}.png"
                fp = out_dir / fname
                if not fp.exists():
                    img = E.generate_t2i(prompt, sty_path, config)
                    img.save(fp)
                csd = E.compute_csd(str(fp), sty_path)
                clip_t = E.compute_clip_t(str(fp), prompt)
                rows.append({"method": method_name, "sty_id": sty_id, "content_id": pid,
                             "csd": f"{csd:.4f}", "clip_t": f"{clip_t:.4f}"})
                done += 1
                if done % 20 == 0:
                    print(f"  [{done}/{n_total}] {method_name} {pid} s{sty_id} CSD={csd:.4f}", flush=True)
    with open(SEED_CSV(seed), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "sty_id", "content_id", "csd", "clip_t"])
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {SEED_CSV(seed)}", flush=True)


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def summarize():
    seeds = [43, 44, 45]
    data = {s: {m: {} for m in METHODS} for s in seeds}
    # seed 43 from canonical full benchmark
    for r in _rows(E.RESULTS_DIR / "t2i_full_benchmark.csv"):
        if r["method"] in METHODS:
            data[43][r["method"]][(r["sty_id"], r["content_id"])] = float(r["csd"]), float(r["clip_t"])
    for s in (44, 45):
        for r in _rows(SEED_CSV(s)):
            data[s][r["method"]][(r["sty_id"], r["content_id"])] = float(r["csd"]), float(r["clip_t"])
    summary = {"seeds": seeds, "per_method": {}}
    for m in METHODS:
        per_seed_csd = [float(np.mean([v[0] for v in data[s][m].values()])) for s in seeds]
        per_seed_clip = [float(np.mean([v[1] for v in data[s][m].values()])) for s in seeds]
        summary["per_method"][m] = {
            "n_per_seed": len(data[seeds[0]][m]),
            "per_seed_csd_mean": {str(s): per_seed_csd[i] for i, s in enumerate(seeds)},
            "csd_mean_over_seeds": float(np.mean(per_seed_csd)),
            "csd_std_over_seeds": float(np.std(per_seed_csd, ddof=1)),
            "clip_t_mean_over_seeds": float(np.mean(per_seed_clip)),
            "clip_t_std_over_seeds": float(np.std(per_seed_clip, ddof=1)),
        }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*80}\nFULL-BENCHMARK MULTI-SEED SUMMARY (seeds {seeds})\n{'='*80}")
    for m in METHODS:
        pm = summary["per_method"][m]
        print(f"  {m:<8} CSD {pm['csd_mean_over_seeds']:.4f} +/- {pm['csd_std_over_seeds']:.4f}"
              f"   CLIP-T {pm['clip_t_mean_over_seeds']:.4f} +/- {pm['clip_t_std_over_seeds']:.4f}"
              f"   (per-seed CSD {[round(v,4) for v in pm['per_seed_csd_mean'].values()]})")
    print(f"Saved: {SUMMARY_JSON}\nFULL_MULTISEED_DONE", flush=True)


def main(seeds=(44, 45)):
    for s in seeds:
        run_seed(s)
    summarize()


if __name__ == "__main__":
    arg = [int(x) for x in sys.argv[1:]] or [44, 45]
    main(tuple(arg))
