"""
TASK #3 — Multi-seed significance (T2I).

Reuses eval_all's generate_t2i / metrics with a single Klein-9B load.
Runs the T2I subset (hybrid2 + adam8) for the requested EVAL_SEED values
(default 44 and 45 — seed 43 already lives in results/t2i_ablation.csv),
saving results/t2i_seed<S>.csv (method,sty_id,content_id,csd,clip_t).

Then computes, across seeds {43,44,45}:
  - per-method mean+/-std of CSD and CLIP-T (both per-seed means and the
    across-seed mean+/-std),
  - paired t-test (scipy.stats.ttest_rel) hybrid2-vs-adam8 on per-(style,prompt)
    CSD, pooled over (style,prompt,seed) and averaged-per-(style,prompt).
Saves results/multiseed_summary.json and prints a clean table.

Usage:
  /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/run_seed.py [seed ...]
"""
import os, sys, csv, json, time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from scipy.stats import ttest_rel

# Importing eval_all triggers the (single) Klein-9B model load.
import eval_all as E

METHODS = {"hybrid2": E.HYBRID_DEFAULT, "adam8": E.ADAM8_DEFAULT}
SEED_CSV = lambda s: E.RESULTS_DIR / f"t2i_seed{s}.csv"
SUMMARY_JSON = E.RESULTS_DIR / "multiseed_summary.json"


def run_seed(seed):
    """Generate hybrid2 + adam8 over STYLES_6 x T2I_PROMPTS_30 at EVAL_SEED=seed."""
    out_dir = Path(E.CWD) / "d_grand" / f"t2i_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EVAL_SEED"] = str(seed)  # read inside generate_t2i per-call

    rows = []
    n_total = len(METHODS) * len(E.T2I_PROMPTS_30) * len(E.STYLES_6)
    print(f"\n{'='*80}\nSEED {seed}: {len(METHODS)} methods x "
          f"{len(E.T2I_PROMPTS_30)} prompts x {len(E.STYLES_6)} styles = {n_total} images")
    print(f"{'Method':<10} {'Prompt':<33} {'Sty':>3} {'CSD':>7} {'CLIP-T':>7} {'Time':>6}")
    print("-" * 72)
    for method_name, config in METHODS.items():
        for pid, prompt in E.T2I_PROMPTS_30:
            for sty_id in E.STYLES_6:
                sty_path = f"{E.CWD}/sty/{sty_id}.png"
                E.torch.cuda.empty_cache()
                t0 = time.time()
                img = E.generate_t2i(prompt, sty_path, config)
                elapsed = time.time() - t0
                fname = f"{method_name}_{pid}_s{sty_id}.png"
                img.save(out_dir / fname)
                csd = E.compute_csd(str(out_dir / fname), sty_path)
                clip_t = E.compute_clip_t(str(out_dir / fname), prompt)
                rows.append({"method": method_name, "sty_id": sty_id, "content_id": pid,
                             "csd": f"{csd:.4f}", "clip_t": f"{clip_t:.4f}"})
                print(f"{method_name:<10} {prompt:<33} {sty_id:>3} "
                      f"{csd:>7.4f} {clip_t:>7.4f} {elapsed:>6.1f}")
                sys.stdout.flush()

    with open(SEED_CSV(seed), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "sty_id", "content_id", "csd", "clip_t"])
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {SEED_CSV(seed)}")
    return rows


def _read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _collect():
    """Return {seed: {method: {(sty,pid): {csd, clip_t}}}} for seeds 43/44/45."""
    data = {}
    # seed 43 from the existing ablation CSV
    abl = _read_rows(E.RESULTS_DIR / "t2i_ablation.csv")
    data[43] = {m: {} for m in METHODS}
    for r in abl:
        if r["method"] in METHODS:
            data[43][r["method"]][(r["sty_id"], r["content_id"])] = {
                "csd": float(r["csd"]), "clip_t": float(r["clip_t"])}
    for seed in (44, 45):
        rows = _read_rows(SEED_CSV(seed))
        data[seed] = {m: {} for m in METHODS}
        for r in rows:
            data[seed][r["method"]][(r["sty_id"], r["content_id"])] = {
                "csd": float(r["csd"]), "clip_t": float(r["clip_t"])}
    return data


def summarize():
    seeds = [43, 44, 45]
    data = _collect()
    summary = {"seeds": seeds, "per_method": {}, "paired_ttest_csd_hybrid_vs_adam": {}}

    # per-method mean/std
    for m in METHODS:
        per_seed_csd_mean, per_seed_clip_mean = [], []
        all_csd, all_clip = [], []
        for s in seeds:
            csd = [v["csd"] for v in data[s][m].values()]
            clip = [v["clip_t"] for v in data[s][m].values()]
            per_seed_csd_mean.append(float(np.mean(csd)))
            per_seed_clip_mean.append(float(np.mean(clip)))
            all_csd += csd; all_clip += clip
        summary["per_method"][m] = {
            "n_per_seed": len(data[seeds[0]][m]),
            "per_seed_csd_mean": {str(s): per_seed_csd_mean[i] for i, s in enumerate(seeds)},
            "per_seed_clip_t_mean": {str(s): per_seed_clip_mean[i] for i, s in enumerate(seeds)},
            # headline: mean +/- std ACROSS the 3 seed-level means (seed sensitivity)
            "csd_mean_over_seeds": float(np.mean(per_seed_csd_mean)),
            "csd_std_over_seeds": float(np.std(per_seed_csd_mean, ddof=1)),
            "clip_t_mean_over_seeds": float(np.mean(per_seed_clip_mean)),
            "clip_t_std_over_seeds": float(np.std(per_seed_clip_mean, ddof=1)),
            # pooled (all pairs x seeds) mean +/- std
            "csd_mean_pooled": float(np.mean(all_csd)),
            "csd_std_pooled": float(np.std(all_csd, ddof=1)),
            "clip_t_mean_pooled": float(np.mean(all_clip)),
            "clip_t_std_pooled": float(np.std(all_clip, ddof=1)),
        }

    # paired t-test on CSD: hybrid2 vs adam8, aligned by (sty,pid,seed)
    keys = sorted(set(data[43]["hybrid2"]) & set(data[43]["adam8"]))
    h_pool, a_pool = [], []
    for s in seeds:
        for k in keys:
            if k in data[s]["hybrid2"] and k in data[s]["adam8"]:
                h_pool.append(data[s]["hybrid2"][k]["csd"])
                a_pool.append(data[s]["adam8"][k]["csd"])
    h_pool, a_pool = np.array(h_pool), np.array(a_pool)
    t_p, p_p = ttest_rel(h_pool, a_pool)
    summary["paired_ttest_csd_hybrid_vs_adam"]["pooled_style_prompt_seed"] = {
        "n": int(len(h_pool)), "t": float(t_p), "p": float(p_p),
        "mean_diff_hybrid_minus_adam": float(np.mean(h_pool - a_pool))}

    # averaged over seeds -> one value per (sty,pid)
    h_avg, a_avg = [], []
    for k in keys:
        hv = [data[s]["hybrid2"][k]["csd"] for s in seeds if k in data[s]["hybrid2"]]
        av = [data[s]["adam8"][k]["csd"] for s in seeds if k in data[s]["adam8"]]
        if hv and av:
            h_avg.append(np.mean(hv)); a_avg.append(np.mean(av))
    h_avg, a_avg = np.array(h_avg), np.array(a_avg)
    t_a, p_a = ttest_rel(h_avg, a_avg)
    summary["paired_ttest_csd_hybrid_vs_adam"]["averaged_per_style_prompt"] = {
        "n": int(len(h_avg)), "t": float(t_a), "p": float(p_a),
        "mean_diff_hybrid_minus_adam": float(np.mean(h_avg - a_avg))}

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # pretty print
    print(f"\n{'='*80}\nMULTI-SEED SUMMARY (seeds {seeds})\n{'='*80}")
    print(f"{'Method':<10} {'CSD (mean+/-std over seeds)':<30} {'CLIP-T (mean+/-std over seeds)':<32}")
    for m in METHODS:
        pm = summary["per_method"][m]
        print(f"{m:<10} {pm['csd_mean_over_seeds']:.4f} +/- {pm['csd_std_over_seeds']:.4f}"
              f"{'':<13} {pm['clip_t_mean_over_seeds']:.4f} +/- {pm['clip_t_std_over_seeds']:.4f}")
    print("\nper-seed CSD means:")
    for m in METHODS:
        print(f"  {m:<10} " + "  ".join(
            f"s{s}={summary['per_method'][m]['per_seed_csd_mean'][str(s)]:.4f}" for s in seeds))
    pt = summary["paired_ttest_csd_hybrid_vs_adam"]
    print("\nPaired t-test (CSD, hybrid2 - adam8):")
    print(f"  pooled (style,prompt,seed) n={pt['pooled_style_prompt_seed']['n']}: "
          f"t={pt['pooled_style_prompt_seed']['t']:.3f}  p={pt['pooled_style_prompt_seed']['p']:.3e}  "
          f"mean_diff={pt['pooled_style_prompt_seed']['mean_diff_hybrid_minus_adam']:+.4f}")
    print(f"  averaged per (style,prompt) n={pt['averaged_per_style_prompt']['n']}: "
          f"t={pt['averaged_per_style_prompt']['t']:.3f}  p={pt['averaged_per_style_prompt']['p']:.3e}  "
          f"mean_diff={pt['averaged_per_style_prompt']['mean_diff_hybrid_minus_adam']:+.4f}")
    print(f"\nSaved: {SUMMARY_JSON}")
    return summary


def main(seeds=(44, 45)):
    for s in seeds:
        if SEED_CSV(s).exists():
            print(f"[skip] {SEED_CSV(s)} exists — reusing.")
            continue
        run_seed(s)
    summarize()


if __name__ == "__main__":
    arg_seeds = [int(x) for x in sys.argv[1:]] or [44, 45]
    main(tuple(arg_seeds))
