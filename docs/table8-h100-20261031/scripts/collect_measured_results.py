#!/usr/bin/env python3
"""Validate and collect completed Table 8 measurements without loading a GPU library.

Only completed, verified seven-configuration runs produce a CSV and TeX table.
This checks recorded results and image integrity; it does not recompute learned
metrics from pixels or establish identity with the historical published run.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys


CONFIGS = [(25, 2), (15, 2), (20, 2), (35, 2), (4, 2), (4, 4), (8, 2)]
STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]
PROMPTS = [
    ("kp0", "A panda eat bamboo"),
    ("kp1", "A castle on a cliff by the sea"),
    ("kp2", "A cat sitting on a windowsill"),
    ("kp3", "A field of sunflowers at sunset"),
    ("kp4", "A portrait of an old fisherman"),
]
PAIRS = {(pid, sty) for pid, _ in PROMPTS for sty in STYLES}
HYBRID = {"mode": "hybrid", "lr": 0.03, "content_w": 1.5,
          "style_w": 2.0, "slt_alpha": 0.1, "freq_hi_boost": 2.0,
          "sinkhorn_mode": "freq", "grag_deep": 0.5, "grag_shallow": 0.7}
CHECK_TOLERANCE = {"rel_tol": 1e-12, "abs_tol": 1e-12}
AUDITED_TIMING_DEFINITION = (
    "Synchronized generation with cached text embeddings on fixed kp0/sty27: "
    "three warmup runs followed by five timed runs; the table uses their median. "
    "Includes style-image VAE encoding, denoising with SLT/Adam optimization, and final VAE decoding. "
    "Excludes model startup, uncached text encoding, image saving, and learned-metric evaluation."
)


class InvalidRun(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidRun(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    require(path.is_file(), f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda x: (_ for _ in ()).throw(InvalidRun(f"Non-finite JSON: {path}: {x}")))


def read_jsonl(path):
    require(path.is_file(), f"Missing file: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(InvalidRun(f"Non-finite JSON: {x}")))
        except (ValueError, TypeError) as exc:
            raise InvalidRun(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        require(isinstance(row, dict), f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def number(value, context, positive=False, cosine=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"Not numeric: {context}")
    value = float(value)
    require(math.isfinite(value), f"Non-finite value: {context}")
    require(not positive or value > 0, f"Non-positive value: {context}")
    require(not cosine or -1 <= value <= 1, f"Outside cosine range [-1,1]: {context}")
    return value


def same_number(recorded, calculated, context):
    recorded = number(recorded, context)
    require(math.isclose(recorded, calculated, **CHECK_TOLERANCE),
            f"Stored arithmetic mismatch: {context}: stored={recorded}, recomputed={calculated}")


def receipt(path):
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def validate_start(run, mode):
    path = run / "RUN_STARTED.json"
    started = read_json(path)
    require(started.get("mode") == mode, f"Run mode must be {mode}: {path}")
    hardware = started.get("hardware", {})
    gpu = hardware.get("nvidia_smi")
    require(isinstance(gpu, list) and len(gpu) >= 2, f"Missing nvidia-smi hardware row: {path}")
    require(re.search(r"\bH100\b", str(gpu[1]), re.I) is not None,
            f"Expected observed H100 hardware; recorded name={gpu[1]!r}")
    require(str(gpu[0]).strip() == hardware.get("uuid"), f"GPU UUID mismatch: {path}")
    require(hardware.get("foreign_compute_processes") == [], f"Initial GPU guard did not record an idle GPU: {path}")
    return {"receipt": receipt(path), "hardware": hardware, "time_unix": started.get("time_unix")}


def validate_record(row, run, expected_configs):
    T, N = row.get("T"), row.get("N")
    require(type(T) is int and type(N) is int and (T, N) in expected_configs,
            f"Unexpected T,N: {T!r},{N!r}")
    require(type(row.get("seed")) is int and row["seed"] == 43, f"Unexpected sample seed: {row}")
    pid, sty = row.get("prompt_id"), row.get("style_id")
    require((pid, sty) in PAIRS, f"Unexpected prompt/style pair: {pid!r}/{sty!r}")
    require(row.get("prompt") == dict(PROMPTS)[pid], f"Prompt text differs from frozen protocol: {pid}/{sty}")
    number(row.get("csd"), f"T{T}_N{N}/{pid}/s{sty} CSD", cosine=True)
    number(row.get("clip_t"), f"T{T}_N{N}/{pid}/s{sty} CLIP-T", cosine=True)
    number(row.get("sample_generation_s"), f"T{T}_N{N}/{pid}/s{sty} generation time", positive=True)
    require(row.get("sample_timing_warmed") is False, "Unexpected sample timing warmup flag")
    label = f"T{T}_N{N}"
    filename = f"{pid}_s{sty}.png"
    remote_path = Path(row.get("image", ""))
    require(remote_path.is_absolute() and remote_path.name == filename and remote_path.parent.name == label,
            f"Recorded remote image path does not match sample identity: {remote_path}")
    local_path = run / label / filename
    require(local_path.is_file(), f"Missing image: {local_path}")
    expected_sha = row.get("image_sha256")
    require(isinstance(expected_sha, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha) is not None,
            f"Invalid image SHA256: {local_path}")
    require(sha256(local_path) == expected_sha, f"Image SHA256 mismatch: {local_path}")
    with local_path.open("rb") as stream:
        header = stream.read(24)
    require(len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR",
            f"Not a PNG with an IHDR header: {local_path}")
    require(struct.unpack(">II", header[16:24]) == (512, 512), f"Unexpected image dimensions: {local_path}")
    return (T, N, pid, sty), {"path": str(local_path.resolve()), "sha256": expected_sha}


def scheduler_check(T, summary, config, label):
    ts = summary.get("scheduler_timesteps")
    require(isinstance(ts, list) and len(ts) == T, f"Scheduler length differs from T: {label}")
    values = [number(t, f"{label} scheduler timestep") for t in ts]
    active = [i for i, t in enumerate(values) if t < 850]
    require(summary.get("active_step_count") == len(active), f"Incorrect active step count: {label}")
    require(config.get("scheduler_timesteps") == ts, f"Config/summary scheduler differs: {label}")
    require(config.get("active_step_indices_t_lt_850") == active, f"Incorrect active step indices: {label}")
    require(config.get("active_step_count") == len(active), f"Config active step count differs: {label}")
    return len(active)


def validate_full(run):
    start = validate_start(run, "full")
    rows = read_jsonl(run / "samples.jsonl")
    require(len(rows) == 280, f"Expected 280 sample records, found {len(rows)}")
    indexed, images = {}, []
    for row in rows:
        key, image = validate_record(row, run, CONFIGS)
        require(key not in indexed, f"Duplicate sample: {key}")
        indexed[key] = row
        images.append(image)
    summary = read_json(run / "summary.json")
    summaries = summary.get("configs")
    require(isinstance(summaries, list) and len(summaries) == 7, "Expected seven stored configuration summaries")
    by_config = {(r.get("T"), r.get("N")): r for r in summaries}
    require(len(by_config) == 7 and set(by_config) == set(CONFIGS), "Stored configuration set is incomplete or duplicated")
    require(summary.get("complete_configs") == 7 and summary.get("sample_count") == 280,
            "Top-level stored completion counts differ")
    require(summary.get("numerical_results_complete") is True, "Runner did not mark numerical results complete")
    require(summary.get("gpu_validated") is True, "Runner GPU validation flag is missing")
    require(isinstance(summary.get("timing_definition"), str) and summary["timing_definition"], "Missing timing definition")
    aggregate = []
    for T, N in CONFIGS:
        label = f"T{T}_N{N}"
        sample_rows = {key[2:]: row for key, row in indexed.items() if key[:2] == (T, N)}
        require(set(sample_rows) == PAIRS and len(sample_rows) == 40, f"Incomplete pair set: {label}")
        require({p.name for p in (run / label).glob("*.png")} == {f"{pid}_s{sty}.png" for pid, sty in PAIRS},
                f"PNG file set differs from protocol: {label}")
        config = read_json(run / label / "config.json")
        require(config.get("seed") == 43 and config.get("KSHOT") == 1 and config.get("OT_LAYER_FRAC") == 0.5,
                f"Stored config protocol differs: {label}")
        require(config.get("styles") == STYLES and config.get("prompts") == [list(p) for p in PROMPTS],
                f"Config styles/prompts differ: {label}")
        require(config.get("config") == dict(HYBRID, num_steps=T, iters=N), f"Generation hyperparameters differ: {label}")
        stored = by_config[(T, N)]
        require(stored.get("n") == 40 and stored.get("seed_variance_measured") is False,
                f"Stored sample count/seed-variance flag differs: {label}")
        local_summary = read_json(run / label / "summary.json")
        require(local_summary == stored, f"Local/top-level summaries differ: {label}")
        values = {"T": T, "N": N, "n": 40, "seed": 43, "gpu": str(start["hardware"]["nvidia_smi"][1]).strip()}
        for metric in ("csd", "clip_t"):
            xs = [r[metric] for r in sample_rows.values()]
            mean, sd = statistics.mean(xs), statistics.stdev(xs)
            same_number(stored.get(f"mean_{metric}"), mean, f"{label} mean {metric}")
            same_number(stored.get(f"sd_across_samples_{metric}"), sd, f"{label} sample SD {metric}")
            values[f"mean_{metric}"] = mean
            values[f"sd_across_samples_{metric}"] = sd
        require(stored.get("warmup_runs") == 3 and stored.get("timed_runs") == 5,
                f"Expected three warmups and five timed runs: {label}")
        require(stored.get("timing_prompt_id") == "kp0" and stored.get("timing_style_id") == "27",
                f"Timing pair differs from kp0/sty27: {label}")
        times = stored.get("inference_times_s")
        require(isinstance(times, list) and len(times) == 5, f"Expected five raw times: {label}")
        times = [number(t, f"{label} fixed-pair timing", positive=True) for t in times]
        median = statistics.median(times)
        same_number(stored.get("inference_median_s"), median, f"{label} timing median")
        values.update({"active_step_count": scheduler_check(T, stored, config, label),
                       "inference_median_s": median, "inference_times_s": times,
                       "warmup_runs": 3, "timed_runs": 5})
        aggregate.append(values)
    return aggregate, indexed, {"start": start, "source_files": [receipt(run / p) for p in ("samples.jsonl", "summary.json")],
                                "verified_images": images, "timing_definition": summary["timing_definition"]}


def baseline_comparison(path, indexed):
    require(path.is_file(), f"Historical baseline CSV missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        old = [r for r in csv.DictReader(stream) if r.get("method") == "hybrid2"]
    require(len(old) == 40, "Historical hybrid2 group must contain exactly 40 rows")
    paired = {}
    for row in old:
        key = row["content_id"], row["sty_id"]
        require(key not in paired, f"Duplicate historical pair: {key}")
        paired[key] = row
    require(set(paired) == PAIRS, "Historical baseline pair set differs from protocol")
    records = []
    for pid, sty in sorted(PAIRS):
        new, old = indexed[(25, 2, pid, sty)], paired[(pid, sty)]
        record = {"prompt_id": pid, "style_id": sty}
        for metric in ("csd", "clip_t"):
            old_value = number(float(old[metric]), f"Historical {pid}/{sty} {metric}", cosine=True)
            record.update({f"old_{metric}": old_value, f"new_{metric}": new[metric],
                           f"delta_{metric}": new[metric] - old_value})
        records.append(record)
    descriptive = {}
    for metric in ("csd", "clip_t"):
        errors = [r[f"delta_{metric}"] for r in records]
        descriptive[metric] = {"old_mean": statistics.mean(r[f"old_{metric}"] for r in records),
                               "new_mean": statistics.mean(r[f"new_{metric}"] for r in records),
                               "mean_signed_delta": statistics.mean(errors),
                               "mean_absolute_delta": statistics.mean(abs(x) for x in errors),
                               "root_mean_squared_delta": math.sqrt(statistics.mean(x * x for x in errors)),
                               "max_absolute_delta": max(abs(x) for x in errors)}
    return {"source": receipt(path), "method": "hybrid2", "n": 40,
            "interpretation": "Descriptive paired differences only. No automatic scientific agreement threshold or source-cause attribution is applied.",
            "published_baseline_identity_verified": False, "descriptive_errors": descriptive, "per_pair": records}


def smoke_comparison(smoke, indexed):
    start = validate_start(smoke, "smoke")
    rows = read_jsonl(smoke / "samples.jsonl")
    require(len(rows) == 1, "Smoke run must contain exactly one sample")
    row = rows[0]
    key, image = validate_record(row, smoke, [(25, 2)])
    require(key == (25, 2, "kp0", "00"), "Smoke pair must be T25/N2/kp0/sty00")
    stored = read_json(smoke / "smoke.json")
    require(stored.get("record") == row and stored.get("gpu_validated") is True and stored.get("table_ready") is False,
            "Smoke receipt differs from its sample record")
    full = indexed[key]
    differences = {metric: full[metric] - row[metric] for metric in ("csd", "clip_t")}
    return {"start": start, "source": receipt(smoke / "smoke.json"), "image": image,
            "same_image_sha256": row["image_sha256"] == full["image_sha256"],
            "same_metrics_exactly": all(delta == 0 for delta in differences.values()),
            "full_minus_smoke": differences,
            "interpretation": "Exact image and metric comparison is recorded separately; no tolerance-based quality equivalence is inferred."}


def write_atomic(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def table_tex(aggregate):
    row_end = chr(92) * 2
    lines = [
        "% Generated from measured_results_audit.json after structural and arithmetic validation.",
        "% Independent H100 reproduction; historical published baseline identity remains unverified.",
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Hyperion denoising-step sweep on one H100 GPU. Each quality mean uses the same 40 pairs (8 styles $\\times$ 5 prompts), seed 43. Active steps satisfy $t<850$. Time is the median of five generations with cached text embeddings on one fixed pair (kp0, style 27), following three warmup runs. Style-image VAE encoding, denoising with optimization, and final VAE decoding are included; model startup, uncached text encoding, image saving, and metric evaluation are excluded.}",
        "\\begin{tabular}{rrrrrr}", "\\toprule",
        "$T$ & $N$ & Active steps & CSD $\\uparrow$ & CLIP-T $\\uparrow$ & Time (s) $\\downarrow$ " + row_end,
        "\\midrule",
    ]
    for row in aggregate:
        lines.append(f"{row['T']} & {row['N']} & {row['active_step_count']} & {row['mean_csd']:.4f} & {row['mean_clip_t']:.4f} & {row['inference_median_s']:.1f} " + row_end)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Downloaded completed full_001 directory")
    parser.add_argument("--smoke", type=Path, help="Downloaded smoke_001 directory")
    parser.add_argument("--baseline", type=Path, default=Path(__file__).resolve().parent / "code/results/t2i_ablation.csv")
    parser.add_argument("--assets", type=Path, help="Optional model_asset_verification.json receipt, kept separate from result validation")
    parser.add_argument("--output", type=Path, help="Output directory; defaults to --run")
    args = parser.parse_args()
    run = args.run.expanduser().resolve()
    output = (args.output or run).expanduser().resolve()
    try:
        aggregate, indexed, evidence = validate_full(run)
        baseline = baseline_comparison(args.baseline.expanduser().resolve(), indexed)
        smoke = smoke_comparison(args.smoke.expanduser().resolve(), indexed) if args.smoke else None
        asset_path = args.assets.expanduser().resolve() if args.assets else next(
            (p for p in [run / "model_asset_verification.json", run.parent / "model_asset_verification.json",
                         run.parent / "logs/model_asset_verification.json"] if p.is_file()), None)
        assets = {"provided": asset_path is not None, "validated_independently_by_collector": False}
        if asset_path:
            assets.update({"source": receipt(asset_path), "recorded_receipt": read_json(asset_path)})
        audit = {"analysis_validation": True, "analysis_validation_scope": "Recorded sample-set completeness, frozen prompt/config fields, H100 start receipt, image SHA256/dimensions, scheduler counts, and aggregate arithmetic. Learned image metrics are not recomputed by this collector.",
                 "paper_ready": False, "published_baseline_identity_verified": False,
                 "run_directory": str(run), "sample_count": 280, "configuration_count": 7,
                 "collector": receipt(Path(__file__).resolve()),
                 "arithmetic_comparison_tolerance": CHECK_TOLERANCE,
                 "seed": 43, "seed_variance_measured": False,
                 "standard_deviation_definition": "Sample standard deviation across the 40 different prompt/style pairs (ddof=1); not variation across random seeds.",
                 "configurations": aggregate, "run_evidence": evidence,
                 "audited_timing_definition": AUDITED_TIMING_DEFINITION,
                 "timing_definition_correction": {
                     "raw_recorded_timing_definition": evidence["timing_definition"],
                     "raw_record_retained_unchanged": True,
                     "explanation": "The raw string describes the generate_t2i call boundary broadly. encode_prompt caches embeddings by prompt text; all five prompts have been used before the fixed-pair timing loop, so these timed calls do not perform uncached text encoding.",
                     "code_evidence": ["code/scripts/eval_all.py:491-508 encode_prompt cache", "code/scripts/eval_all.py:607-622 style VAE encoding followed by encode_prompt", "run_table8.py:234-279 complete quality samples before three warmups and five timed kp0/sty27 calls"],
                     "generation_or_metric_math_changed": False,
                 },
                 "historical_baseline_comparison": baseline, "smoke_comparison": smoke,
                 "model_assets_metadata": assets,
                 "review_required": ["Review measured baseline differences and model/runtime provenance before manuscript use.",
                                     "H100 timings cannot replace or scale historical H200 timings without a separate hardware comparison.",
                                     "These measurements vary scheduler T on the same checkpoint; they do not compare separate distillation methods or checkpoints."]}
        output.mkdir(parents=True, exist_ok=True)
        csv_buffer = io.StringIO(newline="")
        fields = ["T", "N", "active_step_count", "n", "seed", "gpu", "mean_csd", "sd_across_samples_csd",
                  "mean_clip_t", "sd_across_samples_clip_t", "inference_median_s", "warmup_runs", "timed_runs"]
        writer = csv.DictWriter(csv_buffer, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(aggregate)
        write_atomic(output / "measured_results.csv", csv_buffer.getvalue())
        write_atomic(output / "table8_h100.tex", table_tex(aggregate))
        write_atomic(output / "measured_results_audit.json", json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"analysis_validation": True, "sample_count": 280,
                          "output_files": [str(output / p) for p in ("measured_results.csv", "measured_results_audit.json", "table8_h100.tex")]}, indent=2))
        return 0
    except (InvalidRun, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Validation failed; no result artifacts were written: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
