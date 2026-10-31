#!/usr/bin/env python3
"""Isolated Table 8 reproduction using the repository's unchanged generation core.

Preflight uses the standard library only. GPU execution requires an idle GPU.
The original repository is never edited. The adapted module and patch receipt
are saved beside every run so the path changes and T override are reviewable.
"""
import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]
PROMPTS = [
    ("kp0", "A panda eat bamboo"),
    ("kp1", "A castle on a cliff by the sea"),
    ("kp2", "A cat sitting on a windowsill"),
    ("kp3", "A field of sunflowers at sunset"),
    ("kp4", "A portrait of an old fisherman"),
]
CONFIGS = [(25, 2), (15, 2), (20, 2), (35, 2), (4, 2), (4, 4), (8, 2)]
HYBRID = {"mode": "hybrid", "iters": 2, "lr": 0.03,
          "content_w": 1.5, "style_w": 2.0, "slt_alpha": 0.1,
          "freq_hi_boost": 2.0, "sinkhorn_mode": "freq",
          "grag_deep": 0.5, "grag_shallow": 0.7}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def git_commit(repo):
    p = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def prepare_source(args):
    src = args.repo / "scripts/eval_all.py"
    tree = ast.parse(src.read_text(), filename=str(src))
    replacements = {
        "DIFFSYNTH_ROOT": ("third_party/DiffSynth-Studio", str(args.diffsynth_root)),
        "KLEIN9B_ROOT": ("models/klein9b", str(args.model_root)),
        "CWD": ("/home/ubuntu/chengyanli/items/save/attention_20260426_090948", str(args.output)),
        "csd_dir": ("../../CSD_Score", str(args.csd_root)),
    }
    seen, receipt = set(), []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in replacements:
                old, new = replacements[target.id]
                if not isinstance(node.value, ast.Constant) or node.value.value != old:
                    raise ValueError(f"Unexpected upstream assignment: {target.id}")
                node.value = ast.copy_location(ast.Constant(new), node.value)
                seen.add(target.id)
                receipt.append({"line": node.lineno, "assignment": target.id, "old": old, "new": new})
    if seen != set(replacements):
        raise ValueError(f"Missing path assignments: {set(replacements) - seen}")
    csd_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == "sys.path.insert" and len(n.args) == 2
                 and isinstance(n.args[1], ast.Constant) and n.args[1].value == "../../CSD_Score"]
    if len(csd_calls) != 1:
        raise ValueError("Expected one CSD sys.path insertion")
    csd_calls[0].args[1] = ast.Constant(str(args.csd_root))
    receipt.append({"line": csd_calls[0].lineno, "assignment": "CSD sys.path", "new": str(args.csd_root)})
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_t2i"]
    if len(fns) != 1:
        raise ValueError("Expected one generate_t2i")
    step_nodes = [n for n in ast.walk(fns[0]) if isinstance(n, ast.Assign)
                  and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id == "num_inference_steps"]
    if len(step_nodes) != 1 or ast.literal_eval(step_nodes[0].value) != 25:
        raise ValueError("Upstream T assignment changed; manual review required")
    step_nodes[0].value = ast.parse('int(config.get("num_steps", 25))', mode="eval").body
    receipt.append({"line": step_nodes[0].lineno, "assignment": "generate_t2i.num_inference_steps",
                    "old": "25", "new": 'int(config.get("num_steps", 25))'})
    ast.fix_missing_locations(tree)
    adapted = ast.unparse(tree) + "\n"
    compile(adapted, str(src), "exec")
    return src, adapted, receipt


def preflight(args):
    src, adapted, receipt = prepare_source(args)
    missing = []
    checks = [args.diffsynth_root / "diffsynth/pipelines/flux2_image.py",
              args.diffsynth_root / "diffsynth/diffusion/flow_match.py",
              args.model_root / "tokenizer", args.model_root / "vae/diffusion_pytorch_model.safetensors",
              args.csd_root / "model.py", args.csd_root / "models/ViT-L-14.pt",
              args.csd_root / "models/checkpoint.pth"]
    checks += [args.repo / f"sty/{s}.png" for s in STYLES]
    for p in checks:
        if not p.exists():
            missing.append(str(p))
    weights = []
    for kind in ("text_encoder", "transformer"):
        found = sorted((args.model_root / kind).glob("*.safetensors"))
        if not found:
            missing.append(str(args.model_root / kind / "*.safetensors"))
        weights += found
    vae = args.model_root / "vae/diffusion_pytorch_model.safetensors"
    if vae.exists():
        weights.append(vae)
    weights += [p for p in checks if p.is_file() and p.suffix in (".pth", ".pt")]
    source_info = {"path": str(src), "sha256": digest(src), "git_commit": git_commit(args.repo)}
    args.output.mkdir(parents=True, exist_ok=True)
    adapted_path = args.output / "eval_all.isolated.py"
    adapted_path.write_text(adapted)
    save_json(args.output / "source_patch.json", {"source": source_info, "edits": receipt,
                                                  "adapted_sha256": digest(adapted_path)})
    report = {"ok": not missing, "missing": missing, "gpu_validated": False,
              "source": source_info, "input_style_sha256": {
                  s: digest(args.repo / f"sty/{s}.png") for s in STYLES if (args.repo / f"sty/{s}.png").exists()},
              "weights_sha256_verified": False,
              "weight_inventory": [{"path": str(p), "bytes": p.stat().st_size} for p in weights],
              "dataset": {"styles": STYLES, "prompts": PROMPTS, "seed": 43},
              "requested_configs": CONFIGS,
              "baseline_reference": {
                  "source": "results/t2i_ablation.csv: hybrid2; scripts/eval_all.py",
                  "n": 40, "mean_csd": 0.680985, "mean_clip_t": 0.273725,
                  "alternative_source": "results/t2i_iters_sweep.csv: iters=2 uses another implementation",
                  "alternative_mean_csd": 0.6823375, "alternative_mean_clip_t": 0.2683975},
              "timing_definition": "same adapted eval_all.generate_t2i; synchronized warmed generation with cached text embeddings; includes style-image VAE encoding, denoising, Adam optimization and final VAE decoding; excludes model startup, uncached text encoding, image saving and metric evaluation; 3 warmup and 5 timed runs per config on kp0/sty27",
              "timing_comparable_to_old_json": False,
              "timing_caveat": "Old timing_benchmark.json used eval_supplement.py, a distinct generation implementation; old timings are not mixed with this run.",
              "runtime_environment_validated": False}
    save_json(args.output / "preflight.json", report)
    return report, adapted_path


def gpu_guard(selector):
    cmd = ["nvidia-smi", "-i", selector, "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu",
           "--format=csv,noheader,nounits"]
    gpu = subprocess.check_output(cmd, text=True).strip()
    rows = list(csv.reader(gpu.splitlines()))
    if len(rows) != 1:
        raise RuntimeError("Select exactly one GPU")
    uuid = rows[0][0].strip()
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                    "--format=csv,noheader,nounits"], text=True)
    foreign = [r for r in csv.reader(apps.splitlines()) if r and r[0].strip() == uuid
               and r[1].strip() != str(os.getpid())]
    if foreign:
        raise RuntimeError(f"GPU is occupied by other compute processes: {foreign}")
    return {"selector": selector, "uuid": uuid, "nvidia_smi": rows[0], "foreign_compute_processes": foreign}


def finite(value, name):
    x = float(value)
    if not math.isfinite(x):
        raise RuntimeError(f"Non-finite {name}")
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "diffsynth-root", "model-root", "csd-root", "output"):
        p.add_argument("--" + name, type=lambda s: Path(s).expanduser().resolve(), required=True)
    p.add_argument("--gpu", default="0", help="physical GPU index or UUID")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="standard-library path/source checks only")
    mode.add_argument("--smoke", action="store_true", help="one T25 N2 kp0/sty00 image; no table aggregate")
    args = p.parse_args()
    if (args.output / "RUN_STARTED.json").exists():
        raise RuntimeError("Output already contains a started run; choose a fresh directory")
    report, adapted_path = preflight(args)
    if args.preflight:
        print(json.dumps({"ok": report["ok"], "missing": report["missing"], "gpu_validated": False}, indent=2))
        return 0 if report["ok"] else 2
    if report["missing"]:
        raise RuntimeError("Preflight missing assets: " + ", ".join(report["missing"]))
    hardware = gpu_guard(args.gpu)
    os.environ.update({"CUDA_VISIBLE_DEVICES": hardware["uuid"], "DIFFSYNTH_ATTENTION_IMPLEMENTATION": "torch",
                       "EVAL_SEED": "43", "KSHOT": "1", "OT_LAYER_FRAC": "0.5"})
    save_json(args.output / "RUN_STARTED.json", {"time_unix": time.time(), "hardware": hardware,
                                                "mode": "smoke" if args.smoke else "full", "argv": sys.argv})
    runtime = {}
    for dist in ("torch", "torchvision", "transformers", "open-clip-torch", "diffsynth", "numpy", "Pillow"):
        try:
            runtime[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            runtime[dist] = None
    save_json(args.output / "runtime.json", {"python": sys.version, "packages": runtime,
                                             "diffsynth_git": git_commit(args.diffsynth_root), "hardware": hardware})
    sys.path[:0] = [str(args.repo), str(args.repo / "scripts"), str(args.csd_root)]
    module = types.ModuleType("table8_isolated_eval")
    module.__file__ = str(adapted_path)
    exec(compile(adapted_path.read_text(), str(adapted_path), "exec"), module.__dict__)
    E = module
    if E.HYBRID_DEFAULT != HYBRID or E.STYLES_6 != STYLES or E.T2I_PROMPTS_30 != PROMPTS:
        raise RuntimeError("Upstream protocol differs from the audited protocol")
    if not E.torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    raw_path = args.output / "samples.jsonl"
    configs = [(25, 2)] if args.smoke else CONFIGS
    summaries = []
    for T, N in configs:
        gpu_guard(args.gpu)
        cfg = dict(HYBRID, num_steps=T, iters=N)
        diagnostic_scheduler = E.FlowMatchScheduler("FLUX.2")
        diagnostic_scheduler.set_timesteps(T, dynamic_shift_len=1024)
        actual_timesteps = [float(t) for t in diagnostic_scheduler.timesteps]
        active_step_indices = [i for i, t in enumerate(actual_timesteps) if t < 850]
        label = f"T{T}_N{N}"
        dest = args.output / label
        dest.mkdir(exist_ok=False)
        save_json(dest / "config.json", {"config": cfg, "seed": 43, "KSHOT": 1, "OT_LAYER_FRAC": 0.5,
                                          "styles": STYLES, "prompts": PROMPTS,
                                          "scheduler_timesteps": actual_timesteps,
                                          "active_step_indices_t_lt_850": active_step_indices,
                                          "active_step_count": len(active_step_indices),
                                          "source_sha256": report["source"]["sha256"]})
        samples = [(PROMPTS[0][0], PROMPTS[0][1], STYLES[0])] if args.smoke else [
            (pid, prompt, sty) for pid, prompt in PROMPTS for sty in STYLES]
        records = []
        for pid, prompt, sty in samples:
            gpu_guard(args.gpu)
            E.torch.cuda.empty_cache()
            E.torch.cuda.synchronize()
            started = time.perf_counter()
            img = E.generate_t2i(prompt, str(args.repo / f"sty/{sty}.png"), cfg)
            E.torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            gpu_guard(args.gpu)
            image_path = dest / f"{pid}_s{sty}.png"
            img.save(image_path)
            row = {"T": T, "N": N, "seed": 43, "prompt_id": pid, "prompt": prompt, "style_id": sty,
                   "csd": finite(E.compute_csd(str(image_path), str(args.repo / f"sty/{sty}.png")), "csd"),
                   "clip_t": finite(E.compute_clip_t(str(image_path), prompt), "clip_t"),
                   "sample_generation_s": elapsed, "sample_timing_warmed": False,
                   "image": str(image_path), "image_sha256": digest(image_path)}
            with raw_path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            records.append(row)
            print(f"{label} {len(records)}/{len(samples)} {pid}/sty{sty} CSD={row['csd']:.4f} CLIP-T={row['clip_t']:.4f}", flush=True)
        if args.smoke:
            save_json(args.output / "smoke.json", {"gpu_validated": True, "record": records[0], "table_ready": False})
            print("Smoke complete; this single sample is not a table result.", flush=True)
            return 0
        actual = {(r["prompt_id"], r["style_id"]) for r in records}
        expected = {(pid, sty) for pid, _ in PROMPTS for sty in STYLES}
        if len(records) != 40 or actual != expected:
            raise RuntimeError(f"Incomplete or duplicate sample set: {label}")
        # The table's timing uses this exact generation function on one fixed pair.
        times = []
        for i in range(8):
            gpu_guard(args.gpu)
            E.torch.cuda.empty_cache()
            E.torch.cuda.synchronize()
            started = time.perf_counter()
            E.generate_t2i(PROMPTS[0][1], str(args.repo / "sty/27.png"), cfg)
            E.torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            gpu_guard(args.gpu)
            if i >= 3:
                times.append(elapsed)
        result = {"T": T, "N": N, "n": 40,
                  "mean_csd": statistics.mean(r["csd"] for r in records),
                  "mean_clip_t": statistics.mean(r["clip_t"] for r in records),
                  "sd_across_samples_csd": statistics.stdev(r["csd"] for r in records),
                  "sd_across_samples_clip_t": statistics.stdev(r["clip_t"] for r in records),
                  "inference_median_s": statistics.median(times), "inference_times_s": times,
                  "warmup_runs": 3, "timed_runs": 5, "timing_prompt_id": "kp0", "timing_style_id": "27",
                  "scheduler_timesteps": actual_timesteps,
                  "active_step_count": len(active_step_indices),
                  "seed_variance_measured": False}
        if T == 25 and N == 2:
            result["baseline_delta_vs_ablation"] = {
                "csd": result["mean_csd"] - report["baseline_reference"]["mean_csd"],
                "clip_t": result["mean_clip_t"] - report["baseline_reference"]["mean_clip_t"],
                "interpretation": "Descriptive comparison only. Historical CSV lacks frozen runtime/weight identity; investigate material differences before paper use."}
        summaries.append(result)
        save_json(dest / "summary.json", result)
        save_json(args.output / "partial_summary.json", {"complete_configs": summaries, "table_ready": False})
    save_json(args.output / "summary.json", {"configs": summaries, "complete_configs": 7,
                                              "sample_count": 280, "gpu_validated": True,
                                              "numerical_results_complete": True,
                                              "table_ready": False, "historical_baseline_requires_review": True,
                                              "release_note": "Results are complete; manuscript use requires review of baseline identity, model provenance and timing comparability.",
                                              "timing_definition": report["timing_definition"]})
    print("All seven configurations complete. Review baseline differences before manuscript use.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
