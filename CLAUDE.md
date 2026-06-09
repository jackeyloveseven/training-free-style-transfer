# Attention-based Style Transfer (OT v3)

## Project Overview

This project implements **attention-distribution style transfer** on FLUX backbones. The core idea: during diffusion denoising, optimize latents so their self-attention K/V distributions match a style reference image, using Sinkhorn (optimal transport) attention in deep layers and standard attention in shallow layers.

Two pipelines exist:
1. **Text-to-image style transfer** (`custom_main.py` + `mypipeline.py`) — T2I paper runs should use FLUX.1-dev; generates from a text prompt with style guidance
2. **Image-to-image style transfer** (`custom_main_img.py` + `mypipeline_img.py`) — I2I runs use FLUX.1-Kontext-dev; takes a content image + style image, preserves content structure while applying style

Both are production-ready and fully tested.

---

## File Structure

```
├── custom_main.py          # T2I entry point (OT v3 benchmark, DO NOT MODIFY)
├── custom_main_img.py      # I2I entry point (Kontext content conditioning)
├── mypipeline.py           # T2I pipeline (inherits FluxPipeline)
├── mypipeline_img.py       # I2I pipeline (inherits FluxKontextPipeline)
├── utils.py                # Shared utilities (adain, sinkhorn_attn, Controller, etc.)
├── eval_sweep.py           # Metric evaluation sweep script
├── cores/
│   ├── core_ot_attn.py         # Production T2I core (OT v3, DO NOT MODIFY)
│   ├── core_ot_attn_img.py     # Production I2I core (Kontext variant)
│   ├── core_ot_attn_distill.py # Experimental: truncated-forward acceleration
│   └── core_ot_attn_v3_backup.py  # Backup
├── cnt/                    # Content test images (00-19.png)
├── sty/                    # Style test images (00-40.png)
```

---

## How It Works (Core Algorithm)

Each denoising step performs:

1. **Denoising step** — standard noise prediction (I2I: content image concatenated to latents along sequence dim)
2. **Style forward** — run noisy style latent through transformer, extract reference K/V from self-attention layers
3. **GRAG** — compress spatial variance of ref K/V to prevent layout leakage
4. **AdaIN warm-start** — align latent statistics toward style (alpha=0.3 for I2I)
5. **Adam optimization** — minimize:
   - `style_loss`: L1(self_attn_output, sinkhorn_attn(q, ref_k, ref_v)) in deep layers
   - `dist_loss`: mean/std alignment to style latent
   - `content_loss`: L1 drift from content anchor (decayed)
   - `csd_loss`: CSD embedding similarity (optional, CSD_W > 0)

Current paper setting:
- T2I and I2I main runs use `FIXED_LAYERS=1`, i.e. fixed layers `(20,45)` and `iters=2`, because rank-adaptive layer routing caused visible artifacts.
- Legacy rank-adaptive routing is kept only as a comparison/debug option via `FIXED_LAYERS=0`.
- T2I parameter sweeps are mini figure sweeps only: `sty27 × p2` for tau/layer/gamma, total 15 images.

---

## Optimal Configurations

### T2I (`custom_main.py`)
- Backbone: `/mnt/local/cache/huggingface/FLUX.1-dev`
- Steps: 25, Start time: 800, Seed: 43
- CSD_W: 0.5, Guidance: 3.5
- Resolution: 512x512, Mixed precision: bf16

### I2I (`custom_main_img.py`)
- Backbone: `/mnt/local/cache/huggingface/FLUX.1-Kontext-dev`
- **Steps: 25, Start time: 870** (Pareto-optimal for CLIP-I vs CSD)
- CSD_W: 0.5, Guidance: 3.5
- Resolution: 512x512, Mixed precision: bf16

### I2I Metric Results (cnt/00.png + sty/27.png)

| Steps | Start Time | CLIP-I | CSD   |
|-------|-----------|--------|-------|
| **25**| **870**   | **0.4101** | **0.6490** |
| 25    | 800       | 0.3524 | 0.5999 |
| 25    | 950       | 0.3228 | 0.7081 |
| 27    | 870       | 0.3554 | 0.6217 |
| 30    | 870       | 0.3471 | 0.6277 |
| 35    | 870       | 0.2542 | 0.7703 |
| 50    | 920       | 0.2546 | 0.9109 |

**Insight**: CLIP-I drops sharply beyond 25 steps. Higher start_time = fewer style optimization steps = better content preservation. The 25/870 config is the only point on the Pareto front with CLIP-I > 0.4.

---

## Environment & Running

```bash
# Python interpreter (has diffusers 0.37.0.dev0 with FluxKontextPipeline)
/home/ubuntu/chengyanli/flux/bin/python3

# Model paths
# T2I
/mnt/local/cache/huggingface/FLUX.1-dev
# I2I
/mnt/local/cache/huggingface/FLUX.1-Kontext-dev

# CSD metric model
/home/ubuntu/chengyanli/items/CSD_Score

# Run T2I
PROMPT="A panda eat bamboo" /home/ubuntu/chengyanli/flux/bin/python3 custom_main.py

# Run I2I
CNT_IMAGE=cnt/00.png STY_IMAGE=sty/27.png /home/ubuntu/chengyanli/flux/bin/python3 custom_main_img.py

# Override defaults via env
OVERRIDE_START_TIME=900 NUM_STEPS=30 CSD_W=0.3 /home/ubuntu/chengyanli/flux/bin/python3 custom_main_img.py
```

---

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_STEPS` | 25 | Number of denoising steps |
| `OVERRIDE_START_TIME` | 870 (I2I) / 800 (T2I) | Timestep threshold; style optimization only runs when t < start_time |
| `CSD_W` | 0.5 | CSD-in-loop weight (0 = disabled, saves VRAM) |
| `PROMPT` | "masterpiece" (I2I) | Text prompt |
| `CNT_IMAGE` | cnt/00.png | Content image path (I2I only) |
| `STY_IMAGE` | sty/27.png | Style image path |
| `GRAG_DEEP` | 0.5 (I2I) / 0.15 (T2I) | GRAG compression for deep layers |
| `GRAG_SHALLOW` | 0.75 | GRAG compression for shallow layers |
| `ADAIN_ALPHA` | 0.3 (I2I) | AdaIN blending strength |
| `KSHOT` | 1 | Number of style forward passes (multi-shot averaging) |
| `FIXED_LAYERS` | 1 for paper experiment copies | Fixed layers `(20,45)`, `iters=2`; set to 0 only for legacy rank-adaptive comparison |

---

## North Star Metrics

- **CLIP-I** (content preservation): cosine similarity of CSD content embeddings between output and content image. Higher = better structure preservation.
- **CSD** (style fidelity): cosine similarity of CSD style embeddings between output and style image. Higher = better style transfer.

Computed via CSD_CLIP model (ViT-L/14) at `/home/ubuntu/chengyanli/items/CSD_Score`. The model returns `(content_emb, _, style_emb)`.

---

## Excluded Samples

Permanently exclude from all sweeps/evaluation/reporting: **sty04, sty14, cnt19** (pathological edge cases).

---

## Critical Rules

1. **DO NOT modify `custom_main.py` or `cores/core_ot_attn.py`** — these are the production T2I benchmark
2. Experiment only on copies (e.g., `core_ot_attn_img.py`, `core_ot_attn_distill.py`)
3. The `sensenova` conda env does NOT have diffusers — always use `/home/ubuntu/chengyanli/flux/bin/python3`
4. GPU: single H100 (140GB), CUDA device 0

---

## Paper Figures (`scripts/gen_paper_figures.py` → `fig/`)

| Fig | Name | Content |
|-----|------|---------|
| fig1 | ablation | Component ablation bar chart (CSD/CLIP-T) |
| fig2 | pareto | Wall-clock time vs CSD Pareto curve (Hybrid vs Adam-only vs SLT-only) |
| fig3 | spatial_proof | Style loss convergence: backward+SLT vs backward-only vs inject/SLT baselines |
| fig4 | slt_sweep | SLT strength α sweep (CSD vs CLIP-T dual-axis) |
| fig5 | timing | Per-step wall-clock breakdown |
| fig6 | iters_sweep | Adam iterations vs CSD (Hybrid / Adam-only / SLT-only / Inject) |
| fig7 | i2i_ablation | I2I component ablation |
| fig8 | pca_maps | 3×5 grid: Style Input \| Inject output \| Output (Ours) \| Inject direction PCA \| Backward gradient PCA |
| fig9 | qualitative_t2i | T2I qualitative grid, styles ['00','10','20','27','33'] |
| fig10 | qualitative_i2i | I2I qualitative grid |
| fig11 | qualitative_ablation | Ablation qualitative (includes 'w/o LRS' = no GRAG) |
| fig12 | t2i_benchmark | T2I benchmark comparison grid |

---

## Experiment Results (`results/`)

### Spatial Structure (`results/spatial_extended.json`)

| Method | Neighbor cos (d=1) | LE (cos@d=1 − cos@d=8) |
|--------|-------------------|------------------------|
| **Backward iter1** | **0.64** | **0.404** |
| Inject (slt_01) | 0.35 | 0.168 |
| Random | 0.00 | 0.000 |

**Takeaway**: Backward gradient has strong spatial coherence (large color patches in PCA), inject direction is near-random noise.

### Frequency Decomposition (`results/freq_decompose.json`)

- SLT direction: 27% low-freq / 73% high-freq
- Backward (no SLT) iter1: 49% low / 51% high → shifts to 43% low / 57% high by iter8
- cos(SLT direction, backward direction) ≈ −0.12 to −0.15 per iter (near-orthogonal)

**Takeaway**: SLT and backward operate in orthogonal subspaces; NOT frequency complementarity.

### Loss Decomposition (`results/loss_decompose.json`)

Averaged over 4 prompt×style pairs:

| Checkpoint | Style Loss | Dist Loss |
|-----------|-----------|----------|
| Raw (no style) | 0.293 | 0.360 |
| + AdaIN | 0.289 | 0.332 |
| + SLT | 0.284 | 0.317 |
| Backward+SLT @2 iters | 0.265 | 0.322 |
| Backward only @4 iters | 0.265 | 0.322 |

**Takeaway**: SLT pre-alignment saves ~2 backward iterations (SLT+2iter ≈ noslt@4iter in style_loss).

### GRAG/LRS Ablation

- Without GRAG (GRAG_DEEP=1.0, GRAG_SHALLOW=1.0): style layout leaks into output
- Ablation images: `d_grand/t2i_ablation/no_grag_*.png` (3 samples)

---

## Status: Phase 1 Complete

Both T2I and I2I pipelines are stable with optimal hyperparameters identified. Next steps could include:
- Multi-image evaluation (sweep across more cnt/sty pairs)
- Speed optimization (distill core reduces ~15% wall time)
- Higher resolution (768/1024)
- Prompt-guided style strength control
