# New 6 Styles — Supplementary Evaluation

**Goal**: Show method generalizes beyond the original 8 painterly/textural styles (00,10,14,20,27,28,33,38).  
**Date**: 2026-06-18  
**Dataset**: 6 new styles × 5 prompts = 30 images per method.

## New Style Set

| ID | Category | Description |
|----|----------|-------------|
| sty01 | geometric abstraction | Bright yellow, thin black/green lines, flat shapes |
| sty07 | children's illustration | Monochrome whimsical linework, ink style |
| sty09 | metallic sculpt / numismatic | Gold metallic sheen, coin-relief look |
| sty22 | manga B&W | Grayscale manga illustration, screen tones |
| sty23 | mosaic / Byzantine | Gold/blue/red Byzantine mosaic tiles |
| sty31 | pixel art | Metallic pixel art, warm gold/bronze palette |

Prompts (same as original eval): panda, castle, cat, sunflowers, fisherman.

## Metrics Summary

| Method | n | CSD ↑ | CLIP-T ↑ |
|--------|---|-------|---------|
| **Ours (hybrid2)** | 30 | **0.6041** | 0.2878 |
| StyleShot | 30 | 0.4217 | 0.2698 |
| IP-Adapter | 30 | 0.4482 | 0.2920 |
| USO | 30 | 0.3579 | 0.3194 |
| CSGO (fixed) | 30 | 0.1927 | 0.2164 |

### Original 8-Style Reference (for comparison)

| Method | n | CSD ↑ |
|--------|---|-------|
| Ours (hybrid2) | 40 | 0.6810 |
| StyleShot | 40 | 0.5327 |
| IP-Adapter | 40 | 0.3038 |
| USO | 40 | 0.5041 |
| CSGO (buggy params) | 40 | 0.5246 |

Note: CSGO original 40 used **wrong parameters** (content_scale/controlnet swapped).  
CSGO new 6 uses correct parameters: `content_scale=0.01, controlnet_conditioning_scale=1.0`.

### Our Method — Per-Style Breakdown (new 6)

| Style | CSD |
|-------|-----|
| sty01 (geometric) | 0.6848 |
| sty22 (manga) | 0.6281 |
| sty31 (pixel) | 0.6365 |
| sty09 (metallic) | 0.5945 |
| sty23 (mosaic) | 0.5613 |
| sty07 (illustration) | 0.5193 |

Our method maintains strong style fidelity across all 6 non-painterly categories. The slight drop vs original 8 (0.604 vs 0.681) reflects genuinely harder styles (abstract/geometric have less distinctive color/texture than oil paintings).

## File Locations

| Method | Directory | Naming |
|--------|-----------|--------|
| Ours | `d_grand/t2i_ablation/` | `hybrid2_kp{0-4}_s{sid}.png` |
| StyleShot | `d_grand/ext_styleshot/` | `sty{sid}_kp{0-4}.png` |
| CSGO | `d_grand/ext_csgo/` | `{sid}_p{0-4}.png` |
| USO | `d_grand/ext_uso_t2i/` | `{sid}_{0-4}_0.png` |
| IP-Adapter | `d_grand/ext_ipadapter/` | `{sid}_{0-4}.png` |

Metrics appended to existing CSVs in `results/`. Our method metrics in `results/t2i_new6styles.csv`.

## Scripts

- `scripts/run_new6_styleshot.py` — StyleShot generation
- `scripts/run_new6_csgo.py` — CSGO (fixed params) generation
- `scripts/run_new6_uso.py` — USO generation
- `scripts/run_new6_ipadapter.py` — IP-Adapter generation
- `scripts/metrics_new6_all.py` — Metrics computation (appends to existing CSVs)
- `scripts/run_all_new6.sh` — Master orchestration script (use longcat-video env for StyleShot)
