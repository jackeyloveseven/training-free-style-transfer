# 论文数字权威参考 (Paper Numbers — Canonical Reference)

> 单一事实源。写作端所有数字对齐此文件。  
> 生成日期：2026-06-18  
> 所有数字来自 `results/` 目录下对应 CSV / JSON；路径注释在括号内。

---

## 1. 主实验 — T2I 风格迁移对比

### 1.1 消融子集 (Ablation Subset)：8 styles × 5 prompts = 40 pairs

**数据文件**: `results/t2i_ablation.csv` (ours), `results/ext_*.csv` (baselines)  
**样式**: sty{00,10,14,20,27,28,33,38} × prompts {panda,castle,cat,sunflowers,fisherman}

| Method | CSD ↑ | CLIP-T ↑ | n |
|--------|-------|---------|---|
| **Ours (hybrid2)** | **0.6810** | **0.2737** | 40 |
| StyleShot | 0.5327 | 0.2693 | 40 |
| CSGO | 0.5246 | 0.2757 | 40 |
| USO | 0.5041 | 0.3050 | 40 |
| IP-Adapter | 0.3038 | 0.2960 | 40 |

> ⚠️ CSGO参数说明：原40张用错误参数生成（content_scale/controlnet对调），图像质量可接受。新6风格用修正参数但输出极暗（CSD=0.19），不纳入对比表。

### 1.2 内部基线 (Internal Baselines, 40 pairs)

**数据文件**: `results/t2i_baselines.csv`

| Method | CSD ↑ | CLIP-T ↑ |
|--------|-------|---------|
| inject_b10 (λ=1.0) | 0.8205 | 0.1499 |
| inject_b05 (λ=0.5) | 0.0389 | 0.3139 |
| adain_only | 0.1464 | 0.3137 |
| text_only | −0.0112 | 0.3151 |

> inject_b10 高 CSD 是 layout leak（内容→风格图结构泄漏），不代表真正的风格迁移。

### 1.3 全量基准 (Full Benchmark): 39 styles × 6 prompts

**数据文件**: `results/t2i_full_benchmark.csv` (hybrid2 + adam8, n=234 each), `results/ext_*.csv`

| Method | CSD ↑ | CLIP-T ↑ | n |
|--------|-------|---------|---|
| **Ours (hybrid2)** | **0.6405** | **0.2756** | 234 |
| Adam-8 (no SLT) | 0.6080 | 0.2788 | 234 |
| StyleShot | 0.4851 | 0.2695 | 70 |
| USO | 0.4415 | 0.3112 | 70 |
| IP-Adapter | 0.3657 | 0.2943 | 70 |

> ⚠️ 论文现写 hybrid2=0.639、adam8=0.607（均偏离约0.001-0.002），应更新为 0.6405 / 0.6080。  
> 全量外部基线（StyleShot/USO/IP-Adapter）各70张，使用 T2I_PROMPTS_FULL（cat/sports-car/sunflowers/dog/mountain/portrait），与消融子集提示词集不同。

### 1.4 FLUX.1-dev backbone (参考)

**数据文件**: `results/flux1dev_hybrid.csv`

| Backbone | CSD | CLIP-T | n |
|---------|-----|--------|---|
| Klein 9B (default) | 0.6810 | 0.2737 | 40 |
| FLUX.1-dev | 0.7188 | 0.2909 | 40 |

> FLUX.1-dev 在同等消融子集上高 +5.5% CSD，但主文使用 Klein 9B 以与其他实验一致。

---

## 2. 消融实验 — 组件贡献

### 2.1 T2I 组件消融 (40 pairs, `results/t2i_ablation.csv`)

| Method | CSD ↑ | CLIP-T ↑ | ∆CSD vs full |
|--------|-------|---------|-------------|
| **Full (hybrid2)** | **0.6810** | **0.2737** | — |
| w/o SLT (no_slt) | 0.5837 | 0.2910 | −0.0973 |
| w/o Adam (no_adam = SLT only) | 0.2984 | 0.3090 | −0.3826 |
| softmax attn | 0.6328 | 0.2825 | −0.0482 |
| freq_off | 0.6782 | 0.2707 | −0.0028 |

### 2.2 I2I 组件消融 (40 pairs, `results/i2i_ablation.csv`)

**样式**: 8 styles × 5 content images = 40 pairs

| Method | CSD ↑ | CLIP-I ↑ | n |
|--------|-------|---------|---|
| **Full (hybrid2)** | **0.6961** | 0.6700 | 40 |
| adam8 | 0.6489 | 0.6951 | 40 |
| no_slt | 0.5784 | **0.7392** | 40 |
| softmax | 0.6253 | 0.6941 | 40 |
| no_adam (SLT only) | 0.3284 | 0.8305 | 40 |

### 2.3 Adam-only 迭代数扫描 (30 pairs, `results/t2i_adam_only_sweep.csv`)

**样式**: 6 styles × 5 prompts = 30 pairs

| N (iters) | CSD ↑ | CLIP-T ↑ | Time (est.) |
|-----------|-------|---------|------------|
| 1 | 0.3627 | 0.3076 | ~7.5s |
| 2 | 0.5459 | 0.2931 | ~10.6s |
| 4 | 0.6400 | 0.2623 | ~10.6s |
| 5 | 0.6539 | 0.2552 | — |
| 8 | 0.6849 | 0.2374 | ~19.0s |

Hybrid×2: CSD=0.6810, 6.40s — **以 1/3 时间达到 Adam×8 同等性能**。

---

## 3. 理论支撑数据

### 3.1 空间相干性 (`results/spatial_extended.json`, n=6)

| Method | Neighbor Cosine (d=1) |
|--------|----------------------|
| **Backward iter1** | **0.640** |
| Inject | 0.349 |
| SLT (α=0.01/0.03/0.05) | 0.243 |
| Random | 0.000 |

> 梯度方向的空间相干性是 SLT 的 0.640/0.243 = **2.6×**。

### 3.2 时间基准 (`results/timing_benchmark.json`, H200, 512×512)

| Method | Time (s) |
|--------|---------|
| No style (baseline) | 2.20 |
| SLT only (α=0.1) | 2.21 |
| Hybrid + Adam×1 | 4.32 |
| **Hybrid + Adam×2** | **6.40** |
| Adam only ×2 | 6.40 |
| Adam only ×4 | 10.59 |
| Adam only ×8 | **18.96** |

> Hybrid×2 vs Adam×8：**2.96× 更快**（论文近似为"约3×"可接受）。

### 3.3 多种子稳定性 (`results/multiseed_summary.json`, seeds={43,44,45})

| Method | CSD mean | CSD std (over seeds) | p-value vs adam8 |
|--------|---------|---------------------|-----------------|
| hybrid2 | 0.6816 | 0.0025 | 7.1×10⁻¹⁵ |
| adam8 | 0.6503 | 0.0060 | — |

> 配对 t 检验（n=120 pooled）：t=8.91，p=7.1×10⁻¹⁵，mean diff=+0.031。  
> 即使 Adam×8 跑满 8 轮，Hybrid×2 仍显著更优（且快 3×）。

---

## 4. I2I 全量基准

### 4.1 I2I Full (hybrid2)  
**数据文件**: `results/i2i_full_benchmark.csv`，n=741（39 styles × 19 contents，排除 cnt19/sty04/sty14）

| Method | CSD ↑ | CLIP-I ↑ | n |
|--------|-------|---------|---|
| hybrid2 | 0.6620 | 0.6500 | 741 |

### 4.2 I2I 基线 (`results/i2i_baselines.csv`, 40 pairs)

| Method | CSD ↑ | CLIP-I ↑ |
|--------|-------|---------|
| inject_b10 | 0.8871 | 0.5100 |
| inject_b05 | 0.1814 | 0.8668 |
| adain_only | 0.1551 | 0.8631 |
| text_only | 0.1060 | 0.8680 |

### 4.3 Native Edit 基线 (FLUX.1-Kontext-dev)
**数据文件**: `results/native_edit.csv`, n=40

| Method | CSD ↑ | CLIP-T |
|--------|-------|--------|
| native_edit | 0.4874 | 0.3007 |

---

## 5. 补充数据

### 5.1 多样风格泛化 (new 6 styles, `results/t2i_new6styles.csv` + `results/ext_*.csv`)

**数据文件**: 新增6风格 (sty01 几何/sty07 插画/sty09 金属/sty22 漫画/sty23 马赛克/sty31 像素) × 5 prompts

| Method | CSD ↑ | CLIP-T ↑ | n |
|--------|-------|---------|---|
| **Ours (hybrid2)** | **0.6041** | 0.2878 | 30 |
| IP-Adapter | 0.4482 | 0.2920 | 30 |
| StyleShot | 0.4217 | 0.2698 | 30 |
| USO | 0.3579 | 0.3194 | 30 |
| (CSGO excluded — output nearly black with fixed params) | — | — | — |

### 5.2 辅助指标 (`results/supp_metrics.csv`, 40 pairs)

| Method | DINO Style Sim ↑ | Gram Dist ↓ | Layout Leak ↓ |
|--------|-----------------|------------|--------------|
| **Ours (hybrid2)** | 0.1548 | **2.731** | 0.3354 |
| lrs_off (no GRAG) | **0.1639** | 2.536 | 0.3413 |
| StyleShot | 0.1280 | 3.541 | 0.3180 |
| USO | 0.1092 | 3.681 | 0.3221 |

---

## 6. 关键数字速查 (Quick Reference)

| Number | Value | Source | 用途 |
|--------|-------|--------|------|
| Ours CSD (ablation) | **0.6810** | t2i_ablation.csv | 主表 |
| Ours CLIP-T (ablation) | **0.2737** | t2i_ablation.csv | 主表 |
| Ours CSD (full 39 styles) | **0.6405** | t2i_full_benchmark.csv | 全量表 |
| Ours CSD (I2I full) | **0.6620** | i2i_full_benchmark.csv | I2I表 |
| Adam×8 CSD | 0.6849 | t2i_adam_only_sweep.csv | 对比 |
| Hybrid×2 time | **6.40s** | timing_benchmark.json | 效率 |
| Adam×8 time | 18.96s | timing_benchmark.json | 效率 |
| Speed ratio | **~3×** | — | 摘要/intro |
| Spatial coherence (ours) | **0.640** | spatial_extended.json | 理论 |
| Spatial coherence (inject) | 0.349 | spatial_extended.json | 理论 |
| Coherence ratio | **2.6×** | — | 理论 |
| Multi-seed CSD std | ±0.0025 | multiseed_summary.json | 稳定性 |
| t-test p-value | 7.1×10⁻¹⁵ | multiseed_summary.json | 显著性 |

---

## 7. 注意事项 / Caveats

1. **CSGO 参数**: 消融40对使用"错误"参数（content_scale=1.0, controlnet=0.01），但视觉质量可接受；新6风格使用修正参数（output几乎全黑，CSD=0.19），不应出现在主要对比中。
2. **消融子集**: 实际为6 styles × 5 prompts = 30 pairs（adam_only_sweep用），主消融表为8 styles × 5 = 40 pairs（eval_all.py STYLES_6={00,10,14,20,27,28,33,38}）。
3. **全量基线样本**: StyleShot/USO/IP-Adapter全量各70张（CLIP-T列有数，使用 T2I_PROMPTS_FULL 中的6个提示），我们的全量用 T2I_PROMPTS_FULL × 39 styles = 234张。两者提示词集合不同，对比时需注意。
4. **注入基线 inject_b10**: CSD=0.82 是 layout leak 造成的虚高，图像实际是风格图的"扭曲副本"，不代表真正的风格迁移效果。
5. **FLUX.1-dev vs Klein 9B**: 论文主实验用 Klein 9B；FLUX.1-dev 仅作为参考行，不在消融表内。
6. **SLT τ**: 代码实际用 τ=0.02（非 0.15），0.15 是 attention-loss OT 的 OT_TAU_MAX。
7. **LRS γ**: 按层深线性插值 γ_shallow=0.7 → γ_deep=0.5（非 uniform 0.15）。

---

## 8. 论文现稿与规范数字的差异 (Paper vs Canonical)

> 对齐用：写作时以本文件为准，更新 LaTeX 中的过期数字。

| 位置 | 论文现值 | 规范值 | 动作 |
|------|---------|-------|------|
| `04_experiments.tex:114` — 全量表 Ours CSD | 0.639 | **0.6405** | 更新 |
| `04_experiments.tex:112` — 全量表 Adam-8 CSD | 0.607 | **0.6080** | 更新 |
| `04_experiments.tex:94,17` — styles数量 | "38 styles" | **39 styles** | 更新 |
| `04_experiments.tex:106-110` — 外部基线主表 | `\TODO{}` | StyleShot/USO/IPAdapter见1.3 | 补填 |
| `04_experiments.tex:127` — Inversion-Free/StyleAligned增益 | `\TODO{+XX%}` | 待补（未跑StyleAligned全量） | 遗留 |
| `04_experiments.tex:119` — "2.5× fewer than Adam-8" | 2.5× | **~3×** (6.40s vs 18.96s = 2.96×) | 更新 |
| `04_experiments.tex:166` — I2I CSD | 0.696 | **0.6961** | ✓ 已对齐 |
| `04_experiments.tex:239` — FLUX.1-dev CSD | 0.719 | **0.7188** | ✓ 已对齐 |
| `03_method.tex` — SLT τ | 0.15 | **0.02** | 见TPAMI修订计划 |
| `03_method.tex` — LRS γ | uniform 0.15 | **0.7→0.5 线性插值** | 见TPAMI修订计划 |

### 待填 \TODO

1. **全量表外部基线行**（`04_experiments.tex:106-110`）：StyleAligned 和 Inversion-Free Dual RF 未跑，填 StyleShot/USO/IP-Adapter 已有数字（1.3节）。
2. **"+XX% over StyleAligned"**（line 127）：需决定是删除此句还是补跑 StyleAligned 全量。
3. **消融表样式说明**：确认是8 styles×5 = 40 pairs（代码）还是论文写的其他数字。
