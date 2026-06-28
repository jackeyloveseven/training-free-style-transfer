# results/ 数据字典

本目录是风格迁移实验（Klein-9B / FLUX.2，frozen MM-DiT + SLT + Adam-in-loop + LRS）的**底层评测数据**：每个 CSV 是逐样本（per-pair）原始指标，每个 JSON 是机理分析的中间产物，几个 `.md`/`.txt`/`.html` 是旧的人工汇总/可视化。

---

## 0. 数字真源（务必先读）

- **唯一权威数字表 = `/home/ubuntu/chengyanli/items/sec/results/CANONICAL.md`。** 论文里出现的任何指标，以 CANONICAL 为准。
- 本目录的 **CSV / JSON 是上游原始数据**：图表脚本通常读取 `sec/results/` 下的同名镜像，再由 CANONICAL.md 汇总定稿。本 README 在“对应论文图表”列只给**指针**（哪张 fig / 哪张 tab），不抄具体数值，避免双份真值漂移。
- 本目录的 `PAPER_NUMBERS.md`、`new6_styles_results.md`、`paper_tables_summary.txt`、`viz_all.html` 都是**旧汇总，可能已过期** —— 仅作线索，数值一律回查 CANONICAL.md。
- 文件→图表的对应关系取自 `/home/ubuntu/chengyanli/items/sec/provenance/manifest.yaml` 的 `data_sources` 字段。

## 0.1 已归档作废数据

任务约定中提及的两个子目录 `_archive_stale_20260611/`、`_archive_wrong_20260621/` —— **截至本文件生成，results/ 下并不存在这两个目录**（已被清理或不在此处）。若日后再次出现，按“已归档作废数据，勿引用”处理，任何分析/图表都不要从中取数。

## 0.2 通用口径（默认值）

| 项 | 默认口径 |
|---|---|
| 模型 | Klein-9B（FLUX.2），512×512，bf16 |
| 默认 seed | 43（除文件名带 seed44/seed45/s44…s46 的多 seed 复跑） |
| **排除样本** | **sty04, sty14, cnt19**（任何统计都应剔除） |
| ablation 子集 | 8 styles {00,10,14,20,27,28,33,38} × 5 prompts {panda,castle,cat,sunflowers,fisherman} = 40 pairs |
| full benchmark | 全 style × 6 prompts（逐样本表，按 0.2 排除后聚合） |
| `time_s` 列 | 批跑时的逐样本墙钟，**不可靠**；准确耗时见 `timing_benchmark.json` |

## 0.3 常见列名语义

| 列名 | 含义 |
|---|---|
| `method` | 方法/变体名（如 hybrid2 / adam8 / inject_b10 / vshuffle …） |
| `sty_id` / `content_id` / `cnt_id` / `pid` / `prompt` | 风格 id / 内容 id / prompt 标识 |
| `csd` ↑ | CSD 风格保真度（主指标） |
| `clip_t` ↑ | CLIP-T 文本对齐（T2I） |
| `clip_i` / `clip_i_cnt` ↑ | CLIP-I 内容保留（I2I） |
| `lpips` / `tex_energy` / `orient_coh` | I2I 补充：感知距离 / 纹理能量 / 方向相干性 |
| `dino_style_sim` / `gram_dist` / `dino_layout_leak` | supp 补充：DINO 风格相似 / Gram 距离 / 布局泄露 |
| `iters` / `slt_alpha` / `slt_id` | sweep 自变量：Adam 迭代数 / SLT 强度 α |
| `time_s` | 见 0.2，不可靠 |

---

## 1. T2I 主表与全基准（论文引用）

| 文件 | 内容 | 列 | 对应论文图表 | seed/口径 |
|---|---|---|---|---|
| `t2i_full_benchmark.csv` | hybrid2 vs adam8 全基准逐样本 | method,sty_id,content_id,csd,clip_t,time_s | Tab1 tab:main; Tab2 tab:percat; Fig19 fig:vshuffle_cmp | seed43，full 6 prompts |
| `t2i_baselines.csv` | 弱基线（adain_only/inject_b05/inject_b10/text_only） | 同上 | Tab1 tab:main; Tab3 tab:texcoh; Fig6 fig:adam_sweep | seed43，ablation 子集 |
| `t2i_ablation.csv` | 消融变体逐样本（hybrid2/adam8/no_slt/no_adam/softmax/freq_off…） | 同上 | Tab1 tab:main; Tab3 tab:ablation; Tab3 tab:texcoh; fig:ablation_bar | seed43，40 pairs |
| `ext_styleshot.csv` | 训练型基线 StyleShot | method,sty_id,content_id,csd,clip_t | Tab2 tab:percat | ablation 子集 |
| `ext_csgo.csv` | 训练型基线 CSGO（见 PAPER_NUMBERS 参数警告） | 同上 | Tab2 tab:percat | ablation 子集 |
| `ext_uso_t2i.csv` | 训练型基线 USO（T2I） | 同上 | Tab2 tab:percat | ablation 子集 |
| `ext_ipadapter.csv` | 训练型基线 IP-Adapter | 同上 | Tab2 tab:percat | ablation 子集 |

## 2. T2I sweep / 权衡曲线（论文引用）

| 文件 | 内容 | 列 | 对应论文图表 | seed/口径 |
|---|---|---|---|---|
| `t2i_iters_sweep.csv` | Hybrid(SLT+Adam) 随迭代数 N 扫描 | iters,sty_id,content_id,csd,clip_t,time_s | Fig2 fig:iter_tradeoff; Fig6 fig:adam_sweep; Tab3 tab:iter_tradeoff | seed43，40-pair 子集 |
| `t2i_adam_only_sweep.csv` | Adam-only 随迭代数扫描（对照） | 同上 | Fig2 fig:iter_tradeoff; Fig6 fig:adam_sweep; Tab3 tab:iter_tradeoff | seed43，40-pair 子集 |
| `t2i_slt_alpha_sweep.csv` | SLT 强度 α 扫描 | slt_alpha,sty_id,content_id,csd,clip_t,time_s | Fig4 fig:alpha_sweep | seed43，SLT-only 40-pair |

## 3. V-Shuffle 对照（论文引用，Round 6 重跑）

| 文件 | 内容 | 列 | 对应论文图表 | seed/口径 |
|---|---|---|---|---|
| `t2i_vshuffle_b10_full.csv` | **当前**：V-Shuffle β=1.0（b10）Klein 全基准逐样本 | method,sty_id,content_id,csd,clip_t | Fig19 fig:vshuffle_cmp（配 t2i_full_benchmark） | seed43，β=1.0，n=234 配对 |
| `t2i_vshuffle_full.csv` | 旧版 V-Shuffle 全基准（非 b10）；以 b10 为准 | 同上 | （被 b10 取代，勿用于定稿） | seed43 |

> 备注：图脚本读取 `sec/results/t2i_vshuffle_klein_full.csv`（= b10 镜像）与 `d_grand/t2i_vshuffle_b10_full` 出图。CANONICAL Round 6 明确只取 b10（β=1.0）。

## 4. I2I（论文引用）

| 文件 | 内容 | 列 | 对应论文图表 | seed/口径 |
|---|---|---|---|---|
| `i2i_ablation.csv` | I2I 消融逐样本 | method,cnt_id,sty_id,csd,clip_t,clip_i_cnt,time_s | Fig7 fig:i2i_scatter; tab:i2i | seed43 |
| `i2i_baselines.csv` | I2I 弱基线（adain/inject/text_only） | 同上 | Fig7 fig:i2i_scatter; Tab4 tab:i2i_ext; tab:i2i | seed43 |
| `i2i_full_benchmark.csv` | I2I 全基准（ours） | cnt_id,sty_id,csd,clip_i_cnt,time_s | Tab4 tab:i2i_ext | seed43 |
| `i2i_eval_ours_s43.csv` | ours I2I 新指标，seed43 | method,cnt_id,sty_id,csd,clip_i,lpips,tex_energy,orient_coh | Tab4 tab:i2i_ext | seed43 |
| `i2i_eval_ours_s44.csv` / `_s45.csv` / `_s46.csv` | ours I2I 新指标，多 seed 复跑 | 同上 | Tab4 tab:i2i_ext（鲁棒性） | seed 44/45/46 |
| `i2i_eval_ext_csgo.csv` | I2I 外部基线 CSGO，新指标 | 同上 | Tab4 tab:i2i_ext | — |
| `i2i_eval_ext_uso.csv` | I2I 外部基线 USO，新指标 | 同上 | Tab4 tab:i2i_ext | — |
| `i2i_eval_ext_ipadapter.csv` | I2I 外部基线 IP-Adapter，新指标 | 同上 | Tab4 tab:i2i_ext | — |
| `i2i_eval_ext_qwen.csv` | I2I 外部基线 Qwen-Image-Edit，新指标 | 同上 | Tab4 tab:i2i_ext | — |
| `ext_uso_i2i.csv` | USO 的 I2I 结果（旧口径，csd/clip_i） | method,cnt_id,sty_id,csd,clip_i | （I2I 外部基线线索，定稿见 i2i_eval_* / CANONICAL） | — |

## 5. 补充表 / 其它模型（论文引用）

| 文件 | 内容 | 列 | 对应论文图表 | seed/口径 |
|---|---|---|---|---|
| `supp_metrics.csv` | 补充指标（含 lrs_off 等）逐样本 | method,sty_id,pid,dino_style_sim,gram_dist,dino_layout_leak | tab:supp_metrics | seed43 |
| `native_edit.csv` | 原生 edit 模型对照 | method,sty_id,content_id,csd,clip_t | fig:native_cmp | seed43 |
| `flux1dev_hybrid.csv` | Hybrid 在 FLUX.1-dev 上的迁移性 | method,sty_id,content_id,csd,clip_t | tab:flux1dev | FLUX.1-dev |
| `t2i_new6styles.csv` | 6 个新风格泛化（sty01/07/09/22/23/31） | method,sty_id,content_id,csd,clip_t,time_s | 见 new6_styles_results.md（补充） | seed43，6 styles×5 prompts=30 |

## 6. 机理分析 JSON（论文引用）

| 文件 | 内容 | 顶层键 / 结构 | 对应论文图表 | 口径 |
|---|---|---|---|---|
| `gradient_source.json` | 梯度空间结构来源（normal/shuffle_style_kv/content_only/single_patch） | list[{prompt,style,normal,shuffle_style_kv,content_only,single_patch}] | Fig3 fig:spatial_proof; Tab1 tab:spatial_proof | n=4 |
| `csd_per_iter.json` | backward vs inject 的逐迭代 CSD | list[{prompt,style,backward_csd,inject_csd}] | Fig3 fig:spatial_proof | n=4 |
| `spatial_proof_v2.json` | 邻域余弦（backward/slt/random）统计 | n_samples, backward_cos_mean/std, slt_cos_mean/std, random_cos_mean, ratio…, autocorrelation_decay | Fig3 fig:spatial_proof | — |
| `spatial_extended.json` | 邻域余弦 d=1 + 自相关衰减 | n_samples, neighbor_cosine_d1, autocorrelation_decay | Fig3 fig:spatial_proof; Tab1 tab:spatial_proof | — |
| `spatial_viz_summary.json` | PCA / update 方向 cos1（backward/adam/slt/random） | backward_grad_cos1, adam_update_cos1, slt_update_cos1, random_cos1 | Fig8 fig:pca_maps | — |
| `jacobian_ablation.json` | Jacobian 路径消融（normal/random_direction/stop_grad_q） | list[{prompt,style,normal,random_direction,stop_grad_q}] | Fig13 fig:jacobian_ablation | n=4 |
| `freq_decompose.json` | 频域分解（SLT vs backward±SLT 逐迭代） | freq_cutoff, n_samples, slt, backward_no_slt_per_iter, backward_with_slt_per_iter, direction_alignment | Fig17 fig:freq_decomp | — |
| `theory_profiling.json` | 理论常数实测（κ_freq、cos、δ 等） | definitions, adam_config, slt_alpha, n_pairs, epsilon, kappa_freq_*, cos_gc_gt_freq_*, delta, secondary_term_decomposition, paper_assumed, per_pair | tab:measured_constants | — |
| `timing_benchmark.json` | 各方法墙钟中位数（权威耗时源） | list[{method,median_s,times,hybrid_iters,slt_alpha}] | fig:timing; Tab3 tab:ablation | H200，n=8 方法 |

## 7. 多 seed 鲁棒性汇总（JSON）

| 文件 | 内容 | 顶层键 | 口径 |
|---|---|---|---|
| `multiseed_summary.json` | 多 seed 聚合 + Hybrid vs Adam 配对 t 检验 | seeds, per_method, paired_ttest_csd_hybrid_vs_adam | 多 seed |
| `multiseed_full_summary.json` | 多 seed 全量聚合 | seeds, per_method | 多 seed |

## 8. 开发期探索数据（论文未直接引用，manifest 无映射）

以下文件为调参/探索/早期复跑产物，**未被任何论文图表的 `data_sources` 引用**；保留作过程记录，定稿勿引，数值一律以 CANONICAL.md 为准。

| 文件 | 大致用途 |
|---|---|
| `t2i_full_confirm.csv`, `t2i_full_ultA.csv`, `t2i_full_ultB.csv` | full benchmark 的确认/复跑批次 |
| `t2i_full_seed44.csv`, `t2i_full_seed45.csv`, `t2i_seed44.csv`, `t2i_seed45.csv` | 多 seed full / 子集复跑（seed 44/45） |
| `t2i_sltopt.csv`, `t2i_sltopt2.csv`, `t2i_sltadam.csv`, `t2i_sltpareto.csv` | SLT / SLT+Adam 调参与 Pareto 探索 |
| `t2i_clipt.csv`, `t2i_cliptloop.csv` | CLIP-T 相关变体探索 |
| `t2i_decomp.csv`, `t2i_featot.csv`, `t2i_texture.csv` | 分解 / 特征-OT / 纹理路线探索 |
| `t2i_microtune.csv`, `t2i_relaxA.csv`, `t2i_relaxB.csv`, `t2i_p1cheap.csv`, `t2i_kshot2.csv` | 微调 / 松弛 / 低成本 / k-shot 探索 |
| `attn_pca_coherence.json` | attention PCA 相干性中间产物 |
| `loss_decompose.json` | 损失分解（backward ±slt）中间产物 |
| `spatial_smoothing_proof.csv`, `spatial_smoothing_summary.json` | 早期空间平滑性证明（被 spatial_proof_v2/extended 取代） |

## 9. 旧汇总文档（线索，非真源）

| 文件 | 内容 | 状态 |
|---|---|---|
| `PAPER_NUMBERS.md` | 2026-06-18 人工汇总主表/消融数字 + CSGO 参数警告 | 旧汇总，可能过期，回查 CANONICAL |
| `new6_styles_results.md` | 6 新风格补充评测汇总 | 旧汇总，可能过期，回查 CANONICAL |
| `paper_tables_summary.txt` | 2026-05-21 文本版表格汇总（口径偏旧：仅排除 sty04） | 旧汇总，勿引 |
| `viz_all.html` | Chart.js 本地可视化看板 | 浏览用，非数据源 |

---

### 运行解释器约定（如需复算）

- T2I/I2I 生成（需 diffusers）：`/home/ubuntu/chengyanli/flux/bin/python3`
- eval / 分析脚本：`/home/ubuntu/miniconda3/envs/sensenova/bin/python`（无 diffusers，勿混用）
