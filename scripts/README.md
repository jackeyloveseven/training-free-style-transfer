# scripts/ 目录索引

本目录是**风格迁移实验的「跑实验 / 算指标」脚本**集合（52 个 `.py` + 10 个 `.sh`）：生成 T2I/I2I 图、复算 CSD/CLIP/DINO 等指标、做机制分析与参数 sweep。产物落在仓库内 `d_grand/`（图）与 `results/`（CSV/JSON）。

> **职责边界**：本目录**不做论文图表拼装**。论文里的 figure/table 组装脚本在 `/home/ubuntu/chengyanli/items/sec/scripts/`（`build_fig_*.py` 等），其数据溯源见 `/home/ubuntu/chengyanli/items/sec/provenance/`。两仓库分工：`save/.../scripts` 产出原始图与指标 → `sec/scripts` 消费并拼图。
>
> **指标数字的唯一真源**是 `/home/ubuntu/chengyanli/items/sec/results/CANONICAL.md`。本目录脚本会把数值写进各自的 `results/*.csv`，但**最终对外口径以 CANONICAL.md 为准**；本文件只给“某脚本写哪张表”的指针，不抄具体数值。

## 运行环境（两套解释器，别混）

| 用途 | 解释器 |
|---|---|
| T2I/I2I 生成、本仓 Klein/diffusers 流程、eval/分析脚本 | `/home/ubuntu/miniconda3/envs/sensenova/bin/python`（注意：CLAUDE.md 标 sensenova **无 diffusers**；本仓多数脚本走 DiffSynth + Klein9B，不依赖 diffusers，故用此环境）|
| 需要 `diffusers` 的外部 baseline（IP-Adapter / Qwen-Edit / USO / CSGO 等） | `/home/ubuntu/chengyanli/flux/bin/python3` |
| StyleShot 等历史 baseline | `/home/ubuntu/miniconda3/envs/longcat-video/bin/python`（见 `run_*.sh` 中 `$LONGCAT`）|

每个脚本 docstring 顶部都注明了应使用的解释器，以其为准。

## 公共约定

- 仓库根 `CWD = /home/ubuntu/chengyanli/items/save/attention_20260426_090948`，多数脚本硬编码于此。
- 输入图：`sty/<id>.png`（风格参考）、`cnt/<id>.png`（内容图，I2I 用）。
- 标准 8 风格子集 `STYLES = 00,10,14,20,27,28,33,38`；I2I 内容子集 `CNTS = 00,04,08,12,17`。
- **排除样本：`sty04`、`sty14`、`cnt19`**（下游统计/拼图时应剔除；部分脚本的固定 `STYLES` 列表仍含 14，引用其结果时注意排除）。
- CSD/CLIP 指标实现以 `eval_all.py` 的 `compute_csd` / `compute_clip_t` 为基准，外部 baseline 的打分脚本都声明“verbatim 复制自 eval_all”以保证可比。

---

## 1. 批量评测 / 生成主力

| 脚本 | 用途 | 输入 | 输出 | 备注 |
|---|---|---|---|---|
| `eval_all.py` | 统一批量评测：T2I ablation/baseline + I2I ablation/baseline，单进程只加载一次 Klein9B；被大量脚本 `import eval_all` 复用 | `sty/`、`cnt/`、Klein9B、CSD/CLIP 模型 | `d_grand/t2i_*`、`d_grand/i2i_*`、`results/t2i_*.csv`、`results/i2i_*.csv` | `--task {t2i_ablation,i2i_ablation,t2i_baselines,i2i_baselines,t2i_full,i2i_full,t2i_new6styles,all}`；**核心模块** |
| `i2i_eval_all.py` | I2I 统一打分：对某方法 40 对（8 sty×5 cnt）输出算 CSD/CLIP-I/LPIPS/FID_style/ArtFID/纹理能量/方向一致性 | 方法输出目录 + `sty/`、`cnt/` | `results/i2i_eval_<tag>.csv` | 用 `flux/bin/python3`；命令行 `<tag> <out_dir> <fname_template>`（模板带 `{cnt}{sty}`）|
| `run_seed.py` | T2I 多 seed 显著性：hybrid2+adam8 跑 seed 44/45（43 已存），配对 t-test | `eval_all`（单次加载） | `results/t2i_seed<S>.csv`、`results/multiseed_summary.json` | `import eval_all` 复用 |
| `run_seed_full.py` | 全 benchmark 版多 seed（39 sty×6 prompt），seed 44/45 | `eval_all` | `results/t2i_full_seed<S>.csv`、`results/multiseed_full_summary.json` | 不动 canonical seed-43 表 |
| `run_p0.py` | P0 驱动：一进程内先 `run_seed` 再 `profile_theory`（Klein 只加载一次） | 同上两脚本 | 同上两脚本产物 | 编排型 |
| `ours_i2i_multiseed.py` | 我方 Klein-9B hybrid I2I 多 seed（44/45/46）补跑 40 对 | `eval_all.I2I_HYBRID`、`cnt/`、`sty/` | `d_grand/i2i_ours_seed{N}/hybrid2_cnt*_sty*.png` | seed43 已在 `d_grand/i2i_ablation` |
| `run_ours_bestofn.py` | 比较图用：每风格跑 N seed，按 CSD+CLIP-T 联合归一化挑最优一张 | `eval_all`、kp0 prompt | `d_grand/ours_bestofn/<sid>.png` | 仅供拼图；定量表仍用固定 seed |
| `run_test.py` | 单样本快速试跑（指定 sty/prompt/out） | `sty/`、Klein | 指定 `--out` PNG | argparse：`--sty --prompt --out --csd_w --seed` |

## 2. T2I 指标 / 机制分析

多为「在已生成图或重跑去噪过程上」做机制证明，产物多为 `results/*.json`，供论文方法/分析章节引用。

| 脚本 | 用途 | 输入 | 输出 | 备注 |
|---|---|---|---|---|
| `eval_spatial_extended.py` | 空间证明扩展：inject vs backward vs SLT（为何 inject≪backward、SLT 颜色≈Adam 但纹理不同） | Klein 重跑、`sty/` | `results/` + `d_grand/` 下分析产物 | 机制分析 |
| `eval_spatial_viz.py` | 空间平滑可视化（论文图）：梯度场 PCA→RGB、更新幅度热图、自相关衰减曲线 | Klein 重跑 | 出版级 PNG | 证明 brushstroke 机制 |
| `eval_freq_decompose.py` | 频域分解：SLT 低频(色) vs backward 高频(纹理)互补性，逐 iter FFT 能量分带 | Klein 重跑 | `results/freq_decompose.json`、`d_grand/freq_analysis/` | |
| `eval_gradient_source.py` | 梯度来源分析：shuffle 风格 K/V、content-only loss、single-token loss 三组消融 | Klein 重跑、`sty/` | `results/gradient_source.json` | 含 V-Shuffle 思想雏形 |
| `eval_jacobian_ablation.py` | Jacobian 消融：随机梯度方向 + attention 内 stop-grad，证明注意力 Jacobian 耦合是 backward 增益的成因 | Klein 重跑 | `results/jacobian_ablation.json` | |
| `eval_attn_pca.py` | 注意力图 PCA 可视化：opt 前/SLT-only/Adam iter1/iter2 的 attention→PCA→RGB | Klein 重跑（含 attention 钩子） | PNG 网格 + 空间一致性指标 | |
| `eval_loss_decompose.py` | 损失分解：证明 SLT 先降「分布对齐」项，使 backward 起点 loss 更低 | Klein 重跑 | `results/loss_decompose.json` | 推翻“频率互补”改证此机制 |
| `eval_iters_and_smoothing.py` | (1) HYBRID_ITERS∈{0..8} 时间-质量 Pareto；(2) 梯度自相关空间平滑证明 | Klein 重跑、`sty/` | `results/` 下 CSV/JSON | `--task {all,smoothing,iters,slt,adam_only}` |
| `eval_csd_per_iter.py` | 逐 backward iter 解码算 CSD，证明 backward 单调改善外部指标；inject 重复 N 次应立即饱和 | Klein 重跑、CSD 模型 | `results/csd_per_iter.json` | |
| `eval_supp_metrics.py` | 补充感知指标（不重新生成）：DINOv2 风格相似度、VGG19 Gram 距离、DINO 布局泄露 | 已生成 PNG、DINOv2/VGG | `results/supp_metrics.csv` | 防单指标质疑；LPIPS/ArtFID 对 T2I 不适用故跳过 |
| `eval_supplement.py` | 补充实验：补跑新 styles(14/28/38)、空间平滑图重设计、独占 GPU timing | Klein、`sty/` | `results/` 下产物 | `--task {all,补跑,spatial,timing}` |
| `profile_theory.py` | 理论常数实测（supp.tex Thm1/Thm2）：在 30 对上测 ε/κ/δ 等梯度常数 | Klein 重跑（梯度钩子） | `results/theory_profiling.json` | 供理论附录 |
| `eval_gold4.py` | 单次 Klein T2I + 4 项 gold 指标(CSD/CLIP/DINO/time) | Klein、CSD/CLIP/DINOv2 | 打印 | 早期联调用 |
| `eval_kshot2.py` | KSHOT=2（平均 2 次风格前向）是否提升 CSD 而不损 CLIP-T | Klein（`KSHOT=2`）、40 样本子集 | `d_grand/t2i_kshot2/` | 对照现有 hybrid2 KSHOT=1 |

## 3. 外部 baseline：运行 + 打分

> 多个外部 baseline 的**仓库代码 / 权重已从磁盘删除**（StyleShot、USO、CSGO 部分），脚本 docstring 里写明了 `hf download` / `git clone` 的恢复命令；运行前需先按其说明恢复。

### 3a. 新 6 风格 / iconic 风格补跑（T2I）

| 脚本 | 用途 | 输入 | 输出 | 备注 |
|---|---|---|---|---|
| `run_new6_csgo.py` | CSGO（修正 content/controlnet scale 调换 bug）跑新 6 风格×5 prompt | `sty/`、CSGO 权重 | `d_grand/ext_csgo/{sid}_p{N}.png` | `STYLES_OVERRIDE` 可注入风格集 |
| `run_new6_styleshot.py` | StyleShot 跑新 6 风格 | `sty/`、StyleShot+SD1.5 权重 | `d_grand/ext_styleshot/sty{sid}_kp{N}.png` | 权重可能缺失 |
| `run_new6_uso.py` | USO(FLUX.1-dev) 跑新 6 风格（子进程） | `sty/`、USO 权重 | `d_grand/ext_uso_t2i/{sid}_{N}_0.png` | |
| `run_new6_ipadapter.py` | X-Flux IP-Adapter 跑新 6 风格（子进程） | `sty/`、XLabs 权重 | `d_grand/ext_ipadapter/{sid}_{N}.png` | |
| `metrics_new6_all.py` | 5 方法新 6 风格统一算 CSD+CLIP-T，追加各 `ext_*.csv` 并打印汇总 | 上述 `d_grand/ext_*`、`sty/` | 追加 `results/ext_*.csv`、`results/t2i_new6styles.csv` | `STYLES_OVERRIDE` |

### 3b. 单 baseline 打分（复刻 eval_all 指标）

| 脚本 | 用途 | 输入 | 输出 | 备注 |
|---|---|---|---|---|
| `eval_ext_csgo.py` | CSGO 打分（CSD/CLIP-T，verbatim 复刻 eval_all） | `d_grand/ext_csgo`、`sty/` | `results/ext_csgo.csv` | `--smoke PATH STY` |
| `eval_ext_uso_t2i.py` | USO T2I 打分 | `d_grand/ext_uso_t2i`、`sty/` | `results/ext_uso_t2i.csv` | |
| `metrics_ext_csgo.py` | CSGO CSD/CLIP-T（与 eval_ext_csgo 同口径，8 风格表） | `d_grand/ext_csgo` | `results/ext_csgo.csv` | |
| `metrics_ext_styleshot.py` | StyleShot CSD/CLIP-T | `d_grand/ext_styleshot` | `results/ext_styleshot.csv` | |
| `metrics_ext_uso_i2i.py` | USO **I2I** CSD/CLIP-I(content) | `d_grand/ext_uso_i2i`、`sty/`、`cnt/` | `results/ext_uso_i2i.csv` | 风格集 00,10,20,27,33 |
| `recompute_csgo.py` | CSGO cc=0.0 灰图 bug 修复后重算并**覆盖** | `d_grand/ext_csgo` | 覆盖 `results/ext_csgo.csv` | 一次性修复脚本 |

### 3c. 外部 baseline I2I 运行

| 脚本 | 用途 | 输入 | 输出 | 备注 |
|---|---|---|---|---|
| `i2i_run_csgo.py` | CSGO 内容+风格 I2I（40 对） | `cnt/`、`sty/`、CSGO 权重 | `d_grand/i2i_ext_csgo/csgo_cnt*_sty*.png` | 复用 run_new6_csgo 的 build()；权重可能缺失 |
| `i2i_run_ipadapter.py` | IP-Adapter(FLUX.1-dev) I2I（FluxImg2ImgPipeline） | `cnt/`、`sty/`、XLabs 权重 | `d_grand/i2i_ext_ipadapter/ipadapter_cnt*_sty*.png` | strength=0.6 |
| `i2i_run_qwen_edit.py` | Qwen-Image-Edit-2511 I2I（image=[content,style]） | `cnt/`、`sty/`、Qwen 权重 | `d_grand/i2i_ext_qwen_edit/qwen_edit_cnt*_sty*.png` | 可选 Lightning 8 步 |
| `i2i_run_styleshot.py` | StyleShot 轮廓 ControlNet I2I | `cnt/`、`sty/`、StyleShot 权重 | `d_grand/i2i_ext_styleshot/` | repo+权重均缺失，需恢复 |
| `i2i_run_uso.py` | USO 内容+风格 I2I（40 对） | `cnt/`、`sty/`、USO 权重 | `d_grand/i2i_ext_uso/` | 权重缺失，代码在 backup |

## 4. 参数 sweep / 调参

| 脚本 | 用途 | 输入 | 输出 | 备注 |
|---|---|---|---|---|
| `sweep_inproc.py` | 单进程加载一次模型扫参，报真实推理时间 + 末端算指标 | Klein、`sty/` | 打印/临时图 | |
| `sweep_gold4.py` | 关键参数 sweep，报 4 gold 指标 | Klein、CSD/CLIP/DINO | 打印 | |
| `sweep_gold4_v2.py` | 第二轮 sweep，主推 DINO(sty) | 同上 | 打印 | |
| `sweep_7s.py` | 7 秒预算内（iters 2–3）求最佳 4 gold | 同上 | 打印 | |
| `sweep_7s_v2.py` | 7 秒：降 NUM_STEPS / 升 start_time | 同上 | 打印 | |
| `sweep_7s_metrics.py` | 跑 7 秒预算配置并算 4 gold | Klein | 打印 | |
| `search_i2i_params.py` | I2I 参数搜索（cnt00×sty27）：start_time/adain_alpha/content_w/slt_alpha/hybrid_iters | `cnt/`、`sty/` | 打印/临时图 | |
| `probe_ot_layer_frac.py` | 探测 OT 层路由 split（OT_LAYER_FRAC）能否同时 Pareto 改进 CSD 与 CLIP-T | Klein | 打印 | 等推理成本、不改权重 |

## 5. V-Shuffle / 专项

| 脚本 | 用途 | 输入 | 输出 | 备注 |
|---|---|---|---|---|
| `run_vshuffle.py` | V-Shuffle（MM-DiT 适配）baseline：默认 3 对验证，`--full` 跑 8×5 子集 | `eval_all.T2I_BASELINES['vshuffle']`、`sty/` | `d_grand/ext_vshuffle/`、`results/ext_vshuffle.csv` | |
| `run_vshuffle_klein_full.py` | Klein-9B V-Shuffle 全 benchmark（39×6=234），β=1.0 全注入，可断点续 | `eval_all`、`sty/` | `d_grand/t2i_vshuffle_b10_full/`、`results/t2i_vshuffle_b10_full.csv` | 与 hybrid 全表对齐 |
| `eval_native_edit.py` | Klein9B 原生 in-context edit 作为风格迁移 baseline（无 SLT/Adam） | `sty/`、Klein 原生 `edit_image` | `results/` CSV | `import eval_all`（仅加载不跑 benchmark）；`--task {smoke,full}` |
| `eval_flux1dev_hybrid.py` | 评测 FLUX.1-dev Hybrid 输出（我方指标口径） | `d_grand/flux1dev_hybrid`、`sty/` | `results/flux1dev_hybrid.csv` | 用 `flux/bin/python3` |
| `gen_lrs_demo.py` | LRS 演示对：w/ LRS(grag0.5/0.7) vs w/o(grag1.0) 固定 seed 强布局风格 | `eval_all`、`sty/` | `d_grand/lrs_demo/` | 供 LRS 消融图挑图 |

## 6. 可视化 / HTML

| 脚本 | 用途 | 输入 | 输出 | 备注 |
|---|---|---|---|---|
| `gen_sltopt_html.py` | 构建自包含 HTML 对比 SLT 算子（ot_adain/W2/SW，standalone+hybrid），缩略图 base64 内嵌 | `d_grand/t2i_sltopt`、`sty/`、`results/t2i_sltopt.csv` | `d_grand/sltopt_compare.html` | 行=风格 列=算子 |

## 7. Shell 编排

| 脚本 | 串联的 py | 用途 |
|---|---|---|
| `run_all_new6.sh` | `eval_all --task t2i_new6styles` → `run_new6_styleshot/csgo/uso/ipadapter` → `metrics_new6_all` | 5 方法新 6 风格全流程 + 算指标 |
| `run_new6_from_step2.sh` | 同上去掉 step1（ours 已跑） | 从 StyleShot 续跑 |
| `run_new6_from_step4.sh` | `run_new6_uso/ipadapter` → `metrics_new6_all` | 从 USO 续跑 |
| `run_iconic7.sh` | 同 run_all_new6，`STYLES_OVERRIDE=02,08,12,19,34,35,37` | 7 个 iconic 风格家族补跑 + 算指标 |
| `run_eval_resume.sh` | `eval_all --task {t2i_full,i2i_ablation,i2i_baselines,i2i_full}` | 等 PCA 跑完后续跑失败的 eval_all 任务 |
| `run_final_resume.sh` | `eval_all`（各 task 断点续）+ `eval_spatial_viz` | 崩溃后最终补图 |
| `run_remaining_experiments.sh` | `eval_iters_and_smoothing --task all` + `eval_attn_pca` | 等指定 PID 结束后跑剩余机制实验 |
| `run_supplement.sh` | `eval_supplement --task all` | 等 GPU 空闲后跑补充实验 |
| `run_i2i_overnight.sh` | `i2i_run_{qwen_edit,csgo,uso}` → `i2i_eval_all`(4 baseline + 我方 4 seed) → 汇总 | I2I baseline 过夜全跑 + 打分 |
| `i2i_rerun_ip_qwen.sh` | `i2i_run_ipadapter` + `i2i_run_qwen_edit` → `i2i_eval_all` | 调强参数后重跑 IP-Adapter/Qwen 并重评 |

---

## 已知坑（复现前必读）

- **硬编码绝对路径**：几乎所有脚本写死 `CWD=/home/ubuntu/.../attention_20260426_090948`、`DIFFSYNTH_ROOT=/mnt/local/yzh/prog/DiffSynth-Studio`、`KLEIN9B_ROOT=/mnt/local/klein9b`、各 `/mnt/local/baselines/*` 与 `/home/ubuntu/chengyanli/items/CSD_Score`。换机器需逐一改。
- **外部 baseline 代码/权重缺失**：StyleShot、USO、CSGO 部分权重已从磁盘删除（USO 代码仅存于 `..._backup_20260612`）。各脚本 docstring 顶部给了 `hf download` / `git clone` 恢复命令，运行前先恢复。
- **个别 Windows `D:` 路径残留**：少数脚本/配置仍含历史 `D:` 残留，复现前需改为 Linux 路径。
- **风格集仍含被排除样本**：部分固定 `STYLES` 列表含 `14`；统计/拼图时务必剔除 `sty04/sty14/cnt19`。
- **更全的 gap 清单与逐项细节见** `/home/ubuntu/chengyanli/items/sec/provenance/GAPS.md` **§E**。
