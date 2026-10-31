# attention_20260426_090948 — Training-Free 风格迁移实验仓库

MM-DiT 上的 **training-free 风格迁移**：冻结主干，在去噪过程中优化 latent，使其自注意力 K/V 分布对齐风格参考图（deep 层 Sinkhorn-OT attention + Adam-in-loop + SLT 预对齐）。本仓库是**跑实验 / 算指标**的地方；配套论文在 `items/sec`。

- 方法/管线详解 → [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **数字真源（任何指标以此为准）→ `items/sec/results/CANONICAL.md`**
- 论文每张图/表的端到端溯源 → `items/sec/provenance/`（manifest + 44 dossier + GAPS）

## Table 8 补实验档案（H100，2026-10-31）

已完成六组新增步数配置与 T25/N2 对照，每组 40 对、seed 43，共 280 张原始图。

- [完整 ZIP 档案](https://github.com/DC-Z-lab/training-free-style-transfer/releases/download/table8-h100-20261031-full/hyperion_table8_h100_complete_20261031.zip)：含代码和论文快照、280 张原图、逐样本指标、计时原数、配置、依赖、日志及更新稿件。
- [实验结果与复现说明](docs/table8-h100-20261031/README.md)
- [完整精度 CSV](docs/table8-h100-20261031/measured_results.csv) · [Table 8 TeX](docs/table8-h100-20261031/table8_h100.tex)

压缩包 SHA256：`00861ea9ee4a421471b09075ca75c1e5482edd415e268581cab9aa161d0b7b20`。

模型权重约 38.7 GB 及虚拟环境实体未入包，下载来源、版本和哈希随包保存。此次 40 对消融子集包含 sty14；其样本列表以包内记录为准。原 `t<850` 阈值下 T4 只有 1 个有效优化步，N=4 未恢复 T25 的风格水平。

## 安装环境与下载模型

建议使用 Python 3.11/3.12。先安装与本机 CUDA 匹配的 PyTorch，再安装依赖和
DiffSynth-Studio：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
git clone https://github.com/modelscope/DiffSynth-Studio.git third_party/DiffSynth-Studio
python -m pip install -e third_party/DiffSynth-Studio
hf auth login  # FLUX/Klein 等 gated 模型需先在网页接受许可证
```

下载命令按用途拆分，避免一次下载全部大模型：

```bash
bash scripts/download_models.sh core       # FLUX.1-dev、Kontext、Klein 9B
bash scripts/download_models.sh scores     # DINOv2、CLIP 打分模型
bash scripts/download_models.sh baselines  # IP-Adapter、USO、CSGO、StyleShot、Qwen
bash scripts/download_models.sh all        # 全部下载
```

`core` 是主实验所需；CSD 打分还需同级 `../../CSD_Score/models/` 下的
`ViT-L-14.pt` 和 `checkpoint.pth`。`scores` 下载 DINOv2 与 Hugging Face CLIP；
VGG19/LPIPS 首次运行时由 torchvision/lpips 自动获取。所有权重都放在当前仓库
`models/` 下，具体目录见 [`models/README.md`](models/README.md)。

## Quickstart

```bash
# 在本目录执行；模型统一放在 models/
FLUXPY=python
EVALPY=python

# T2I 单图（FLUX.1-dev）
PROMPT="A panda eat bamboo" STY_IMAGE=sty/27.png $FLUXPY custom_main.py
# I2I 单图（FLUX.1-Kontext-dev）
CNT_IMAGE=cnt/00.png STY_IMAGE=sty/27.png $FLUXPY custom_main_img.py
# 论文主 backbone = Klein 9B：T2I custom_main_klein9b_t2i.py / I2I custom_main_klein9b.py
# 批量评测（论文 benchmark）
$EVALPY scripts/eval_all.py --task all      # 细节见 scripts/README.md
```

env vars（NUM_STEPS / OVERRIDE_START_TIME / CSD_W / GRAG_* / ADAIN_ALPHA / FIXED_LAYERS / SEED / LR / ITERS）完整表见 `docs/ARCHITECTURE.md`。

## 目录分类总表

### 🟩 源码 Source（含硬编码路径，勿随意移动；production 核勿改）
| 路径 | 说明 |
|---|---|
| `custom_main.py` | T2I 入口（FLUX.1-dev）— **DO NOT MODIFY**（论文 benchmark） |
| `custom_main_klein9b_t2i.py` | T2I 入口（**Klein 9B，论文主 backbone**） |
| `custom_main_img.py` | I2I 入口（FLUX.1-Kontext-dev） |
| `custom_main_klein9b.py` | I2I 入口（Klein 9B） |
| `run_flux1dev.py` | FLUX.1-dev 跑批 |
| `mypipeline.py` / `mypipeline_img.py` | T2I / I2I pipeline 封装 |
| `utils.py` | 共享工具（sinkhorn / adain / Controller…） |
| `cores/` | 注意力核（production `core_ot_attn*.py` 勿改）→ [`cores/README.md`](cores/README.md) |
| `scripts/` | 52 py + 10 sh：评测 / baseline / sweep / 分析 / 编排 → [`scripts/README.md`](scripts/README.md) |

### 🟦 数据 Data（输入，只读）
| 路径 | 大小 | 说明 |
|---|---|---|
| `sty/` | 16M | 40 张风格图 + `manifest.json`；**排除 sty04, sty14**（full benchmark=39，仅排 sty04） |
| `cnt/` | 8.3M | 20 张内容图；**排除 cnt19** |

### 🟨 结果 Results（派生指标数据）
| 路径 | 大小 | 说明 |
|---|---|---|
| `results/` | 592K | CSV/JSON 指标 → [`results/README.md`](results/README.md)。**数字真源 = `items/sec/results/CANONICAL.md`**；含 `_archive_*/` 作废数据（勿引用） |

### 🟧 生成产物 Output（重，gitignored，可重生）
| 路径 | 大小 | 说明 |
|---|---|---|
| `d_grand/` | 3.8G | 全部生成图（39 子目录，KEEP/REPLICATE 标注）→ [`d_grand/INDEX.md`](d_grand/INDEX.md) |
| `experiments/` | 797M | `paper_run_20260519`（**FLUX 实验跑**，仅参考行，论文主 backbone 是 Klein） |
| `fig/` | 74M | 21 张论文图副本 + **已作废 `README_figures.md`**（数字以 CANONICAL 为准） |
| `viz/` `eval_gallery/` `benchmark_ot_v3/` `i2i_param_search/` `slt_gallery/` | 43M/28M/29M/6M/32K | 画廊 / 可视化 HTML |
| `logs/` | 2.4M | 运行日志（含从 scripts/ 归整来的旧 .log） |

### 📄 文档 Docs
| 路径 | 说明 |
|---|---|
| `README.md` | 本文（总入口 + 目录分类） |
| `docs/ARCHITECTURE.md` | 方法与管线技术文档 |
| `CLAUDE.md` | Agent 行为规则 / 项目速览（Critical Rules 在此） |
| `EXPERIMENTS_NEEDED.md` | 实验需求清单 |
| `scripts/ cores/ results/` 的 `README.md` · `d_grand/INDEX.md` | 各目录索引 |
| `experiment-log/` | algo 微日志（jsonl） |
| `.agent_reports/` | 整理报告（如 `save_tidy_2026-06-28.md`，gitignored） |
| `blog/` | 对外 writeup 物料：blog HTML(EN/ZH) + zip + 旧 paper PDF 导出（gitignored；28M PDF 为旧导出，可删候选） |

### 🗄️ 归档 Archive（旧快照，gitignored）
| 路径 | 大小 | 说明 |
|---|---|---|
| `archive_20260509/` | 1.8G | 旧版快照（含旧 d_grand/figures/docs） |
| `archive_20260513/` | 250M | benchmark 变体快照 |
| `archive_eval/` | 96K | 旧 eval 脚本 |

### ⚙️ 系统/工具
`.git/` · `.claude/`（工具状态，勿动）· `__pycache__/` · `.gitignore`

## 与论文库 `items/sec` 的关系

| | `items/save/...`（本仓库） | `items/sec` |
|---|---|---|
| 职责 | 跑实验、生成图、算指标 | 写论文、拼图表 |
| 图脚本 | 实验/评测脚本（`scripts/`） | 图表**拼装**脚本（`sec/scripts/`、`sec/fig/`） |
| 真源 | 底层 CSV/JSON（`results/`） | 数字 `CANONICAL.md` + 溯源 `provenance/` |

工作流：从 `sec` 现状提需求 → 在本仓库跑 → 图回填 `sec`。改某张论文图先查 `sec/provenance/<图>` 的 `gen_script_current`。

## 约定

- **解释器**：T2I/I2I = `flux` env；eval/分析 = `sensenova` env（无 diffusers）。
- **排除样本**：`sty04, sty14, cnt19`（pathological）。
- **gitignore 策略**：重型/可重生目录（`d_grand/ experiments/ archive_*/ viz/ logs/ blog/` 等）+ 根目录散落二进制（`*.png/*.pdf/*.html`）+ `*.log` + agent 目录均不入库；git 只跟踪 code + `results/` + `fig/` + 文档。
- **已知坑**（复现前必看）：多个脚本含硬编码绝对路径 / 个别 Windows `D:\` 残留 → `items/sec/provenance/GAPS.md §E`。
