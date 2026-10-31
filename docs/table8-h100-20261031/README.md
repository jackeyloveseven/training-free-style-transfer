# Table 8：H100 步数消融与完整复现档案

七组配置均已实测，使用一张 H100 80GB、同一 40 对子集、seed 43、512×512、bf16。六组新增配置和 T25/N2 对照共生成 280 张质量评估图片。

[下载完整 ZIP](https://github.com/DC-Z-lab/training-free-style-transfer/releases/download/table8-h100-20261031-full/hyperion_table8_h100_complete_20261031.zip)。下载后阅读解压目录中的 `README.md`，并用 `SHA256SUMS` 核验包内文件。

ZIP SHA256：`00861ea9ee4a421471b09075ca75c1e5482edd415e268581cab9aa161d0b7b20`。

## 实测结果

数据源：[完整精度 CSV](measured_results.csv)、[表格 TeX](table8_h100.tex)。

| T | N | 有效优化步 | CSD ↑ | CLIP-T ↑ | 秒/张 ↓ |
|---:|---:|---:|---:|---:|---:|
| 15 | 2 | 6 | 0.5114 | 0.2990 | 3.41 |
| 20 | 2 | 9 | 0.6530 | 0.2806 | 4.96 |
| 25（对照） | 2 | 11 | 0.6820 | 0.2737 | 6.07 |
| 35 | 2 | 16 | 0.7241 | 0.2545 | 8.72 |
| 4 | 2 | 1 | 0.0843 | 0.3148 | 0.71 |
| 4 | 4 | 1 | 0.0911 | 0.3146 | 1.06 |
| 8 | 2 | 3 | 0.2563 | 0.3145 | 1.76 |

40 对为风格 00/10/14/20/27/28/33/38 与五个固定提示词的笛卡尔积。CSV 中标准差来自不同配对，不是跨 seed 方差。

耗时取固定 kp0/sty27 配对，3 次预热后 5 次生成的中位数，复用文本嵌入缓存。包含风格图 VAE 编码、去噪及 SLT/Adam 优化和最终 VAE 解码；排除模型启动、未缓存文本编码、存图与指标计算。这些 H100 时间应与历史 H200 时间分别标注。

T4 在原 `t<850` 阈值下只有 1 个有效优化步；增加 N 到 4 未恢复 T25 的风格水平。各 T 没有另跑 Adam-only，因此本表不支持跨 T 预算支配 Adam-only 或 T25 已饱和的结论。

## 可直接审查的脚本

- [run_table8.py](scripts/run_table8.py)：完整七组实验入口，包含 GPU 占用检查、T 参数适配、逐样本记录及计时。
- [collect_measured_results.py](scripts/collect_measured_results.py)：标准库校验器，核验图片哈希、协议、调度与统计汇总。
- [计时说明修正记录](runner_documentation_amendment.json)：GPU 运行后只修正计时描述，原始执行入口和收据仍保存在 ZIP 中。

下面的复算命令在解压后的完整档案根目录运行。仓库中的脚本副本用于直接审查；若从仓库副本运行，请显式传入所有数据路径和 `--baseline results/t2i_ablation.csv`。

## 压缩包内容与使用

包内保存完整执行代码、固定依赖源码、风格/内容输入、280 张原图、逐样本 JSONL、质量均值与原始计时、模型版本与哈希、环境清单、执行日志、审计记录、论文和 rebuttal 的修改稿、PDF 及差异补丁。

约 38.7 GB 的模型权重和完整虚拟环境不入包。包内保留模型来源、版本、字节数和 SHA256，以及环境构建记录。`models/klein9b/` 只带配置和 tokenizer 元数据。重新生成图像需另行取得模型和兼容 GPU 环境；用已有逐样本分数重算表格和验证图像完整性只需 Python 标准库。

在解压目录可执行：

```bash
python3 scripts/collect_measured_results.py \
  --run results/full_001 \
  --smoke smoke_001 \
  --baseline code/results/t2i_ablation.csv \
  --assets provenance/dependency_sources/model_asset_verification.json \
  --output regenerated/results
```

此命令核对图像 SHA256、配置、40 个配对、调度、均值和计时；不从像素重新计算 CSD/CLIP-T。图像重跑、模型下载来源、离线恢复原稿和 LaTeX 编译步骤见包内根 README。

归档实验代码固定为 `435dce070c507c5edb1b955d48eed72bab4e697d`，论文原稿为 `b50a3a16732e8f1648e6fe054c3d879a6db50e8f`。实际执行的 DiffSynth 源码固定为 `e7b5b89851f75b3dae9820551ca2ad09d2395fd6`，不应直接用其当前主分支替换。

新增 Table 8 位于包内论文 PDF 第 14 页；原组件表成为 Table 9。旧稿若干历史 CLIP-T 数字与仓库 CSV 不一致，定位记录随包保存，本次没有替这些旧数字确认来源。
