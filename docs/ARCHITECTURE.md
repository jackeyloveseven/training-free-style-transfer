# ARCHITECTURE — 方法与管线技术文档

> 本文件描述**方法原理 + 代码管线**，是比 `CLAUDE.md` 更完整的技术参考。
> `CLAUDE.md` 保留作 agent 行为规则（"勿改 production 文件"等）。
>
> **数字真源**：所有定量指标（CSD / CLIP-T / CLIP-I 等）的唯一真值表为
> `/home/ubuntu/chengyanli/items/sec/results/CANONICAL.md`（及其引用的 `results/*.csv`）。
> 本文档**只给指针**，不抄具体数值，避免出现第二份会过期的表。
>
> **排除样本**：`sty04`, `sty14`, `cnt19`（病态边缘样本），所有 sweep / 评测 / 报告一律剔除。

---

## 1. 概述

本仓库实现 **training-free（免训练）风格迁移**：MM-DiT 骨干**全程 frozen**，不更新任何网络权重；
风格迁移完全发生在**去噪轨迹中对 latent 的在线优化**上。

核心思想：在每一步去噪时，让当前 latent 的 **self-attention K/V 分布**对齐风格参考图。
深层用 **Sinkhorn（最优传输 / OT）attention** 做匹配（双随机传输计划，避免 mode collapse），
浅层保留标准 attention（保结构）；并用 latent 空间的统计对齐（AdaIN / SLT）做预对齐。

**Backbone 角色：**

| Backbone | 角色 | 框架 |
|---|---|---|
| **Klein 9B（FLUX.2）** | **论文主 backbone**（headline 表 / 图拼块取自 `d_grand` 的 Klein/eval_all） | DiffSynth-Studio `Flux2ImagePipeline` |
| FLUX.1-dev | 另一组 T2I 入口 | diffusers `FluxPipeline` |
| FLUX.1-Kontext-dev | 另一组 I2I 入口（内容图条件化） | diffusers `FluxKontextPipeline` |

> 注：图/表引用以 `CANONICAL.md` 为准；其中明确"图拼块只取 Klein/eval_all，禁用 FLUX 老版 paper_run"。

---

## 2. 四个 custom_main_* 入口

仓库有两条管线（T2I / I2I）× 两组 backbone（FLUX 家族 / Klein 9B），共 4 个入口：

| 入口 | Backbone | 任务 | Pipeline 类 | 算法核心位置 |
|---|---|---|---|---|
| `custom_main.py` | FLUX 家族（diffusers） | **T2I** | `AttDistPipeline`（继承 `FluxPipeline`），见 `mypipeline.py` | `cores/core_ot_attn.py` |
| `custom_main_img.py` | FLUX.1-Kontext-dev | **I2I** | `AttDistImgPipeline`（继承 `FluxKontextPipeline`），见 `mypipeline_img.py` | `cores/core_ot_attn_img.py` |
| `custom_main_klein9b_t2i.py` | **Klein 9B（FLUX.2）** | **T2I** | DiffSynth `Flux2ImagePipeline`（自包含：hook / SLT / Adam 都在文件内） | 文件内（无外部 core） |
| `custom_main_klein9b.py` | **Klein 9B（FLUX.2）** | **I2I（edit 条件化）** | DiffSynth `Flux2ImagePipeline`（自包含） | 文件内（无外部 core） |

要点：

- **FLUX 家族两个入口（`custom_main.py` / `custom_main_img.py`）**走 diffusers 管线，
  在 `mypipeline*.py` 的 `sample()` 主循环里逐步 `importlib` 调用 `cores/core_ot_attn*.py:step()`。
  算法与骨干解耦，可通过 `CORE_TYPE` 环境变量切换 core。
- **Klein 两个入口（`custom_main_klein9b*.py`）**走 DiffSynth `Flux2ImagePipeline`，
  attention hook、SLT、Adam loop **全部内联在脚本里**（`register_klein9b_attn_hooks` monkey-patch
  `Flux2DiT` 的 attention，layout 为 `[B,S,H,D]`，内部转成 `[B,H,S,D]` 复用 `sinkhorn_attn`）。
- T2I 与 I2I 的差别只在**第 1 步去噪**：I2I 把 content latent 沿 sequence 维 **concat** 到
  hidden_states（Kontext / Klein-edit 式条件化），noise 预测后切回前半段；
  风格 forward 与 Adam forward **只作用在 latent 上，与 T2I 完全一致**。

> 一致性提示：`custom_main.py` 文件内 `model_name` 当前指向 `FLUX.1-Kontext-dev`，
> 而 `CLAUDE.md` 记录 T2I 论文 backbone 为 `FLUX.1-dev`。运行 FLUX T2I 时请按需核对 `model_name`，
> 以 `CLAUDE.md` 的 optimal 配置为准。

---

## 3. 单步算法（一次去噪步内做了什么）

下述顺序对应 `cores/core_ot_attn*.py:step()`（FLUX）与 `custom_main_klein9b*.py` 内联 step（Klein）。
当 `t >= start_time` 时**只做标准去噪、直接返回**（高噪声阶段不优化）；`t < start_time` 才进入风格优化。

1. **Denoise step（标准去噪）**
   frozen transformer 预测 noise → `scheduler.step`。
   I2I 在此处把 `content_latent_packed` concat 进 hidden_states（Kontext / Klein-edit 条件化），
   预测后切片 `[:, :latent_seq_len]`。

2. **Style forward 取 ref K/V**
   把"加噪到下一时间步"的风格 latent 过一次 frozen transformer，
   通过 attention hook 缓存所选层（`self_layers`）的 reference K/V。
   支持 **K-shot**（`KSHOT>1`）：多次采样风格 latent，K/V 取均值（manifold expectation）；
   `NOISE_MODE` 控制采样方式（std / A=统计重采样 / B=latent bank）。

3. **GRAG（Gradient/Reference Attention Gating，压缩 ref K/V 空间方差）**
   对每层 ref K/V 的图像 token 朝其空间均值收缩：`k_img = mean + δ·(k_img - mean)`。
   `δ` 随层深线性插值（shallow→deep）。目的：抑制风格图的**布局/版式泄露**（layout leakage），
   只保留风格的统计/纹理。`δ=1.0` 等价关闭（`GRAG_OFF=1`）。
   消融中称 "w/o LRS" = no GRAG。

4. **AdaIN warm-start（预对齐）**
   `utils.feature_adain(latents, style_latent_, alpha)`：把 latent 的 per-token mean/std 朝风格对齐，
   作为 Adam 的起点。I2I 用较弱 `ADAIN_ALPHA=0.3` 以保住 Kontext 内容结构。
   同时记录 `content_anchor`（去噪后、warm-start 前的 latent）用于漂移控制。

5. **Adam 优化 latent**（`iters` 次；FLUX core 强制 `iters>=3`）
   每次 iter：latent 过 transformer → hook 取 `q_list` / `self_out_list`，构造参考输出 `ref_self_out`：
   - **深层（底部 `OT_LAYER_FRAC` 比例）**：`utils.sinkhorn_attn(q, ref_k, ref_v, tau)` —— OT attention；
     `tau` 随去噪进度从 `OT_TAU_MAX→OT_TAU_MIN`（早期软=保结构，后期锐=push 风格）。
   - **浅层**：标准 `scaled_dot_product_attention(q, ref_k, ref_v)` —— 保结构。

   损失项：
   | loss | 含义 | 权重（env） |
   |---|---|---|
   | `style_loss` | `L1(self_out, ref_self_out)`，深层 OT / 浅层标准 | `_cached_style_w`（rank 自适应，`STYLE_W_SCALE` 缩放） |
   | `dist_loss` | latent 的 mean/std 对齐风格 latent | `DIST_W` (0.15) |
   | `content_loss` | `L1(latent, content_anchor)`，随 iter 衰减 1.0→0.5 | `CONTENT_W` (0.6) |
   | `csd_loss` | 解码 Tweedie x0 → CSD style 嵌入与 ref 的余弦距离（**in-loop**，`CSD_W>0` 启用） | `controller.csd_w`（rank 自适应） |

   （另有 `cnt_kv_loss` / `cnt_loop_loss` 等可选项，默认权重为 0，非主路径。）

6. **SLT（Sinkhorn Latent Transport）预对齐**（**Klein 主路径 `OPTIM_MODE=hybrid`**）
   `utils.sinkhorn_latent_transport(...)`：在 latent 空间做一次 OT 匹配 + per-token OT-AdaIN
   （`mode="ot_adain"`，频域感知 cost：用高频残差按纹理而非位置匹配）。
   **零 forward/backward、近乎免费**，把颜色/统计先对齐，让随后的 Adam 用更少 iter 收敛（省 ~2 iter）。
   论文 "Ours" = **Hybrid = SLT 预对齐 + 2 次 Adam**（`OPTIM_MODE=hybrid HYBRID_ITERS=2`）。
   FLUX core 默认是纯 Adam 路径；SLT/Hybrid 的完整实现在 Klein 入口内联。

> 全局收尾：去噪循环结束后，FLUX `mypipeline.py` 会对最终 latent 再做一次 `adain` 全局色彩对齐，
> 再 VAE 解码（对 ODE 轨迹安全）；Klein 入口用 `adain_1d`。

---

## 4. 关键超参（环境变量）

> 默认值以**代码读取处**为准（core / 入口脚本）。同名变量在不同 backbone 默认值可能不同，已分列。

| 变量 | 默认 | 含义 |
|---|---|---|
| `NUM_STEPS` | 25 | 去噪步数 |
| `OVERRIDE_START_TIME` | I2I 870 / FLUX-T2I 800 / Klein-T2I 850 | 时间步阈值：仅当 `t < start_time` 才做风格优化（优先于 core 内 rank 自适应 start_time） |
| `SEED` | 43 | 随机种子 |
| `LR` | 0.01 | Adam 学习率 |
| `ITERS` | FLUX core 内 `max(iters,3)`；Klein-T2I 5 / Klein-I2I 3 | Adam 迭代数（Hybrid 用 `HYBRID_ITERS`） |
| `HYBRID_ITERS` | 2 | Hybrid 模式下 SLT 后的 Adam 次数 |
| `OPTIM_MODE` | Klein `adam`（论文用 `hybrid`） | Klein 优化模式：`adam` / `hybrid` / `slt` / `inject` / `scfg` / `sds` |
| `CSD_W` | 0.5 | CSD in-loop 权重（0=关闭，省显存；>0 需全模型常驻 GPU） |
| `FIXED_LAYERS` | 论文实验副本 = 1 | =1 固定层 `(20,45)` + `iters=2`；=0 启用 legacy rank 自适应层路由（曾致 artifact，仅作对比） |
| `LO_LAYER` / `HI_LAYER` | 30 / 40 | rank 自适应层路由阈值（`FIXED_LAYERS=0` 时生效） |
| `GRAG_DEEP` | I2I 0.5 / FLUX-T2I 0.15 | 深层 GRAG 压缩 δ |
| `GRAG_SHALLOW` | 0.75 | 浅层 GRAG 压缩 δ |
| `GRAG_OFF` | 0 | =1 关闭 GRAG（δ=1.0，消融 "w/o LRS"） |
| `GRAG_FORCE` | — | 强制统一 deep δ（shallow 自动 +0.3/+0.25 上限钳制） |
| `ADAIN_ALPHA` | I2I 0.3 | AdaIN warm-start 强度 |
| `KSHOT` | 1 | 风格 forward 次数（多次 K/V 取均值） |
| `NOISE_MODE` | std（FLUX core）/ A | 风格 latent 加噪方式：std / A=统计重采样 / B=latent bank |
| `OT_LAYER_FRAC` | 0.5 | 底部多少比例的层用 OT attention（其余用标准 attn） |
| `OT_TAU_MAX` / `OT_TAU_MIN` | 0.15 / 0.07 | Sinkhorn 温度随去噪进度的上/下界 |
| `OT_ITERS` | 6 | Sinkhorn 迭代数 |
| `STYLE_W_SCALE` | 1.0 | style_loss 权重的全局缩放 |
| `DIST_W` / `CONTENT_W` | 0.15 / 0.6 | dist / content anchor 权重 |
| `SLT_ALPHA` / `SLT_TAU` / `SLT_ITERS` | 0.2 / 0.02 / 10 | SLT 传输强度 / cost 温度 / Sinkhorn 迭代（Klein） |
| `SLT_GRAG` | 0.7 | SLT 内风格 token 的 GRAG δ |
| `CORE_TYPE` | `core_ot_attn`(T2I) / `core_ot_attn_img`(I2I) | FLUX 管线选用的 core 模块 |
| `PROMPT` | "A panda eat bamboo" / "masterpiece"(I2I) | 文本提示 |
| `CNT_IMAGE` / `STY_IMAGE` | `cnt/00.png` / `sty/27.png` | 内容图（I2I）/ 风格图路径 |

---

## 5. 最优配置

| 项 | T2I（论文主：Klein 9B） | I2I（论文主：Klein 9B edit） | FLUX-T2I | FLUX-Kontext-I2I |
|---|---|---|---|---|
| 入口 | `custom_main_klein9b_t2i.py` | `custom_main_klein9b.py` | `custom_main.py` | `custom_main_img.py` |
| Backbone | Klein 9B (FLUX.2) | Klein 9B (FLUX.2) | FLUX.1-dev（见第 2 节注） | FLUX.1-Kontext-dev |
| Steps | 25 | 25 | 25 | **25** |
| start_time | 850 | 870 | 800 | **870**（CLIP-I vs CSD 的 Pareto 最优） |
| 优化模式 | `hybrid`（SLT + 2 Adam） | adam | adam(core) | adam(core) |
| CSD_W / guidance | 0.5 / 3.5 | 0.5 / 3.5 | 0.5 / 3.5 | 0.5 / 3.5 |
| 分辨率 / 精度 | 512×512 / bf16 | 512×512 / bf16 | 512×512 / bf16 | 512×512 / bf16 |

> "Ours" 在 `CANONICAL.md` 的 method taxonomy 中即 `hybrid2`（SLT + 2 Adam）。
> I2I 中 start_time 越高 → 风格优化步越少 → 内容保持越好；具体 trade-off 曲线见 `CANONICAL.md`。

---

## 6. 北极星指标

| 指标 | 含义 | 方向 |
|---|---|---|
| **CSD** | 风格保真：输出与风格图的 CSD **style 嵌入**余弦相似度 | 越高越好 |
| **CLIP-T** | 文本一致性（T2I）：输出与 prompt 的 CLIP 相似度 | 越高越好 |
| **CLIP-I** | 内容/结构保持（I2I）：输出与内容图的嵌入余弦相似度 | 越高越好 |

- **CSD 模型**：`CSD_CLIP`（ViT-L/14），位于 `/home/ubuntu/chengyanli/items/CSD_Score`
  （权重 `models/ViT-L-14.pt` + `models/checkpoint.pth`）。前向返回 `(content_emb, _, style_emb)`，
  输入按 CLIP mean/std 归一化、resize 224。in-loop `csd_loss` 与离线评测共用同一模型。
- **指标数值**：一律查 `/home/ubuntu/chengyanli/items/sec/results/CANONICAL.md`
  及其引用的 `results/*.csv`（T2I/I2I 的 ablation、baselines、full benchmark、iters/SLT sweep 等）。
  本仓库 `results/` 下为同源 CSV 拷贝，**真值以 CANONICAL 为准**。

---

## 7. 复现入口命令

> 运行解释器二选一，**两套环境勿混**：
> - **T2I / I2I 生成**（需 diffusers / DiffSynth）：`/home/ubuntu/chengyanli/flux/bin/python3`
> - **eval / 分析脚本**：`/home/ubuntu/miniconda3/envs/sensenova/bin/python`（注意：sensenova **无 diffusers**）

```bash
# ── T2I（论文主 backbone：Klein 9B，Hybrid = SLT + 2 Adam）──
OPTIM_MODE=hybrid HYBRID_ITERS=2 NUM_STEPS=25 OVERRIDE_START_TIME=850 \
PROMPT="A panda eat bamboo" STY_IMAGE=sty/27.png \
/home/ubuntu/chengyanli/flux/bin/python3 custom_main_klein9b_t2i.py
# → output_klein9b_t2i.png, style_klein9b_t2i.png

# ── I2I（论文主 backbone：Klein 9B edit 条件化）──
NUM_STEPS=25 OVERRIDE_START_TIME=870 \
CNT_IMAGE=cnt/00.png STY_IMAGE=sty/27.png \
/home/ubuntu/chengyanli/flux/bin/python3 custom_main_klein9b.py
# → output_klein9b.png, content_klein9b.png, style_klein9b.png

# ── 另一组：FLUX T2I / Kontext I2I 入口 ──
PROMPT="A panda eat bamboo" \
/home/ubuntu/chengyanli/flux/bin/python3 custom_main.py            # T2I (FLUX)
CNT_IMAGE=cnt/00.png STY_IMAGE=sty/27.png \
/home/ubuntu/chengyanli/flux/bin/python3 custom_main_img.py        # I2I (Kontext)
```

---

## 8. 关键文件索引

| 文件 | 作用 |
|---|---|
| `custom_main.py` | FLUX T2I 入口（**production，勿改**） |
| `custom_main_img.py` | FLUX.1-Kontext-dev I2I 入口 |
| `custom_main_klein9b_t2i.py` | Klein 9B T2I 入口（论文主，自包含 hook/SLT/Adam） |
| `custom_main_klein9b.py` | Klein 9B I2I（edit）入口（论文主，自包含） |
| `mypipeline.py` | FLUX T2I 管线（`AttDistPipeline`，去噪主循环 + 调度 core） |
| `mypipeline_img.py` | Kontext I2I 管线（`AttDistImgPipeline`） |
| `cores/core_ot_attn.py` | FLUX T2I 单步算法（**production，勿改**） |
| `cores/core_ot_attn_img.py` | Kontext I2I 单步算法 |
| `utils.py` | 共享工具：`register_attn_control`(hook)、`feature_adain`、`sinkhorn_attn`、`sinkhorn_attn_freq`、`sinkhorn_latent_transport`(SLT)、`batch_manifold_align`(Block-Diag OT)、`Controller` / `DataCache` |
| `CSD_Score`（`items/CSD_Score`） | CSD 指标模型（ViT-L/14） |
| `results/` | 同源 CSV 拷贝（真值见 `sec/results/CANONICAL.md`） |
</content>
</invoke>
