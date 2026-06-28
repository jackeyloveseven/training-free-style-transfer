# cores/ — 注意力核（attention cores）说明

本目录存放每条管线在 denoising 每一步实际调用的 `step(**kwargs)` 实现。
入口脚本（`custom_main.py` / `custom_main_img.py`）与 pipeline（`mypipeline.py` /
`mypipeline_img.py`）把 latents、transformer、controller 等打包成 `kwargs`，逐步交给
这里选定的 core 执行「去噪 + 风格优化」。换 core ≈ 换风格迁移算法。

> 指标数值不在本文件给出。任何 CSD / CLIP-I / CLIP-T 等数字以
> `/home/ubuntu/chengyanli/items/sec/results/CANONICAL.md` 为唯一真源，请到那里查对应表/CSV。
> 评测样本统一排除 **sty04、sty14、cnt19**。

---

## 文件总览

| 文件 | 角色 | 管线(T2I/I2I) | 状态 | 关键点 |
|---|---|---|---|---|
| `core_ot_attn.py` | OT v3 主核 | **T2I** | **production（勿改）** | deep 层 Sinkhorn-OT / shallow 层标准 attn；GRAG 压 ref K/V；rank-95 自适应；CSD-in-loop |
| `core_ot_attn_img.py` | OT v3 Kontext 变体 | **I2I** | **production（勿改）** | 与主核同算法，去噪 forward 把 `content_latent_packed` 沿序列维拼到 hidden_states 做 Kontext 条件化；style key 按风格缓存避免跨样本泄露 |
| `core_ot_attn_distill.py` | 截断前向加速实验 | T2I | 实验 | Adam 迭代只跑 blocks `0..end_layer`（truncated forward），前缀块跳过 gradient_checkpointing；自述提速 12–16%、CSD 差 <0.02 |
| `core_ot_attn_v3_backup.py` | 主核备份 | T2I | backup | 与 `core_ot_attn.py` 当前逐字节相同（`diff` 无差异），仅作快照留存 |
| `core_softmax_t2i_exp.py` | softmax 消融 wrapper | T2I | 实验（论文消融） | 进程内把 `utils.sinkhorn_attn` 临时替换成 `F.scaled_dot_product_attention`，再调 `core_ot_attn.step`；不动 production 核 |
| `core_vshuffle_t2i_exp.py` | V-shuffle baseline wrapper | T2I | 实验（论文对照） | 同上 monkey-patch 思路，把 style V token 做固定 seed=42 的确定性 shuffle 后走标准 attn |
| `__init__.py` | 包标记 | — | — | 空文件 |

---

## Critical Rules：production 核勿改

- **`core_ot_attn.py`（T2I）** 与 **`core_ot_attn_img.py`（I2I）** 是已上线、已测的 production 核。
  本仓库 `CLAUDE.md` 明确标注 **DO NOT MODIFY**。改算法请新建实验 core，不要原地改这两个。
- 两个实验 wrapper（softmax / vshuffle）刻意采用「import production 核 + 临时替换 `utils.sinkhorn_attn`
  + `try/finally` 还原」的方式，正是为了**不触碰** production 核，保持论文对照的可比性。仿照即可。
- `core_ot_attn_distill.py` = **truncated-forward 加速实验**，非 production；上线前需对齐质量。
- `core_ot_attn_v3_backup.py` = **备份**，不要当作可运行入口维护。

---

## 核心机制（简述）

每个 denoising step（细节见 `docs/ARCHITECTURE.md`）大致做：

1. **去噪前向** — 标准噪声预测；I2I 额外把 content latent 沿序列维拼入（Kontext 条件化）。
2. **Style 前向** — 把带噪 style latent 过 transformer，从 self-attention 各层取 reference K/V。
3. **GRAG（layout 抑制）** — 把 ref K/V 的图像 token 向其均值收缩
   `k_img = km + delta·(k_img - km)`，按层深在 `grag_delta_shallow → grag_delta_deep` 间插值，
   压掉 ref 的空间结构方差，**防止 style 图的 layout 泄漏**到生成图。delta 由 style 的 rank-95
   自适应（`LO_GRAG`/`HI_GRAG` 阈值），可用 `GRAG_OFF` / `GRAG_FORCE` 覆盖。
4. **分层注意力路由** — 底部 `OT_LAYER_FRAC` 比例的 **deep 层走 Sinkhorn-OT attention**
   （`utils.sinkhorn_attn`，搬运风格分布），**shallow 层走标准 attention**（保结构）。
   tau 随 denoising 进度由 `OT_TAU_MAX` 退火到 `OT_TAU_MIN`（早期软、后期锐）。
5. **AdaIN 暖启 + Adam 优化** — 最小化 `style_loss`(deep 层 OT 输出的 L1) + `dist_loss`(mean/std 对齐)
   + `content_loss`(对 content anchor 的漂移，衰减) + 可选 `csd_loss`(`CSD_W>0`)。

> 当前论文设置走 `FIXED_LAYERS=1`（固定层 `(20,45)`、`iters=2`）；rank-adaptive 层路由仅作
> `FIXED_LAYERS=0` 的对照/调试项。可调环境变量与最优配置见仓库根 `CLAUDE.md`。

---

## 运行解释器（别混）

- T2I / I2I 推理（需 diffusers）：`/home/ubuntu/chengyanli/flux/bin/python3`
- eval / 分析脚本：`/home/ubuntu/miniconda3/envs/sensenova/bin/python`（**该环境无 diffusers，勿跑推理**）

两个 wrapper 因为 import 了 production 核与 `utils`，仍属推理路径，用 `flux` 解释器。
