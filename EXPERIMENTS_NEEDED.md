# Experiment Requirements for Paper

> **给实验 agent 看。读完整份文档再开始动手。**
>
> 工作目录: `/home/ubuntu/chengyanli/items/save/attention_20260426_090948/`
> Python: `/home/ubuntu/miniconda3/envs/sensenova/bin/python`
> Backbone: Klein 9B (FLUX.2, 8 double + 24 single = 32 layers, DiffSynth 后端)
> 模型路径: `/mnt/local/klein9b/`（transformer + text_encoder + vae）
> T2I 入口: `custom_main_klein9b_t2i.py`
> I2I 入口: `custom_main_klein9b.py`
> GPU: single H200, CUDA device 0
>
> **金标指标（所有实验统一）**: CSD、CLIP-T + Time(s)
> **CSD 模型路径**: `/home/ubuntu/chengyanli/items/CSD_Score`

---

## 样本集定义

### Style 集合
- `sty/00.png` ~ `sty/39.png`，**排除 sty04**，共 **39 张**
- sty14 已恢复使用

### Ablation/Sweep 子集 (8 styles)
```
["00", "10", "14", "20", "27", "28", "33", "38"]
```

### T2I Content prompts（论文主表，共 6 条）
```
content_id | prompt
-----------+--------------------------------------------------
p0         | A cat sitting on a wooden table
p1         | A sports car on a mountain road
p2         | A field of sunflowers at sunset
p3         | A dog running in a park
p4         | A snowy mountain peak at dawn
p5         | Portrait of a young woman
```

### 30-sample 子集（消融/sweep 用）
```
content_id | prompt
-----------+--------------------------------------------------
kp0        | A panda eat bamboo
kp1        | A castle on a cliff by the sea
kp2        | A cat sitting on a windowsill
kp3        | A field of sunflowers at sunset
kp4        | A portrait of an old fisherman
```

### I2I Content 图集合
- `cnt/00.png` ~ `cnt/18.png`，共 **19 张**

---

## 实验进度

### ✅ 全部完成 (2026-05-21)

| 实验 | 样本数 | 输出位置 |
|------|--------|----------|
| T2I Ablation (hybrid2, adam8, no_slt, no_adam, softmax, freq_off) | 240 | `d_grand/t2i_ablation/` |
| T2I Baselines (text_only, inject) | 60 | `d_grand/t2i_baselines/` |
| T2I Iters Sweep (0,1,2,3,4,5,8) | 315 | `d_grand/t2i_iters_sweep/` |
| T2I SLT Alpha Sweep (0~0.5) | 315 | `d_grand/t2i_slt_sweep/` |
| T2I Adam-Only Sweep (1~8) | 270 | `d_grand/t2i_adam_only_sweep/` |
| T2I Benchmark (full, 39 sty × 6 prompts × 2 methods) | 456 | `d_grand/t2i_benchmark/` |
| I2I Ablation (5 methods × 8 sty) | 202 | `d_grand/i2i_ablation/` |
| I2I Baselines (text_only, inject) | 80 | `d_grand/i2i_baselines/` |
| I2I Benchmark (full, 39 sty × 19 cnt) | 741 | `d_grand/i2i_benchmark/` |
| Spatial Smoothing Proof v2 | 21 viz | `d_grand/spatial_viz/` |
| Attention PCA Viz | 8 | `d_grand/attn_pca_viz/` |
| Timing Benchmark (独占GPU, 5次中位数) | 8 configs | `results/timing_benchmark.json` |

### 汇总输出

- `results/paper_tables_summary.txt` — 论文所有表格数据
- `results/timing_benchmark.json` — 精确计时
- `results/spatial_proof_v2.json` — 空间平滑证明数据

---

## 最优配置

### T2I Hybrid
```bash
SLT_ALPHA=0.10       # SLT 预对齐强度
SLT_TAU=0.02         # Sinkhorn temperature
SLT_ITERS=10         # Sinkhorn normalization iterations
HYBRID_ITERS=2       # Adam 笔触迭代次数
LR=0.03              # Adam learning rate
CONTENT_W=1.5        # Content preservation
STYLE_W=2.0          # Style loss weight
GRAG_DEEP=0.5        # GRAG 深层压缩
GRAG_SHALLOW=0.7     # GRAG 浅层压缩
START_TIME=850       # 时间步阈值
SINKHORN_MODE=freq   # 频域分离 Sinkhorn
FREQ_HI_BOOST=2.0    # 高频增强
```

### I2I Hybrid
```bash
SLT_ALPHA=0.10
HYBRID_ITERS=2
LR=0.03
CONTENT_W=1.5
STYLE_W=2.0
GRAG_DEEP=0.5
START_TIME=870
ADAIN_ALPHA=0.3
```

---

## 30-sample 结果（T2I Ablation 平均）

| Method | CSD ↑ | CLIP-T ↑ | Time ↓ |
|--------|-------|----------|--------|
| **hybrid2** | **0.6471** | **0.2735** | **6.4s** |
| adam8 | 0.5687 | 0.2802 | 19s* |
| no_slt (Adam-2 only) | 0.5143 | 0.2847 | 6.4s |
| no_adam (SLT only) | 0.3261 | 0.2946 | 3.9s |
| softmax | 0.5981 | 0.2769 | 6.4s |
| freq_off | 0.6102 | 0.2751 | 6.4s |

*timing 受 GPU 负载影响，独占 GPU 测量见 `results/timing_benchmark.json`

---

## Spatial Smoothing Proof

### 核心问题
"为什么 backward 产生笔触，而 SLT 不能？"

### 实验设计
在风格优化的第一步，获取每个 latent token (32×32 grid) 的"更新信号"：
- **Backward**: `loss.backward()` 得到的梯度
- **SLT**: Sinkhorn OT 的 per-token 位移
- **Random**: 随机噪声（对照）

然后测量：**相邻 token 的更新方向一致吗？**
指标：cos(update[x,y], update[x+1,y]) 的空间平均值

### 结果

| Method | Neighbor Cosine ↑ | 含义 |
|--------|-------------------|------|
| **Backward gradient** | **0.601 ± 0.037** | 相邻 token 60% 方向一致 → 笔触 |
| SLT update | 0.254 ± 0.021 | 低相关 → 碎点状 |
| Random noise | ≈ 0 | 完全无关（对照） |

**Backward/SLT ratio: 2.4x**

### 物理解释
Backward 之所以空间相关，因为 attention Jacobian 把梯度在空间相邻的 token 之间耦合：
- 相邻 token 的 Q 向量相似 → attend 到相似的 K 集合
- softmax Jacobian 沿着这个 shared attention pattern 传播梯度
- → 相邻 token 收到相关的梯度信号 → 形成笔触

SLT 是 per-token 独立的 OT matching，token i 的分配仅取决于自己与 style tokens 的相似度，与邻居无关。

### 可视化

1. **梯度方向 PCA→RGB 图**: backward 是大片连续色块（空间耦合），SLT 是碎点（独立）
2. **Autocorrelation decay 曲线**: backward 在 distance=8 仍有 cos>0.2，SLT 在 distance=2 就趋近0

输出: `d_grand/spatial_viz/spatial_proof_paper.png`, `d_grand/spatial_viz/autocorrelation_decay.png`

---

## Timing Benchmark

独占 GPU 测量（warmup 3 runs，测 5 次取中位数）：

| Method | Forward Passes / step | Time (median) |
|--------|----------------------|---------------|
| No style (baseline) | 1 | ~2.5s |
| SLT only (α=0.1) | 2 (denoise + style) | ~3.9s |
| Hybrid (SLT + Adam×1) | 3 | ~5.5s |
| **Hybrid (SLT + Adam×2)** | **4** | **~7.5s** |
| Hybrid (SLT + Adam×3) | 5 | ~9.5s |
| Adam only ×2 | 4 | ~7.5s |
| Adam only ×4 | 6 | ~13s |
| Adam only ×8 | 10 | ~22s |

*精确数据见 `results/timing_benchmark.json`（独占GPU跑完后更新）*

论文用 **forward pass count** 作为 cost metric 避免硬件差异。

---

## 方法演进叙事（Paper Method Section）

### Stage 1: Attention Injection (StyleID baseline)
```
self_attn_out = Attention(Q_content, K_style, V_style)
latent = (1-β) × normal_out + β × self_attn_out
```
- 极快 (~2.5s)，CSD ≈ 0.12，仅微弱色调偏移
- 瓶颈：hard replacement，无迭代反馈

### Stage 2: Adam 反向传播优化
```
for iter in range(N):
    forward → Q, self_attn_out
    target = sinkhorn_attn(Q, ref_K, ref_V)
    loss = L1(self_attn_out, target)
    loss.backward()  → Jacobian 空间平滑 → 笔触
    optimizer.step()
```
- CSD 0.57 (8 iters)，真笔触，但慢 (19s)
- 前几步浪费在色彩对齐

### Stage 3: Sinkhorn Latent Transport (SLT)
```
T = Sinkhorn(cosine_sim(content, style) / τ)
transported = T @ style_tokens
latent = (1-α) × content + α × transported
```
- 色彩对齐高效 (CSD 0.33, ~3.9s)，但**无笔触**
- 根因：per-token 独立操作，无 inter-token 空间耦合

### Stage 4: Hybrid — SLT + Adam（最终方案 ★）
```
# Phase 1: SLT 色彩预对齐 (~0.1ms per step)
latent = sinkhorn_latent_transport(latent, style, alpha=0.1)

# Phase 2: Adam 笔触精修（仅需 2 iter）
for iter in range(2):
    loss = attention_OT_loss(latent, ref_K, ref_V)
    loss.backward()
    optimizer.step()
```
- **CSD 0.65, CLIP-T 0.27, Time ~7.5s**
- 比 Adam-8 快 3x，CSD 高 14%
- SLT 做完色彩对齐 → Adam 只需处理纹理 → 2 iter 就够

---

## 消融逻辑

| 消融 | 去掉什么 | 验证什么 | 预期变化 |
|------|---------|---------|---------|
| No SLT | SLT_ALPHA=0 | SLT 预对齐加速 | CSD↓（Adam 2步不够对齐色彩） |
| No Adam | ITERS=0 | Backward 对笔触的不可替代性 | 无笔触，CSD↓ |
| Softmax | Sinkhorn→sdpa | OT 归一化贡献 | CSD 微降 |
| No freq | HI_BOOST=0 | 频域分离贡献 | CSD 微降 |

---

## 全局注意事项

- **排除样本**：sty04 永久排除（sty14 已恢复使用）
- **seed 固定**：所有实验 `seed=43`
- **数值精度**：指标 4 位小数；time_s 1 位小数
- **模型加载**：生成模型 + CSD/CLIP 各加载一次复用
- **Timing**：论文用 forward pass count；wall-clock 仅作参考，需独占 GPU 测量
