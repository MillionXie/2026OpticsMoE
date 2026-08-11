# Qwen3-VL Grocery10 Optical Retrieval

这是旧版、已经在真实光路上验证过的商品图像检索工程。当前对外发布的唯一主线是
`configs/release/` 中的 MoE4 两阶段方案；训练、仿真和实物处理不要与后续 token-wise
实验混用。

## 最终模型

```text
RGB 商品图像
→ frozen Qwen3-VL patch/position embedding
→ Vision Optical MoE4
→ frozen Qwen visual bridge
→ Language Optical MoE4
→ frozen final RMSNorm
→ LayerNorm + Linear(2048,64)
→ L2-normalized 64D embedding
→ cosine retrieval
```

Vision 与 Language 的光学结构相同：

```text
电子 Top-2 router
→ 2×2 四专家的一层 phase-only mask
→ 5 cm 传播与平方探测
→ 每个选中专家独立 LN + ReLU
→ 重新乘原 routing weight，未选专家严格置零
→ global phase-only mask
→ 5 cm 传播与 CCD
```

模型没有分类 head；gallery 和 query 都由 Student 编码为 64D 向量。

## 尺寸与光路

- 仿真采样：`16 μm`。
- 单专家：`224×224`。
- 专家 pitch：`254`，即相邻专家间隔 30 pixels。
- 2×2 有效区域：`478×478`。
- FFT canvas：`518×518`，四周各 padding 20 pixels。
- 波长：532 nm。
- expert→CCD、reload→global、global→CCD：均为 5 cm。
- K-space：开启，`theta_max=0.65°`。
- phase raw parameter 零初始化；经 `2π·sigmoid(raw_phase)` 后初始物理相位为 `π`。
- phase DC loss：关闭。
- transformer residual：关闭。

物理 SLM pixel pitch 为 `8 μm`。导出时对仿真振幅/相位做 2× 最近邻扩展，
所以 `478×478 @ 16 μm` 对应 `956×956 @ 8 μm`，再居中放入：

- amplitude BMP：1920×1080；
- phase BMP：1920×1200。

## 可复现结果

推荐 MoE4 checkpoint：

```text
runs/release_moe4_grocery10_epoch40/ema_last_checkpoint.pt
```

- Top-1：67.69%
- Top-3：87.31%
- MRR：79.16%
- Teacher Top-1：90.77%
- 260 个 test queries，10 张 gallery images。
- checkpoint 是固定绝对 epoch 40 的 EMA；测试集没有参与损失或 checkpoint 选择。

历史 MoE16 达到 Top-1 73.46%，但使用 4×4/16 专家、986×986 有效区，不能与
MoE4 checkpoint、CCD capture 或 phase mask 混用。它只保存在
`configs/archive/historical_moe16_best.yaml` 作为论文数值和结构对照。

## 目录约定

- `configs/release/model_moe4.yaml`：模型、数据、光路及通用训练参数。
- `configs/release/stage1_grocery31_pretrain.yaml`：31 类包装商品预训练 26 epoch。
- `configs/release/stage2_grocery10_finetune.yaml`：目标 10 SKU 继续训练 14 epoch。
- `configs/release/hardware_moe4.yaml`：BMP、CCD 注册及逐层硬件微调参数。
- `reproduce_release.py`：一键完成两阶段训练、评测和可视化。
- `hardware_pipeline.py`：从 checkpoint 生成四个物理平面的 BMP，并处理逐层 CCD。
- `hardware_finetune.py`：用某层真实 CCD 固定上游、微调其下游。
- `experiments/hardware_sdk/`：独立的 SLM 播放和相机采集工程。

完整命令见 `RUN_COMMANDS.md`，所有结构参数说明见 `ARCHITECTURE.md`。
