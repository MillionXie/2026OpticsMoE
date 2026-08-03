# Grocery10 最终版本与复现说明

## 结论

截至服务器现有结果，最高且**有 checkpoint 可加载复现**的版本是：

```text
31 个包装 SKU 通用预训练
→ 替换两个弱类后的 10 SKU 微调
→ packaging-safe stronger augmentation + EMA 续训
→ epoch 159 EMA minimum-training-loss checkpoint
```

对应服务器结果：

| 指标 | Optical Student | Frozen Teacher |
|---|---:|---:|
| Top-1 | 73.46% | 90.77% |
| Top-3 | 91.92% | 99.23% |
| MRR | 0.8362 | 0.9481 |

具体 checkpoint：

```text
runs/qwen3_vl_embedding_2b_grocery10_replaced_continue_epoch141_stronger_augmentation_ema/
  ema_best_train_loss_checkpoint.pt
```

训练日志在 epoch 152 曾观察到 live/EMA Top-1 都为 74.23%，但当时代码遵守“不以 test
选择 checkpoint”，只保存最低训练损失权重，因此 epoch 152 权重没有保留。不能把这个日志峰值
当作当前可复现模型。当前正式模型应报告 epoch 159 EMA 的 73.46%。

## 结构

Vision 和 Language 各使用一套独立的光学系统：

```text
Qwen hidden
→ Linear(D,224) + LayerNorm + Softplus
→ 224×224 token-amplitude field
→ electronic Top-4 router
→ 4×4 MoE16 expert phase plane（每专家 224×224，仅一层）
→ 10 cm propagation
→ square-law detection
→ per-expert LayerNorm + ReLU
→ 重施同一组 routing weight，未选专家置零
→ 986×986 global phase
→ 10 cm propagation
→ CCD 986×986
→ pool/readout 224×224
→ Linear(224,D) + Transformer identity residual
```

Language 最后一个有效 token 的 detector row 再经过：

```text
LayerNorm(224) → Linear(224,64) → L2 Normalize
```

两套 optical stack 加检索 readout 的总可训练参数为 `4,951,848`。Qwen 原生参数冻结。

固定物理参数：532 nm、8 µm、4×4 专家、expert pitch 254（30 像素间隔）、active
986×986、FFT canvas 1026×1026、Top-K=4、expert/global/CCD 相关传播距离为 10 cm。

## 三阶段训练

1. `Grocery31`：964 train / 781 test / 31 gallery，100 epochs；保存的最低训练损失
   checkpoint 位于 epoch 92。
2. 替换后 `Grocery10`：306 train / 260 test / 10 gallery，从 epoch 92 权重出发微调
   50 epochs；最低训练损失 checkpoint 位于 epoch 141，Top-1 71.54%。
3. 从 epoch 141 出发继续 40 epochs；裁剪比例最低 0.75，亮度/对比度扰动 0.20，旋转
   ±10°，不做镜像、MixUp、CutMix、强模糊或擦除；EMA decay=0.99；最终选择 epoch
   159 的最低训练损失 EMA 权重。

最终阶段使用 KD weight 8、retrieval weight 1、gallery weight 0.25、router balance
0.02、importance 0.005；服务器保存的 resolved config 与 checkpoint metadata
记录的 base/router/phase learning rate 分别为 `1e-5 / 2e-5 / 1e-3`。

## 一键复现

从仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.reproduce_best --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_best_reproduction.yaml
```

该命令顺序运行三阶段，自动传递 checkpoint，复用第二、三阶段相同的 Teacher cache，
最后用 EMA best checkpoint 评测并导出光学硬件包。若某阶段的 canonical checkpoint
已经存在，默认跳过该阶段，便于中断后恢复。先检查命令而不执行可加 `--dry-run`。

## 最佳相位、光场与 BMP

训练中每当最低训练损失刷新，都会立即写出：

```text
best_optical_artifacts/live_weights/phase_parameters.pt
best_optical_artifacts/ema_weights/phase_parameters.pt
```

完整训练结束后还会生成：

```text
best_optical_artifacts/
  manifest.json
  weights/phase_parameters.pt
  slm_bmp/vision/
  slm_bmp/language/
  samples/sample_*/
```

每个样本包含原图、Qwen 224×224 输入、router 权重、expert-plane 振幅、OEO 后且与
global phase 共面的 reload 振幅、CCD intensity/readout、原始 `.pt` 和 PNG。

BMP 约定经过自动检查：

- 仿真和 SLM 像素均为 8 µm，所以 `scale_factor=1`；
- 振幅 BMP：1920×1080，986×986 有效区边界 `[467,47,1453,1033]`；
- 相位 BMP：1920×1200，986×986 有效区边界 `[467,107,1453,1093]`；
- 相位按 `phase mod 2π → uint8 [0,255]`；
- OEO 振幅可能大于 1，因此先除以该平面最大值再映射到 uint8，缩放因子逐文件记录；
- 只用整数 nearest-neighbor scaling，不使用双线性插值，不裁剪有效区；
- 每个 BMP 的 SHA256、尺寸、位深、中心边界和归一化因子都记录在 manifest。

对服务器已有最佳 checkpoint 单独补导出：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.export_best_optical_artifacts \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_best_reproduction_stage3_strong_ema.yaml \
  --checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_replaced_continue_epoch141_stronger_augmentation_ema/ema_best_train_loss_checkpoint.pt \
  --sample-count 8
```
