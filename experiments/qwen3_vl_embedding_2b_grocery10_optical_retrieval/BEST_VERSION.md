# Best-version audit

## 推荐发布和实物部署版本：MoE4

- 配置：`configs/release/`。
- 训练：Grocery31 预训练 26 epoch，再在 Grocery10 继续 14 epoch。
- checkpoint：`runs/release_moe4_grocery10_epoch40/ema_last_checkpoint.pt`。
- 权重：绝对 epoch 40 EMA。
- 结果：Top-1 0.676923、Top-3 0.873077、MRR 0.791590。
- 可训练参数：1,794,672。
- 物理结构：Vision/Language 各一层 MoE4 expert phase 加一层 global phase。
- 可部署性：与 956×956 active-area 的 8 μm SLM 导出和已有四平面 CCD 流程一致。

原日志曾在 epoch 40 出现 live Top-1 0.696154，但该 live snapshot 没有保留下来，
不能将它写成可复现 checkpoint 结果。

## 历史最高软件结果：MoE16

- 配置：`configs/archive/historical_moe16_best.yaml`。
- checkpoint（服务器归档）：
  `runs/archive/qwen3_vl_embedding_2b_grocery10_replaced_continue_epoch141_stronger_augmentation_ema/ema_best_train_loss_checkpoint.pt`。
- 结果：Top-1 0.734615、Top-3 0.919231、MRR 0.836157。
- 可训练参数：4,951,848。
- 限制：4×4/16 专家，与当前 MoE4 的面板、BMP、CCD 和 checkpoint 均不兼容。

因此，“数值最高”是 MoE16，“最终可复现且用于真实光路”是 MoE4。对外分享时默认使用 MoE4。
