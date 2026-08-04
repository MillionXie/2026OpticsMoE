# 保存版本审计

## 历史最佳 4×4 MoE16

- checkpoint：`qwen3_vl_embedding_2b_grocery10_replaced_continue_epoch141_stronger_augmentation_ema/ema_best_train_loss_checkpoint.pt`
- checkpoint epoch：159 EMA；
- 260 query + 10 gallery；
- Student Top-1 0.734615、Top-3 0.919231、MRR 0.836157；
- Teacher Top-1 0.907692、Top-3 0.992308、MRR 0.948077；
- 训练配置：`configs/grocery10_moe16_best.yaml`；
- 硬件配置：`configs/grocery10_moe16_best_hardware.yaml`。

保留 `checkpoints/pre_resume_epoch_0141/resume_checkpoint.pt` 作为这段 40-epoch continuation 的复现起点。其余重复的 live/last checkpoint 不作为正式结果。

## 当前最新 2×2 MoE4

- checkpoint：`qwen3_vl_embedding_2b_grocery10_moe4_hardware_robust/best_train_loss_checkpoint.pt`；
- checkpoint epoch：96；
- Student Top-1 0.542308、Top-3 0.861538、MRR 0.710127；
- 配置：`configs/grocery10_moe4_latest.yaml`；
- 硬件配置：`configs/grocery10_moe4_latest_hardware.yaml`。

这两者不是同一结构，checkpoint 不可交叉加载。
