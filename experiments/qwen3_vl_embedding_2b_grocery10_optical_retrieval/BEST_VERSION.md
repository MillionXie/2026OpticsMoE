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

## 当前推荐 2×2 MoE4：Grocery31 → Grocery10

- checkpoint：`qwen3_vl_embedding_2b_grocery10_moe4_from31_epoch40_replay/ema_last_checkpoint.pt`；
- checkpoint：目标微调绝对 epoch 40 EMA；
- 260 query + 10 gallery；
- Student Top-1 0.676923、Top-3 0.873077、MRR 0.791590；
- 31-SKU 预训练配置：`configs/optimization/grocery31_moe4_pretrain.yaml`；
- 10-SKU 微调配置：`configs/optimization/grocery10_moe4_from31_strong_ema.yaml`；
- 硬件配置：`configs/grocery10_moe4_from31_hardware.yaml`。

训练过程第一次运行曾在 live epoch 40 观察到 Top-1 0.696154；该数值参与了
训练轮次判断，且当轮 checkpoint 未保留，所以只作为 selection-biased 诊断值，
不冒充当前已保存 checkpoint 的指标。上方 0.676923 是对已保存 EMA checkpoint
重新完整评测得到的可复查结果。

## 原始 2×2 MoE4 从零训练基线

- checkpoint：`qwen3_vl_embedding_2b_grocery10_moe4_hardware_robust/best_train_loss_checkpoint.pt`；
- checkpoint epoch：96；
- Student Top-1 0.542308、Top-3 0.861538、MRR 0.710127；
- 配置：`configs/grocery10_moe4_latest.yaml`；
- 硬件配置：`configs/grocery10_moe4_latest_hardware.yaml`。

MoE16 与 MoE4 不是同一结构，checkpoint 不可交叉加载；两个 MoE4 版本物理
结构相同，但新硬件实验必须重新加载新 mask，不能混用旧 mask 对应的 CCD。
