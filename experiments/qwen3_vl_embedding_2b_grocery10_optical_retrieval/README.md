# Grocery10 Optical Retrieval

这是 10 种包装商品的图像到图像检索实验，不是十分类器。Frozen Qwen3-VL-Embedding-2B 提供 64D teacher embedding；Student 的 Vision 与 Language transformer stack 都由一层 expert phase + 一层 global phase 的 Optical MoE 替换，最后输出带正负值的 64D L2-normalized embedding。

## 当前只维护两个版本

| 配置 | 专家布局 | 保存结果 |
|---|---|---:|
| `grocery10_moe16_best.yaml` | 4×4、16 experts、Top-4、8 µm、active 986 | Top-1 73.46%、Top-3 91.92%、MRR 0.8362 |
| `grocery10_moe4_latest.yaml` | 2×2、4 experts、Top-2、16 µm、active 478 | Top-1 54.23%、Top-3 86.15%、MRR 0.7101 |

MoE16 的正式 checkpoint 是：

```text
runs/qwen3_vl_embedding_2b_grocery10_replaced_continue_epoch141_stronger_augmentation_ema/ema_best_train_loss_checkpoint.pt
```

MoE4 的正式 checkpoint 是：

```text
runs/qwen3_vl_embedding_2b_grocery10_moe4_hardware_robust/best_train_loss_checkpoint.pt
```

历史最佳训练日志曾出现 74.23%，但对应 checkpoint 未保存，因此正式只报告可加载的 73.46%。

## 自动硬件实验

`hardware_automation.py` 按四个平面依次完成：人工确认共享相位 mask、SDK 播放整批振幅、等待 40 ms、SDK 拍照、CCD 与理论对照、电子后处理、生成下一层输入，最后输出 retrieval accuracy 与混淆矩阵。

厂商接口位于 `hardware_devices.py`；SDK 二进制位于被 Git 忽略的 `sdk/`。详见 `HARDWARE_DEPLOYMENT.md` 与 `RUN_COMMANDS.md`。
