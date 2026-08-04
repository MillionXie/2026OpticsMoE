# Grocery10 Optical Retrieval

这是 10 种包装商品的图像到图像检索实验，不是十分类器。Frozen Qwen3-VL-Embedding-2B 提供 64D teacher embedding；Student 的 Vision 与 Language transformer stack 都由“一层 expert phase + 一层 global phase”的 Optical MoE 替换，最终输出允许正负值的 64D L2-normalized embedding。

## 当前维护版本

| 配置 | 专家布局 | 已保存结果 |
|---|---|---:|
| `grocery10_moe16_best.yaml` | 4×4、16 experts、Top-4、8 µm、active 986 | Top-1 73.46%、Top-3 91.92%、MRR 0.8362 |
| `grocery10_moe4_latest.yaml` | 2×2、4 experts、Top-2、16 µm、active 478 | Top-1 54.23%、Top-3 86.15%、MRR 0.7101 |

历史日志曾出现 74.23%，但对应 checkpoint 未保存，因此正式只报告可加载 checkpoint 的 73.46%。

## 硬件流程

`hardware_automation.py` 按四个平面完成：人工确认共享相位 mask、预加载并播放整批振幅、等待 SLM 可见和稳定、DVP 拍照、CCD/theory 对照、电子后处理、生成下一层振幅，最后输出检索指标和混淆矩阵。

公共硬件 driver 和厂商 SDK 统一放在 `experiments/hardware_sdk/`。主流程暂时不自动控制相位 SLM；独立的振幅/CCD 与相位 SLM demo 见 [共享硬件说明](../hardware_sdk/README.md)。完整命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
