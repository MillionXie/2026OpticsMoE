# Grocery10 Optical Retrieval

这是 10 种包装商品的图像到图像检索实验，不是十分类器。Frozen Qwen3-VL-Embedding-2B 提供 64D teacher embedding；Student 的 Vision 与 Language transformer stack 都由“一层 expert phase + 一层 global phase”的 Optical MoE 替换，最终输出允许正负值的 64D L2-normalized embedding。

## 当前维护版本

| 配置 | 专家布局 | 已保存结果 |
|---|---|---:|
| `grocery10_moe16_best.yaml` | 4×4、16 experts、Top-4、8 µm、active 986 | Top-1 73.46%、Top-3 91.92%、MRR 0.8362 |
| `grocery10_moe4_latest.yaml` | 2×2、4 experts、Top-2、16 µm、active 478 | Top-1 54.23%、Top-3 86.15%、MRR 0.7101 |

历史日志曾出现 74.23%，但对应 checkpoint 未保存，因此正式只报告可加载 checkpoint 的 73.46%。

## 硬件流程

当前正式流程采用“实验室采集、服务器后处理”解耦方式。实验室电脑上的 `hardware_sdk/acquire_folder.py` 只按文件名播放一批振幅 BMP，并保存同名 CCD `.npy`；它不加载模型、不理解层数，也不需要 Torch。采集结果上传到服务器对应层后，由 `hardware_pipeline.py` 生成下一层振幅。四层依次为 Vision expert、Vision global、Language expert、Language global，相位 mask 暂时人工更换。

公共硬件接口见 [共享硬件说明](../hardware_sdk/README.md)，逐层目录和全部命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
