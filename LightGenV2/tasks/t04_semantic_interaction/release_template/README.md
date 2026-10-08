# OpenMoji：我们的光电模型，独立仿真复现包

先看 `COMMAND.md`，接手的 AI 再看 `AI_README.md`。本包是已恢复的 **standard 原读出头**，不是 slim 小头；只交付我们的 best 模型，不包含 Qwen / D2NN baseline 权重，也不是实验室硬件控制包。

## 包含内容

| 位置 | 用途 |
| --- | --- |
| `reproduce.py` | 唯一推荐入口：校验、单例、全测试、重新训练 |
| `weights/best_checkpoint.pt` | epoch 40 完整模型 state_dict，包括冻结视觉输入层 |
| `weights/phases.pt` | 从同一 best 导出的物理相位，弧度，供分析；不是另一个训练 checkpoint |
| `frontend/model.safetensors` | 从 best 中抽出的冻结 Qwen patch / position 参数，用于构建模型 |
| `data/` | 5000 train + 1000 test，固定清单、图像、相应指令的原始词嵌入缓存 |
| `assets/` | 16 类 OpenMoji 图标，用于渲染预测 |
| `settings.json` | 原训练参数；运行入口自动重定位路径 |
| `reference/` | 原 best 的全部指标、逐样本预测、架构、训练历史、去光结果与可视化 |
| `MANIFEST.json` | 全部交付文件的 SHA256、打包 commit 和原训练 commit |
| `outputs/` | 执行后生成；不会改写原权重、参考指标和数据 |

`experiments/` 和 `LightGenV2/` 是实际依赖的源代码，必须随包保留。部分公共实现沿用 Caltech 等历史包名，不代表需要那些任务的数据、权重或训练服务。

## 应复现的结果

固定 best，在随包 1000 张 test 上：

| 指标 | 数值 |
| --- | ---: |
| 修改格准确率 changed_cell_accuracy（主要指标） | 0.8715 |
| 全格准确率 cell_accuracy | 0.987861 |
| 编辑区域 IoU | 0.832733 |
| Object F1 | 0.933949 |
| 整场景完全正确 scene_exact_match | 0.6900 |

不要把 98.79% 全格准确率当成 87.15% 修改格准确率：背景和未编辑格很容易正确。完整分操作指标在 reference JSON 中。

本次使用定期 test 选 best，没有独立 validation，报告显式标记 `selection_biased=true`；这是固定协议下复现结果，不能描述为完全未参与选模的独立测试。重新训练受设备、浮点数和随机性的影响，不承诺逐位相同。

## 输出的边界

任务是有限图标集合的文字条件 **网格编辑**，不是开放世界图像生成。模型从 source RGB 和文字 embedding 预测“哪格改、改成哪类”；后处理还使用已知 `source_grid` 保留未编辑格，最后用图标重绘。目标网格只用于 loss / 评分，不输入模型。

本包可离线复现随包场景、训练及测试指令；文本缓存只覆盖这些指令。任意新指令需要用同一 Qwen tokenizer 和原始 embed_tokens 重新制备缓存。本包没有携带完整词表权重或完整 Transformer，不能假装支持任意新文本。

这里不重新生成数据、不联网寻找权重、不调用其他服务器；安装 Python 依赖后即可离线执行。硬件 SDK、CCD 标定、SLM LUT、设备驱动不在此次交付范围内。
