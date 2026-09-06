# RTX 5090 D 跨任务大模型 baseline

本报告填补老师表格中红框任务的电子大模型 baseline。五项正式测试均在同一块 NVIDIA GeForce RTX 5090 D 上完成，batch size 为 1，不并行处理样本；模型在一个进程中只加载一次，不做额外显式 warm-up，第一条测试样本计入统计。

## 统一计时与功率口径

- 计时起点：Qwen 原生 Vision Transformer 的第 1 个 block 输入。
- 固定头任务的终点：最终检索排名、关键点热图、显著性图或质量分数已在 GPU 上得到。
- OpenMoji 正常 baseline 的终点：普通结构化任务头产生 `6×6` 类别与编辑网格；不包含
  自回归生成和 CPU JSON 解析。旧生成路径单独保留为 zero-shot diagnostic。
- 主模型时间不含模型/processor 加载、文件读取、MP4/PNG 解码、裁剪缩放、tokenizer、CPU→GPU 传输和 Vision patch embedding。
- GPU 功率由 `nvidia-smi power.draw` 以 20 Hz 采样。`active mean` 是实测平均板卡功率；`rated upper` 按 5090 D 的 575 W 功率上限乘以实测平均时间计算，它是严格上界，不是实测功率。
- `measured energy = active mean power × mean latency`。表格中的 W 是功率，J 是能耗，二者不能混写。

## 结果

| 任务 | 输入 | 主要性能 | 模型时间 mean / median / P95 | 实测平均 / 峰值功率 | 实测能耗 / 575 W 上界 |
|---|---|---|---:|---:|---:|
| LGVQ 空间质量 | 4 帧，448×448 | SRCC 0.6909；PLCC 0.7048 | 63.363 / 69.132 / 81.389 ms | 115.647 / 242.86 W | 7.328 / 36.433 J·video⁻¹ |
| Caltech101 图像检索 | 单图；固定 30 图 gallery | Top-1 99.5%；Top-3 100%；MRR 0.9975 | 26.407 / 25.731 / 28.985 ms | 151.252 / 163.23 W | 3.994 / 15.184 J·query⁻¹ |
| LSP 关键点 | 单图，224×224 | **PCK@0.2 0.7217；PCKh@0.5 0.8846** | **9.504 / 9.470 / 9.632 ms** | 140.128 / 145.03 W | 1.332 / 5.465 J·image⁻¹ |
| SALICON 显著性 | 单图，224×224 | CC 0.8811；SIM 0.8327；NSS 0.9903；AUC-Judd 0.7731 | 10.176 / 9.687 / 12.544 ms | 117.837 / 126.07 W | 1.199 / 5.851 J·image⁻¹ |
| OpenMoji 语义交互 | 单图 224×224 + 指令 | **changed-cell 0.5475；IoU 0.2909；exact 0.0160** | **27.166 / 26.628 / 30.280 ms** | 171.370 / 172.68 W | 4.656 / 15.624 J·sample⁻¹ |

完整数值见 [summary.csv](summary.csv)，总览见 [qwen5090d_baseline_overview.png](qwen5090d_baseline_overview.png)。不同任务的主要指标定义不同，不能根据总览图柱高进行跨任务优劣比较。

## 时间质量：16 个视频与光学并行对照

时间质量在论文表中固定采用此前方案二的原始 4 帧证据：SRCC 0.7693、PLCC 0.7797，
平均 65.433 ms/视频；16 个视频顺序执行为 1046.928 ms，实测能量 120.680 J。

| 系统 | 工作负载 | 总时间 | 平均功率 | 实测能耗 | 理论上界 |
|---|---|---:|---:|---:|---:|
| 光学六层 | 16 个视频并行 | 9.084 ms | 80.388 W | 0.730 J | 未给出设备额定功率，不能臆造 |
| Qwen 方案二 | 16 个视频顺序执行 | **1046.928 ms** | **115.271 W（由能量/时间反算）** | **120.680 J** | **601.984 J（575 W）** |

对应六层光学硬件时间比约 115.25×；以实测能耗比较约 165.26×；以 GPU 额定能耗上界比较约 824.36×。图见 [temporal16_power_comparison.png](temporal16_power_comparison.png)。

后续独占 GPU 复测的 985.691 ms / 115.237 J 属于另一轮测量，本表不再用它覆盖老师已经采用的
1046.928 ms / 120.680 J。两轮证据都保留，但同一行不会混用不同轮次的时间与能量。

## OpenMoji 结果如何解释

旧 OpenMoji 结果要求冻结 Qwen 自由生成两个 JSON 网格，1000/1000 样本都生成到
192-token 上限且解析失败，所以它只能作为 zero-shot diagnostic，不能代表正常 baseline。

纠正后的正常 baseline 仍冻结全部 Qwen Vision/Language Transformer，只增加一个普通的
1,212,434 参数结构化任务头；训练 50 epoch，每 5 epoch 在同一 1000 test 上评估并按
changed-cell 选择。它得到 changed-cell 0.5475，而不是旧的 0；add/replace/move/remove
分别为 0.224/0.336/0.650/0.980。结果说明冻结表示对删除/移动较易读出，对精确“新增
某类别到某位置”和替换较难。没有用 LoRA、主干微调、额外 Transformer、定制增强或
特殊 loss 抬高数值。

## 可追溯证据

- LSP/OpenMoji 纠正版代码提交：`50294388b4c59a881083a1b3460d475a4bc4834d`
- 环境：Python 3.12.3；PyTorch 2.8.0+cu128；CUDA 12.8；bfloat16；batch 1。
- 旧五任务证据包仍保留；纠正后的 LSP/OpenMoji run 另行归档，见统一报告的 evidence manifest。
- 服务器为节省空间采用 sparse checkout；`git_worktree_clean=false` 来自 sparse
  materialization 与 run 证据目录，不代表正式源码被私改。

运行图表：

```powershell
python LightGenV2\tasks\t06_video_quality_assessment\reports\qwen5090d_cross_task_baselines_20260906\plot.py
```
