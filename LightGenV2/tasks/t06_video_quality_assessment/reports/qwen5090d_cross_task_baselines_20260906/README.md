# RTX 5090 D 跨任务大模型 baseline

本报告填补老师表格中红框任务的电子大模型 baseline。五项正式测试均在同一块 NVIDIA GeForce RTX 5090 D 上完成，batch size 为 1，不并行处理样本；模型在一个进程中只加载一次，不做额外显式 warm-up，第一条测试样本计入统计。

## 统一计时与功率口径

- 计时起点：Qwen 原生 Vision Transformer 的第 1 个 block 输入。
- 固定头任务的终点：最终检索排名、关键点热图、显著性图或质量分数已在 GPU 上得到。
- OpenMoji 的终点：自回归文本生成完成且 CPU JSON 解析结束，因此采用 host latency；其余任务采用 CUDA Event latency。
- 主模型时间不含模型/processor 加载、文件读取、MP4/PNG 解码、裁剪缩放、tokenizer、CPU→GPU 传输和 Vision patch embedding。
- GPU 功率由 `nvidia-smi power.draw` 以 20 Hz 采样。`active mean` 是实测平均板卡功率；`rated upper` 按 5090 D 的 575 W 功率上限乘以实测平均时间计算，它是严格上界，不是实测功率。
- `measured energy = active mean power × mean latency`。表格中的 W 是功率，J 是能耗，二者不能混写。

## 结果

| 任务 | 输入 | 主要性能 | 模型时间 mean / median / P95 | 实测平均 / 峰值功率 | 实测能耗 / 575 W 上界 |
|---|---|---|---:|---:|---:|
| LGVQ 空间质量 | 4 帧，448×448 | SRCC 0.6909；PLCC 0.7048 | 63.363 / 69.132 / 81.389 ms | 115.647 / 242.86 W | 7.328 / 36.433 J·video⁻¹ |
| Caltech101 图像检索 | 单图；固定 30 图 gallery | Top-1 99.5%；Top-3 100%；MRR 0.9975 | 26.407 / 25.731 / 28.985 ms | 151.252 / 163.23 W | 3.994 / 15.184 J·query⁻¹ |
| LSP 关键点 | 单图，224×224 | PCK@0.2 0.5114；PCKh@0.5 0.7959 | 10.340 / 9.510 / 16.483 ms | 117.930 / 139.26 W | 1.219 / 5.946 J·image⁻¹ |
| SALICON 显著性 | 单图，224×224 | CC 0.8811；SIM 0.8327；NSS 0.9903；AUC-Judd 0.7731 | 10.176 / 9.687 / 12.544 ms | 117.837 / 126.07 W | 1.199 / 5.851 J·image⁻¹ |
| OpenMoji 语义交互 | 单图 224×224 + 指令 | parse failure 100%；scene exact 0 | 3172.412 / 3168.780 / 3282.391 ms | 168.544 / 175.20 W | 534.693 / 1824.137 J·sample⁻¹ |

完整数值见 [summary.csv](summary.csv)，总览见 [qwen5090d_baseline_overview.png](qwen5090d_baseline_overview.png)。不同任务的主要指标定义不同，不能根据总览图柱高进行跨任务优劣比较。

## 时间质量：16 个视频与光学并行对照

时间质量使用此前方案二的 4 帧五质量词权重，在独占 5090 D 条件下重新完成 558 个 test 视频的同轮时间/功率采样：SRCC 0.7693、PLCC 0.7797，平均 61.606 ms/视频，实测平均功率 116.909 W、峰值 249.92 W。

| 系统 | 工作负载 | 总时间 | 平均功率 | 实测能耗 | 理论上界 |
|---|---|---:|---:|---:|---:|
| 光学六层 | 16 个视频并行 | 9.084 ms | 80.388 W | 0.730 J | 未给出设备额定功率，不能臆造 |
| Qwen 方案二 | 16 个视频顺序执行 | 985.691 ms | 116.909 W | 115.237 J | 566.772 J（575 W） |

对应时间加速约 108.51×；以实测能耗比较约 157.81×；以 GPU 额定能耗上界比较约 776.14×。图见 [temporal16_power_comparison.png](temporal16_power_comparison.png)。

先前表格中的 601.985 J 来自另一轮 1046.928 ms 计时；本报告使用独占 GPU 且时间与功率同轮采集的 985.691 ms。无显式 warm-up 的 5090 D 运行会受首条编译/P-state 与样本序列影响，因此论文如需误差条，应至少再做两次完整重复，而不是混用不同轮的时间和功率。

## OpenMoji 结果如何解释

OpenMoji baseline 严格遵循既有合同：冻结 Qwen3-VL-2B-Instruct，不增加训练头，不微调，以自由文本返回两个 6×6 网格。1000/1000 样本都生成到 192-token 上限且无法解析；另做的 512-token 单样本诊断仍生成到上限并重复无效数组。因此这不是可以通过放宽正则或换 checkpoint 解决的小波动，而说明零样本自由生成不是该任务可用的结构化 baseline。

程序在解析失败时使用全零网格以保持逐样本指标数组完整，所以 `changed-cell accuracy=0.375` 会受到 remove/move 中目标零单元的 fallback 偏置，不能表述为模型具有 37.5% 的真实编辑能力。论文表应优先填写 `parse failure=100%`、`scene exact=0`，并把任务专用读出头作为另一个明确命名的 baseline，而不是悄悄替换本实验。

## 可追溯证据

- 代码提交：`6795a8237769728e597536777a4f4dbcc9ed2b48`
- 环境：Python 3.12.3；PyTorch 2.8.0+cu128；CUDA 12.8；bfloat16；batch 1。
- 5090 D 原始证据包：`qwen5090d_baselines_key_20260906.tar.gz`
- 证据包 SHA256：`f593f8eac267e5743b728c24efb7428bd8a08e252ce444ef85f489962a890365`
- 包内含五任务的逐样本预测/延迟 CSV、原始功率采样、命令、报告，以及时间质量独占 GPU 复测记录。
- 服务器为节省空间采用 sparse checkout；报告中的 `git_worktree_clean=false` 来自 sparse materialization 与只用于导入旧 teacher 的兼容符号链接，正式 baseline 源码本身对应上述 GitHub commit。LSP/SALICON 性能与历史冻结 teacher 结果复算一致。

运行图表：

```powershell
python LightGenV2\tasks\t06_video_quality_assessment\reports\qwen5090d_cross_task_baselines_20260906\plot.py
```
