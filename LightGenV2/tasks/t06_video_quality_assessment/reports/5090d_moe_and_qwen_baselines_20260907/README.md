# RTX 5090 D：光学 MoE 分段电处理与 Qwen baseline 汇总

本报告把两类数据放到同一个可追溯入口：

1. T01、T02、T03、T04、T06 光学 MoE 在 CCD 边界之后的电子处理、并行残差、跨层重装和任务头时间；
2. 同一型号 RTX 5090 D 上已经归档的冻结 Qwen baseline 性能、模型核心时间、平均/峰值功率和能量。

最重要的结论是：不能再把所有任务统一写成“六层 9.084 ms”。T01、T04、T06 是 4 次特征传播 + 2 次光路由，共 6 次光场；T02、T03 没有 language 路径，只有 2 次特征传播 + 1 次光路由，共 3 次。加入当前计算图真实的 CCD 后电处理后，临界路径分别为：

| 任务 | 路径 | 每次物理调用负载 | 光场次数 | CCD 后特征处理平均中位数 | 并行残差平均中位数 | 尾部任务头 | 估算临界路径 |
|---|---|---|---:|---:|---:|---:|---:|
| T01 Caltech101 | Vision + Language | 1 query | 6 | 0.410 ms | 0.275 ms | 0.075 ms | **10.061 ms/query** |
| T02 LSP | Vision only | 1 image | 3 | 0.408 ms | 0.281 ms | 0.548 ms | **5.537 ms/image** |
| T03 SALICON | Vision only | 1 image | 3 | 0.397 ms | 0.275 ms | 0.693 ms | **5.654 ms/image** |
| T04 OpenMoji | Vision + Language | 1 image + instruction | 6 | 0.394 ms | 0.266 ms | 0.815 ms；另有 bridge 0.132 ms | **10.862 ms/sample** |
| T06 LGVQ 16×4 | Frame vision + video sequence | 一幅光场并行 16 视频×4 帧 | 6 | 2.193 ms/field | 0.114 ms/field | 0.519 ms/field；另有 bridge 0.272 ms | **28.744 ms/16 videos**（1.796 ms/video 折算） |

这里的“CCD 后特征处理平均值”按各任务所有特征层取平均：中间专家层包含 CCD 规范化、光电融合以及下一层 SLM 场重建；该模态最后一层只包含 CCD 规范化和融合，不再虚构一次不存在的 SLM 重装。所有精确分量的 median/P95 见 `optical_component_timings_5090d.csv`。

## 临界路径怎么计算

每次物理光场采用当前给定常数：

```text
传播 0.714 ms + 相位 SLM 0.100 ms + CCD 0.500 ms = 1.314 ms/pass
```

特征层不是简单把残差相加。电残差与光路并行，因此：

```text
feature pass = max(1.314 ms, electronic residual) + serialized post-CCD work
router pass  = 1.314 ms + router post-processing / expert-field construction
total        = all router passes + all feature passes + bridge (if any) + task head
```

本轮所有残差中位数都小于 1.314 ms，所以均可被物理光路覆盖，不增加临界路径。但残差仍然消耗 GPU 电能，不能在“完整光电系统能耗”中删除。

T06 的 28.744 ms 是当前严格 16×4 整幅场软件图的结果，不是光学传播变慢：其中 frame router 需要处理 64 个 frame lane，video router 处理 16 个 video lane；当前 eager 实现的两个 router 后处理分别约 8.988 ms 和 2.308 ms。这个数值反映当前软件实现，可以优化，但本报告没有用理想并行或预拼接替换它。

## 测量边界与数据量

- GPU：NVIDIA GeForce RTX 5090 D，33,670,758,400 bytes 显存；PyTorch 2.8.0+cu128，CUDA 12.8。
- 数值格式：float32；`torch.inference_mode()`；eager execution；没有 `torch.compile`。
- 每个分量先 50 次 warm-up，再正式测 1000 次；每一次正式调用都同时记录 CUDA Event 和完成同步后的 wall time。
- T01–T04：每个分量 1000 个逻辑样本、1000 个物理场。
- T06：每个分量 1000 个物理场；每场 16 个逻辑视频，因此等价观察 16,000 个 video workload。这里不能把 16,000 误写成 16,000 次独立光路传播。
- 计时输入使用与正式计算图一致的张量形状和模块边界，不包含磁盘读取、MP4/PNG 解码、Qwen 前置 patch embedding，也不包含真实 SLM/CCD 驱动的 USB/PCIe 抖动。
- 原始 component 结果同时报告 median 和 P95；临界路径使用各 component 的同步 wall median 组合，是计算图估算，不是实验台端到端实测。

## Language 与 Vision 的对应关系

| 任务 | Vision 路径 | Language/sequence 路径 | 任务尾部 |
|---|---|---|---|
| T01 | 196 tokens，专家层 + 全局层 | 76 tokens，专家层 + 全局层 | 最后一条有效 language row → LN → Linear-64 → L2 embedding |
| T02 | 196 tokens，专家层 + 全局层 | 无 | 恢复 14×14 空间布局 → 14 通道 56×56 关键点热图 |
| T03 | 196 tokens，专家层 + 全局层 | 无 | 恢复 14×14 空间布局 → 1 通道 224×224 显著图 |
| T04 | 196 vision tokens | 64 language tokens | language mean/max bridge + 三个 residual blocks → category/edit/task 输出 |
| T06 | 16×4×49 frame tokens | 38 prompt tokens 合并为每视频 42-token sequence | 16 个视频各输出一个 MOS |

T05、T07、T08 当前没有已经冻结并可运行的正式光学 MoE 计算图，因此没有填伪造时间；等各自计算图和输入合同冻结后，使用同一 profiler 增加对应 profile。

## 能耗口径

本报告同时保留两个光学能量数字：

- `physical-only optical energy`：`80.388 W × 光场次数 × 1.314 ms`。六次光场为 **0.633779 J/call**，三次为 **0.316889 J/call**。
- `optical-rig wall proxy`：假设 80.388 W 的光学设备在整个临界路径持续上电，使用 `80.388 W × critical path`。T01/T02/T03/T04/T06 分别为 **0.809/0.445/0.455/0.873/2.311 J per physical call**。

第二项仍然只是光学设备墙上能量代理，不是完整光电系统能耗。CCD 后处理、残差和任务头运行在 5090 D 上；本轮只测了它们的时间，没有同步采集 GPU 平均功率。因此不能把上面的代理值与 Qwen 的完整 GPU 实测能量直接宣称为端到端节能倍数。完整光电能耗还需在最终串联 runner 中同步采样 GPU power，并加入激光器、两块 SLM、CCD、控制机的 idle/active 功率。

旧的 9.084 ms 六层数值对应 `80.388 W × 9.084 ms = 0.730245 J`。它可以作为既往硬件总时长参考；新的任务级数字用于解释真实计算图。两者不能在同一列中混用。

## 与冻结 Qwen baseline 对照

| 任务 | Ours 性能 | Qwen baseline 性能 | Ours 当前计算图时间 | Qwen 模型核心时间 | Qwen 实测平均功率 / 能量 |
|---|---|---|---:|---:|---:|
| T01 | Top-1 0.9000 | Top-1 0.9950 | 10.061 ms/query | 26.407 ms/query | 151.252 W / 3.994 J |
| T02 | PCK@0.2 0.5773 | PCK@0.2 0.5114 | 5.537 ms/image | 10.340 ms/image | 117.930 W / 1.219 J |
| T03 | CC 0.8291 | CC 0.8811 | 5.654 ms/image | 10.176 ms/image | 117.837 W / 1.199 J |
| T04 | changed-cell 0.9800 | scene exact 0；parse failure 100% | 10.862 ms/sample | 3172.412 ms/sample | 168.544 W / 534.693 J |
| T06 temporal | SRCC 0.8044 | SRCC 0.7693 | 28.744 ms/16 videos | 985.691 ms/16 sequential videos | 116.909 W / 115.237 J |

T04 两列不是同一个主指标，不能据此画性能提升结论。T06 的 Qwen 数字是单视频模型执行 16 次之和；Ours 是一幅物理场同时承载 16 个视频。对照的负载数量一致，但并行方式不同，必须随表注明。

完整 baseline 的 median/P95、额定 575 W 上界、输入尺寸和性能次指标仍以相邻归档 `../qwen5090d_cross_task_baselines_20260906/summary.csv` 为准。本目录的 `baseline_and_moe_summary.csv` 只把论文常用字段整理到同一张表。

## 图表与复现

- `optical_moe_stage_timing.png/.pdf`：CCD 后串行电处理、并行残差与任务头；并显示 1.314 ms 覆盖线。
- `moe_qwen_comparison.png/.pdf`：相同任务负载下的模型核心时间，以及性能/能量口径提示。

重新绘图：

```powershell
python LightGenV2\tasks\t06_video_quality_assessment\reports\5090d_moe_and_qwen_baselines_20260907\plot.py
```

重新测量（必须在 5090 D 上）：

```bash
python LightGenV2/scripts/profile_optical_electronics_5090d.py \
  --tasks t01 t02 t03 t04 t06 \
  --warmup 50 \
  --repeats 1000 \
  --output-dir LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/optical_moe_electronics_5090d_20260907
```

源码 commit、原始证据包及 SHA256 见 `evidence_manifest.json`。
