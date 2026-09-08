# A100：CCD 电子读出到残差融合的严格边界复测

## 结论

本次不再采用旧报告中的 `paper_serial_path`。计时边界收紧为：

> 已采集的 CCD 强度张量 → 必需的强度归一化/池化 → 可学习电子读出与非线性 → 与已经并行算好的电子残差进行尺度匹配融合。

以下操作全部不计时：CCD/SLM 驱动与曝光、文件 I/O、图像解码、ROI/小块裁切与堆叠、下一层 SLM 排版或画布重建、光 Router 后处理与 fan-out、并行电子残差本身、跨模态/帧序列 bridge、最终任务头/decoder、光学传播。

视频任务的 64/16 个 CCD 小块在计时前已经排好；因此不会再把 Python 循环裁切误算成电子神经网络推理。

## A100 实测结果

每个分量先 warm-up 100 次，再独立测量 1000 次。精度为 FP32、eager `torch.inference_mode()`、无 `torch.compile`。`CUDA` 是纯 GPU kernel 中位数，可作为严格边界下的低估口径；`Wall` 是每次都同步后的主机墙钟中位数，更适合审计。没有使用不稳定的单次 minimum。

| 任务 | 融合 block 数 | CUDA 总时间 / call | Wall 总时间 / call | CUDA 平均 / block | Wall 平均 / block |
|---|---:|---:|---:|---:|---:|
| Caltech101 图像检索 | 4 | 1.929 ms | 1.998 ms | 0.482 ms | 0.499 ms |
| LSP 关键点检测 | 2 | 0.948 ms | 0.982 ms | 0.474 ms | 0.491 ms |
| SALICON 显著性分析 | 2 | 0.951 ms | 0.986 ms | 0.476 ms | 0.493 ms |
| OpenMoji 语义交互 | 4 | 1.923 ms | 1.991 ms | 0.481 ms | 0.498 ms |
| LGVQ 时间一致性，16 视频并行 | 4 | 1.892 ms / 16 视频 | 1.961 ms / 16 视频 | 0.473 ms | 0.490 ms |
| LGVQ 空间一致性，单视频 4 帧 | 4 | 1.892 ms | 1.961 ms | 0.473 ms | 0.490 ms |
| ABO 图搜文 T08 | 4 | 1.923 ms | 1.992 ms | 0.481 ms | 0.498 ms |

## 最终任务头复测

任务头采用相同的 A100、FP32、100 次 warm-up 和 1000 次独立测量。下表先只列真正的最终任务头；LGVQ 和 OpenMoji 为了产生最终输出还需要一个跨模态 bridge，另列而不混入任务头。

| 任务 | 任务头 CUDA 中位数 | 任务头 Wall 中位数 | 必需 bridge CUDA 中位数 |
|---|---:|---:|---:|
| LGVQ 时间一致性，16 视频并行 | 0.860 ms / call | 0.879 ms / call | 0.465 ms |
| LGVQ 空间一致性，单视频 4 帧 | 0.823 ms | 0.841 ms | 0.247 ms |
| ABO 图搜文 T08 | 0.134 ms | 0.150 ms | — |
| LSP 关键点检测 | 0.971 ms | 0.989 ms | — |
| SALICON 显著性分析 | 1.232 ms | 1.250 ms | — |
| OpenMoji 语义交互 | 1.428 ms | 1.447 ms | 0.223 ms |

把任务头加到前面的严格 CUDA 电子时间后，分别为：LGVQ 时间 `2.753 ms/16视频`、LGVQ 空间 `2.716 ms`、ABO T08 `2.057 ms`、LSP `1.919 ms`、SALICON `2.183 ms`、OpenMoji `3.352 ms`。这些值按用户要求尚未加入 bridge；若要求从光学输入一直产生有效最终输出，则 LGVQ 时间、LGVQ 空间、OpenMoji 还必须分别加 `0.465/0.247/0.223 ms`。

LGVQ 时间一致性的 `1.892 ms` 是 16 个视频一次并行 call 的真实墙前 GPU 时间，不应除以 16 后宣称单视频 latency。若只报告吞吐等效值，则 CUDA 为 `0.118 ms/video`，但必须明确写成 throughput-equivalent。

ABO similarity-10 图搜图 T07 当前只有冻结 Qwen baseline，没有已经冻结的对应光学 MoE 图，因此不能伪装成实测值。如果后续明确采用与 LSP/SALICON 相同的“两层纯 vision feature block”，可先用约 `0.95 ms CUDA / 0.98 ms Wall` 作为工程估计，但正式表格应等真实图建立后复测。

## 如需填写“光学 block 最小串行路径”

仅在采用每次物理光场 `0.714 + 0.100 + 0.500 = 1.314 ms` 的合同下，可把上述融合时间加到物理光场时间。该值仍然不含最终任务头，不是完整端到端 latency。

| 任务类型 | 物理 pass | 物理时间 | 加严格 CUDA 电子处理 | 加严格 Wall 电子处理 |
|---|---:|---:|---:|---:|
| 四层视觉+语言任务（Caltech/OpenMoji/ABO T08） | 6 | 7.884 ms | 约 9.81 ms | 约 9.88 ms |
| 两层纯视觉任务（LSP/SALICON） | 3 | 3.942 ms | 约 4.89 ms | 约 4.93 ms |
| LGVQ 时间一致性，16 视频并行 | 6 | 7.884 ms | 9.776 ms / call | 9.845 ms / call |
| LGVQ 空间一致性，单视频 4 帧 | 6 | 7.884 ms | 9.776 ms / call | 9.845 ms / call |

论文主表若坚持只报“电子神经网络推理”，建议采用 CUDA 中位数；若报实际流水线 latency，则采用 Wall 中位数，并把 bridge、任务头和硬件驱动等另列，不能悄悄省略。

## 可复现文件

- 复测程序：`LightGenV2/scripts/profile_ccd_fusion_only.py`
- 完整逐分量证据：`evidence/ccd_fusion_only_a100.json`
- 汇总表：`evidence/ccd_fusion_only_a100.csv`
- 任务头逐分量证据：`evidence/task_heads_a100.json`
- 首轮全组件复测仅用于发现边界污染，不作为本报告结果。

服务器原始结果保存在：

`/DATA/DATA1/guest3/2026OpticsMoE_a100_measurements/20260908_fusion_boundary/strict/`
