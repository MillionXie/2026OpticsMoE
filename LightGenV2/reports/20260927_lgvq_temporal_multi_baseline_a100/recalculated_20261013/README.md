# LGVQ temporal quality: matched A100 inference costs and batch diagnostics

Updated 14 October 2026. All 19 frozen settings and both bounded numerical diagnoses are complete. The displayed curves now use only the uninterrupted `full_remeasurement02` measurements. Interrupted rounds and the initial failed DeepSeek input-fingerprint diagnostic remain preserved, excluded from final data. The original report location and three-workbook-sheet structure are retained.

## 两图含义与实验方法

图a展示每16视频推理能量与时间质量SRCC的关系，图b展示相同工作量的推理时间与SRCC的关系。横轴分别为Energy per 16 videos (J)和Inference time per 16 videos (ms)，使用对数坐标；纵轴为无量纲Temporal quality SRCC，范围0–1。B为每次GPU调用的视频数，N为每幅光场的视频数。每一个电子点都有其batch自己的558个TEST预测和同轮成本，不复制统一分数。LightGen-II仅为N=16物理SRCC参考和部件成本估算点，不是本轮重测光路。

LGVQ固定划分含2250 TRAIN、558 TEST，共2808个不同视频。每视频在10%、37%、63%、90%相对位置取4帧。Qwen/DeepSeek/CLIP/YOLO输入分别448/384/224/640像素正方形；后三者沿原短边65%中心裁剪。权重冻结，eval模式；各模型各B独立新进程。Qwen采用原历史head，其他三者采用2026-10-13原协议重训head，均保持PT/manifest SHA不变。重训head为50epoch/seed42/五级质量行/冻结backbone/CE，最高TEST选模为开发指标，无VAL。本轮无训练或选模。正式骨干精度Qwen与DeepSeek为BF16，CLIP与YOLO为FP16，质量头FP32；后三者保留FP16特征量化再FP32读出。

每批先CPU解码、裁剪和processor，再H2D，再推理，随后下一批。计时从首原生backbone block入口到GPU标量MOS就绪并完成同步；YOLO首block为卷积，其余为视觉Transformer。模型加载、CPU准备、H2D、之后CPU预测复制和审计开销不计最终成本。起点预钩子内无额外同步，因此为操作定义的同步墙钟边界，不是纯GPU kernel积分。无显式预热，第一调用保留。完整558视频统计SRCC，成本仅固定前200视频，并在200/558边界拆批，不丢尾样本。

T16=sum(first200 synchronized inference wall ms)×16/200；E16=(338.2 W主机代理+同轮tagged-active A100平均板卡W)×T16/1000。B=1/2/4/8时等价于mean batch ms×16/B；B=3/9/10含实际不足满批的尾批，不能用均值×16/B偷换。16仅为吞吐工作量归一化，不是仅测试16视频。

| B | Full TEST calls | Final TEST call videos | First200 cost calls | Final cost call videos |
| --- | --- | --- | --- | --- |
| 1 | 558 | 1 | 200 | 1 |
| 2 | 279 | 2 | 100 | 2 |
| 4 | 140 | 2 | 50 | 4 |
| 8 | 70 | 6 | 25 | 8 |
| 9 | 63 | 7 | 23 | 2 |
| 10 | 56 | 8 | 20 | 10 |
| 3 | 187 | 1 | 67 | 2 |

功率、GPU利用率与显存同轮按50ms轮询并标记阶段。驱动自己的平均窗口并非50ms瞬时值，util是kernel-busy而非SM occupancy或峰值FLOPS使用率。短推理窗口可包含前序CPU间隔影响，CLIP/YOLO主交错util较低不代表算子没有GPU运行。均值功率×边界时间是沿用的能量估算，不是校准短调用积分。A100设250W；原始短功率读数中存在超过250W的峰值（DeepSeek约321–342W），保留不裁掉，不把设置值描述为瞬时硬限。

## 图a：能量与时间质量性能

LightGen-II物理SRCC为0.7977，每16视频能量估算1.1777J。新同轮Qwen B1为458.3508J，B8为228.3324J，分别对应389.19和193.88倍任务能效比（相同工作量的能量反比），能量降低99.7431%和99.4842%。多数基线增大batch后每视频成本下降，但这不是必须单调的理论：CLIP B9高于B8，包含尾批及运行噪声，不能人为平滑。性能曲线近水平源于SRCC变化很小，不是固定分数复制。

## 图b：时间与时间质量性能

LightGen-II每16视频时间估算10.2456ms。新同轮Qwen B1与B8为1180.8067与512.4304ms，时间比分别115.25与50.01。增大B通常使单调用更长，但完成固定工作量的调用数下降并摊销开销，吞吐改善随后可能饱和或因显存/计算限制退化；不要求所有成本点单调，也不存在同权重准确率随batch上升的理论趋势。

## 全部绘图数据及主交错遥测

数据来源同一轮14 October 2026。以下20点含19电子设置和1光学参考。成本保留6位用于核对，不表示6位有效精度。对应未提前舍入数据为`plot_data.csv`与`timing_energy_recalculated.csv`；后者还保留mean/peak功率、利用率、显存、调用数及SHA。

| Model | B / N | SRCC (558 TEST) | Time/16 (ms) | Energy/16 (J) | Active util mean (%) | Peak VRAM (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| LightGen-II | N=16 | 0.7977138739 | 10.245570 | 1.177690 | not measured | not measured |
| Qwen3-VL-2B | 1 | 0.7663428037 | 1180.806745 | 458.350757 | 14.972 | 4705 |
| Qwen3-VL-2B | 2 | 0.7664148425 | 765.604070 | 296.237303 | 11.183 | 4883 |
| Qwen3-VL-2B | 4 | 0.7694958345 | 553.714911 | 233.495159 | 3.121 | 5107 |
| Qwen3-VL-2B | 8 | 0.7694089803 | 512.430360 | 228.332385 | 13.556 | 5603 |
| CLIP ViT-B/32 | 1 | 0.7252558080 | 166.339300 | 63.111235 | 1.898 | 903 |
| CLIP ViT-B/32 | 2 | 0.7252452404 | 94.548801 | 35.723075 | 2.474 | 923 |
| CLIP ViT-B/32 | 4 | 0.7252388170 | 52.953970 | 20.029098 | 0.600 | 931 |
| CLIP ViT-B/32 | 8 | 0.7252388170 | 34.412313 | 13.029472 | 3.273 | 975 |
| CLIP ViT-B/32 | 9 | 0.7252247615 | 36.311530 | 13.795622 | 2.000 | 995 |
| CLIP ViT-B/32 | 10 | 0.7252388170 | 31.707193 | 12.048240 | 4.444 | 999 |
| DeepSeek-VL2-Tiny | 1 | 0.5684669010 | 2917.680463 | 1332.941805 | 20.556 | 7771 |
| DeepSeek-VL2-Tiny | 2 | 0.5703507558 | 2039.096788 | 1008.243099 | 18.043 | 8479 |
| DeepSeek-VL2-Tiny | 4 | 0.5680463058 | 1714.183199 | 883.138613 | 38.224 | 9609 |
| DeepSeek-VL2-Tiny | 8 | 0.5698244506 | 1543.665272 | 819.439993 | 62.197 | 12065 |
| YOLO11s | 1 | 0.7015970082 | 336.490797 | 127.530261 | 4.358 | 841 |
| YOLO11s | 2 | 0.7015940037 | 189.279941 | 71.566832 | 3.409 | 1037 |
| YOLO11s | 3 | 0.7015940037 | 154.108012 | 58.414706 | 3.188 | 1259 |
| YOLO11s | 4 | 0.7015916554 | 132.892382 | 50.314170 | 2.032 | 1467 |
| YOLO11s | 8 | 0.7015904121 | 121.058940 | 45.901801 | 3.935 | 2327 |

## 连续负载：配套时延、能量与GPU利用率

另在同一进程、同一PT/B下，用前B个不同TEST视频已驻GPU的输入重复推理约15秒。此处没有每次CPU准备，成本和util来自连续负载自己的记录，严禁将本表高util配给上一表交错成本。时间折算mean boundary ms×16/B，B=3/9/10为吞吐等效值，不宣称整数次调用刚好处理16条。连续负载未独立计算新的全TEST性能，性能仍来自同PT/B的558视频独立前向。CLIP B9平均80.7617%触发补B10；B10平均89.1913%未达98%，不扫描更大B。YOLO B3平均85.6284%，B4为96.5436%，B8为97.0976%；不能把峰值98%当作平均98%。

| Model | B | Time/16 (ms) | Energy/16 (J) | Mean util (%) | Peak util (%) | Mean board power (W) | Peak VRAM (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B | 1 | 778.934426 | 407.094159 | 63.117 | 71 | 184.430 | 4705 |
| Qwen3-VL-2B | 2 | 493.315589 | 284.183387 | 83.431 | 88 | 237.868 | 4883 |
| Qwen3-VL-2B | 4 | 455.337138 | 263.522301 | 85.367 | 90 | 240.541 | 5107 |
| Qwen3-VL-2B | 8 | 422.711758 | 243.963920 | 89.220 | 95 | 238.940 | 5603 |
| CLIP ViT-B/32 | 1 | 89.181167 | 36.400936 | 39.010 | 60 | 69.968 | 903 |
| CLIP ViT-B/32 | 2 | 45.296298 | 20.212696 | 40.805 | 63 | 108.033 | 923 |
| CLIP ViT-B/32 | 4 | 22.608331 | 10.856470 | 50.671 | 64 | 141.998 | 935 |
| CLIP ViT-B/32 | 8 | 11.922274 | 6.511954 | 71.295 | 89 | 208.001 | 975 |
| CLIP ViT-B/32 | 9 | 10.267646 | 5.880074 | 80.762 | 98 | 234.480 | 999 |
| CLIP ViT-B/32 | 10 | 8.932255 | 5.185225 | 89.191 | 95 | 242.306 | 999 |
| DeepSeek-VL2-Tiny | 1 | 2471.141476 | 1302.795259 | 71.463 | 87 | 189.004 | 7771 |
| DeepSeek-VL2-Tiny | 2 | 2064.253894 | 1120.997979 | 74.415 | 83 | 204.852 | 8481 |
| DeepSeek-VL2-Tiny | 4 | 1579.469002 | 909.900767 | 89.662 | 98 | 237.880 | 9651 |
| DeepSeek-VL2-Tiny | 8 | 1438.482441 | 833.185891 | 95.258 | 100 | 241.012 | 12065 |
| YOLO11s | 1 | 206.996660 | 90.978331 | 39.503 | 69 | 101.316 | 841 |
| YOLO11s | 2 | 88.403786 | 45.909937 | 67.772 | 87 | 181.121 | 1037 |
| YOLO11s | 3 | 62.638440 | 35.967365 | 85.628 | 98 | 236.006 | 1259 |
| YOLO11s | 4 | 53.780790 | 31.315951 | 96.544 | 99 | 244.089 | 1469 |
| YOLO11s | 8 | 49.939259 | 29.122482 | 97.098 | 99 | 244.958 | 2329 |

## 精度变化的诊断证据

同权重精确数学推理不应因batch而改变输出，也无性能随B单调提升或下降的预期。第一步核所有558个样本ID、MOS及每模型抽帧一致；同模型PT与manifest SHA一致。Qwen和DeepSeek另对前8个固定TEST做B1/2/4输入tensor SHA及重复前向。逐视频有效文本、图像、抽帧相同，DeepSeek还核图像crop及图像token mask；原精度重复结果相同，原精度诊断与正式前向最大差0。

| Model | Probe | B2–B1 max absolute MOS difference | B4–B1 max absolute MOS difference |
| --- | --- | --- | --- |
| qwen | native | 2.350662231 | 3.395130157 |
| qwen | native_repeat | 2.350662231 | 3.395130157 |
| qwen | native_math_sdpa | 1.700469971 | 1.902404785 |
| qwen | cast_deployed_weights_fp32_math_sdpa | 0.000484467 | 0.000289917 |
| deepseek | native | 1.472774506 | 1.418304443 |
| deepseek | native_repeat | 1.472774506 | 1.418304443 |
| deepseek | native_math_sdpa | 1.472774506 | 1.418304443 |
| deepseek | cast_deployed_weights_fp32_math_sdpa | 0.001331329 | 0.001548767 |

FP32探针同时将部署BF16权重转FP32并使用math SDPA，不恢复权重被量化丢弃的位，也不替代正式结果。DeepSeek保留部署FP16特征取值；math SDPA不控制未调用torch SDPA的算子。Qwen math-only未消除差异，DeepSeek math-only无改变；FP32探针大幅减小差异，支持低精度计算路径是主要贡献，而不是输入改变或随机重训。范围仅8视频，未逐层分解每一算子，不能声称完整唯一因果证明。SRCC基于MOS排序，数值差异可改变秩；近水平曲线不意味着每条MOS完全相同。正式19组未改精度、权重或选模。

## 历史总表锚点与LightGen-II口径

历史Qwen B1仍保留SRCC0.7663428036522781、1188.3861464262009ms/16、462.1134010767563J/16，不写入新同轮曲线，也没有GPUutil记录。新B1为1180.8067452907562ms/16、458.3507566057941J/16，性能相同但成本为独立测量；不能新util与旧成本拼接冒称配套。九任务总表未被本轮修改。

LightGen-II沿此前表格参考：N=16同光场，六周期6×(.714+.300+.0307)=6.2682ms；电子部件3.9773695913702247ms是整幅16槽场计时，不是测一条视频再乘16。router .8302489486336708、CCD后处理2.0079718434810636、bridge .2413414404541254、head .8978073588013649ms。总10.245569591370225ms；E16=.0062682×(80.358+41.388)+.0039773695913702247×(41.388+62.842)=1.1776895097085185J。每视频摊销.0736055943567824J，13.5859238518309videos/J；.6403480994606391ms/video为摊销服务时间而非单视频调用时延。光周期与电子FP32 CUDA-event部件估算不冒称本轮实测精确正式PT光路端到端，不能与baseline同步墙钟说成相同仪器/精度。

## English Methods

We evaluated temporal video-quality prediction on the fixed LGVQ split of 2,250 training and 558 test videos. Each video contributed four frames at relative positions of 10%, 37%, 63% and 90%. All 19 frozen model–batch configurations were evaluated independently on the same 558 test videos. Inference cost was measured on the first 200 videos and normalized to a workload of 16 videos. Partial batches were retained at the 200- and 558-video boundaries. Each configuration ran in a fresh process on the same NVIDIA A100-PCIE-40GB. For each batch, CPU preprocessing and host-to-device transfer preceded inference; these stages were recorded for audit but excluded from the reported costs. Synchronized wall-clock timing began at the first native backbone block and ended after the scalar quality score was available on the GPU. No explicit warm-up was performed, and the first measured call was retained. Energy was estimated as inference duration multiplied by the sum of the 338.2-W host-power proxy and the same-run inference-active mean board power.

GPU utilization, board power and memory use were sampled together at a nominal 50-ms polling interval. GPU utilization denotes the driver-window fraction of time during which kernels were active, not streaming-multiprocessor occupancy. A separate approximately 15-s continuous-load experiment repeatedly processed the first B distinct test inputs already resident on the GPU. Continuous-load latency, energy and utilization are reported together and are not assigned to the interleaved-inference costs. Qwen and DeepSeek retained BF16 backbone execution, CLIP and YOLO retained FP16 execution, and quality heads used FP32. The LightGen-II point retains the physical SRCC reference and the previously approved optical-cycle plus electronic-component cost estimate; it is not a new end-to-end optical measurement.

## English figure description

Temporal quality versus inference cost on LGVQ. (a) Energy and (b) inference time normalized to 16 videos. Electronic points use the complete matched A100 remeasurement of 14 October 2026; B indicates videos per GPU call. SRCC is calculated separately from all 558 test predictions at each batch size. The LightGen-II reference represents 16 videos per optical field (N=16), with physical SRCC 0.7977 and component-estimated costs of 1.1777 J and 10.2456 ms. Larger batches generally reduce amortized electronic cost, but this trend need not be monotonic. The nearly horizontal performance curves do not imply batch-invariant predictions: bounded same-input probes support low-precision computation as a major contributor to the observed batch-dependent differences. Historical measurements, preprocessing costs and continuous-load telemetry are not mixed into these points.

## Reproducibility and evidence

Formal run: `LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_temporal_telemetry_a100_20261014/full_remeasurement02`. Measurement entry SHA28fc58f1; telemetry28ad0666. Manifest SHA607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc. Original Qwen PT e102adc6; reproduced CLIP28729f41/YOLO47606708/DeepSeek3748b08f. Exact per-file SHA and sources are in `recalculation.json` and the canonical run backup receipts. Old refreshed report files are recoverably archived in the same run's `report_before_refresh` with a manifest; all original measurements and failed attempts remain unchanged. The historical side-effecting `build_analysis.py` was not executed. Report refresh is CPU-only and launches no GPU processes.
