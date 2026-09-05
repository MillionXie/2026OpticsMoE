# LightGenV2 八任务总清单

更新时间：2026-09-05

本表是老师查看整体进展时的唯一跨任务入口。每个任务固定一行；详细事实和证据仍放在
对应任务目录。`—` 表示尚无合格证据，不表示数值为零。旧工程结果统一标为“历史候选”，
迁移并复核前不能写成 LightGenV2 正式结果。

## 一页总表

| 优先级 | 任务 | 当前数据/协议 | 主指标 | 我们的仿真性能 | 我们的实测性能 | 我们的速度 | Baseline 性能 | Baseline 速度 | 功耗与能耗 | 仿真—实测一致性 | 当前状态与下一闭环 |
|---:|---|---|---|---|---|---|---|---|---|---|---|
| P1 | **T06 视频质量评价** | LGVQ；Spatial-4、Temporal-36、九视频×四帧是三个独立协议；test 558 | SRCC↑；同时报 KRCC/PLCC↑、RMSE/MAE↓ | **Temporal-36 正式：SRCC 0.8454，KRCC 0.6394，PLCC 0.8650，RMSE 7.183，MAE 5.451**；**九视频×四帧全场并行：SRCC 0.8082，KRCC 0.5999，PLCC 0.8131，RMSE 8.226，MAE 6.109**；Spatial-4 候选：SRCC 0.6371，PLCC 0.6694 | — | 九视频×四帧仍为六次整场传播、一次产生 9 个 MOS；实验台端到端延迟/吞吐 **未测** | 匹配的冻结 Qwen3-VL-2B + 线性头：Temporal-36 SRCC 0.7820；Spatial-4 SRCC 0.6440 | RTX 5090D、batch 1、原 MP4 到标量：Temporal-36 均值 **1133.494 ms/视频**；Spatial-4 **134.058 ms/视频** | — | 九槽位仿真循环审计 SRCC 均值 0.8019；实测仍需 PCC、SSIM、gain-aligned NMAE、强度比与饱和率 | **Temporal-36 与九视频×四帧仿真完成；Spatial 候选完成；硬件闭环为最高优先级。**先测九视频串扰矩阵、端到端速度/功耗和逐级 CCD 一致性，再做实测微调 |
| P2 | **T07 商品检索（图搜图）** | ABO；正式子集、gallery/query 划分尚未冻结 | Top-1/Top-5/Top-10、MRR、Recall@K↑ | — | — | — | — | — | — | — | **尚未运行。**先冻结可发表的数据协议和电子 baseline，再做同协议光电模型；不能引用仓库中旧 ABO 文件作为本任务结果 |
| P3 | **T08 商品检索（图搜文）** | ABO；文本字段、候选库和负样本协议尚未冻结 | R@1/R@5/R@10、MRR、median rank↓ | — | — | — | — | — | — | — | **尚未运行。**先明确 image→text 检索单位、prompt 和候选库，再跑电子 baseline 与光电版本 |
| P4 | T01 物品检索 | Caltech101-10 历史协议：train 2625、gallery 30、query 200 | Top-1/Top-3、MRR↑ | 历史可部署鲁棒候选：Top-1 **85.0%**、Top-3 94.5%、MRR 0.9063；尚未迁移为 LightGenV2 正式 run | — | — | 历史纯电子参考：Top-1 **87.0%**、Top-3 96.0%、MRR 0.9159 | — | — | 局部 CCD/仿真对照曾做探索，但没有冻结成任务级正式汇总 | 迁移 accuracy-first 鲁棒 checkpoint、配置和证据；复核 phase 可训练、数据 split 与选模口径后，再补硬件四阶段结果 |
| P5 | T02 关键点检测 | LSP；固定 test 1000 | PCK@0.2、PCKh@0.5↑；MPE/NME↓ | 历史单次候选：PCK **0.7130**、PCKh **0.8375**、MPE 15.951 px；尚未迁移 | — | — | 历史 teacher 证据与当前候选不完全匹配，暂不作公平 baseline | — | — | — | 先迁移并锁定同参数/同输入 baseline；再决定是否继续，因为当前论文优先级低于 T06/T07/T08 |
| P6 | T03 显著性分析 | SALICON 官方公开 validation 5000 | CC/SIM/NSS/AUC-Judd↑；KLD/MAE↓ | 历史单次候选：CC **0.86274**、AUC-Judd 0.77000、SIM 0.82411、NSS 0.97200；尚未迁移 | — | — | 历史 teacher 约 CC 0.87972、AUC-Judd 0.77263；需复核是否同协议 | — | — | — | 迁移冻结证据；补同协议电子 baseline、速度和硬件可行性评估 |
| P7 | T04 语义交互 | OpenMoji，add/replace/move/remove；test 1000 | changed-cell accuracy、edit-grid IoU、object F1、scene exact match↑ | 历史 pilot：changed-cell **0.8765**、IoU 0.7629、object F1 0.9016、scene exact 0.6640；尚未迁移 | — | — | — | — | — | — | 先补严格 baseline；数据版权/可发表性确认后再决定是否投入硬件实验 |
| P8 | T05 视频分类 | 数据集与论文问题尚未确定 | Top-1/Top-5 或 mAP（待协议确定） | — | — | — | — | — | — | — | **未开始。**在数据集确定前不建空模型、不产生 runs |

## “做完一行”的最低标准

一个任务只有同时满足以下项目，才可在“当前状态”写为完成：

1. 冻结数据版本、train/test 划分、样本数、输入尺寸/帧数、prompt 和 checkpoint 选择规则。
2. 报告任务主指标及必要次指标；至少三个 seed 时写 `mean ± std`，单次运行明确标注。
3. Baseline 与我们的方法使用相同数据、预处理、评价脚本和测试集合。
4. 同时给出计算核心时间与真实端到端时间；端到端至少报告 mean、median、P95、batch、
   设备和每秒样本数。不得把理想传播时间写成实验台实测速度。
5. 实测性能至少区分“直接部署”和“硬件微调后”；写清实际样本数和重复次数。
6. 仿真—实测至少报告 PCC、SSIM、gain-aligned NMAE、均值强度比和饱和像素率；两端必须
   使用同一 ROI、方向合同和网络输入归一化，显示用 `log1p` 不得混入指标路径。
7. 功耗同时报告 idle、active average、peak 和增量能耗：
   `energy/sample = integral(P_active - P_idle) dt / N`。光电系统需说明是否包含激光器、
   两块 SLM、CCD、控制机/GPU；baseline 需说明 GPU 型号和测量边界。
8. 保存 config、命令、Git commit、checkpoint SHA256、原始逐样本预测/时序/功率记录和汇总
   文件；表中每个数字必须能回到这些证据。

## 统一速度与功耗记录字段

为便于以后组合成论文总图，每个任务最终都按相同字段记录：

| 类别 | 必填字段 |
|---|---|
| 任务负载 | 输入数、帧数、并行 lane/视频数、batch、光传播次数、SLM 写入次数、CCD 采集次数 |
| 我们的方法速度 | 预处理、振幅 SLM 写入与稳定、相位 SLM 写入与稳定、曝光、CCD 读出、几何矫正/归一化、电子尾部、端到端 mean/median/P95、throughput |
| Baseline 速度 | 数据读取/解码、预处理、模型前向、后处理、端到端 mean/median/P95、throughput、GPU 与精度模式 |
| 我们的方法功耗 | idle W、active mean W、peak W、J/sample；部件边界和功率计采样率 |
| Baseline 功耗 | GPU/整机 idle W、active mean W、peak W、J/sample；`nvidia-smi` 只能作为 GPU 侧证据，不能冒充整机功耗 |
| 可靠性 | warm-up 次数、重复次数、均值/标准差或置信区间、失败/超时/饱和比例 |

## 推进顺序

1. **T06 硬件闭环**：Temporal-36 优先，随后 Spatial-4。先完成速度/功耗/一致性，再报告
   直接部署和微调后性能；这是目前最接近完整论文表格的一行。
2. **T07 ABO 图搜图**：冻结数据与检索协议，先跑可复现 baseline，再训练光电版本。
3. **T08 ABO 图搜文**：协议必须独立于 T07，不能把分类准确率或图搜图结果代替跨模态检索。
4. T01–T04 只迁移可追溯的正式候选并补齐公平 baseline；T05 等数据集确定后再启动。

## 当前证据入口

- T06 Temporal-36：
  `tasks/t06_video_quality_assessment/reports/paper_results/temporal36_balanced/result.json`
- T06 九视频×四帧：
  `tasks/t06_video_quality_assessment/reports/paper_results/temporal_multivideo9x4_contentroute/result.json`
- T06 Spatial 历史候选：
  `../experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/SPATIAL_OPTIMIZATION_RESULT.md`
- T06 baseline 时间：
  `../experiments/qwen3_vl_2b_lgvq_temporal_framecount_timing/PERFORMANCE_TIMING_REPORT.md`
- T01 历史鲁棒候选：
  `../experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff/RESULTS.md`
- T02/T03 历史冻结证据：`../document/18_vision2_hybrid_dense_tasks/README.md`
- T04 历史 pilot：
  `../experiments/qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing/README.md`
