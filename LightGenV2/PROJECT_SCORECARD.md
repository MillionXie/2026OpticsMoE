# LightGenV2 八任务总清单

更新时间：2026-09-06

本表是老师查看整体进展时的唯一跨任务入口。每个任务固定一行；详细事实和证据仍放在
对应任务目录。`—` 表示尚无合格证据，不表示数值为零。旧工程结果统一标为“历史候选”，
迁移并复核前不能写成 LightGenV2 正式结果。

## 一页总表

| 优先级 | 任务 | 当前数据/协议 | 主指标 | 我们的仿真性能 | 我们的实测性能 | 我们的速度 | Baseline 性能 | Baseline 速度 | 功耗与能耗 | 仿真—实测一致性 | 当前状态与下一闭环 |
|---:|---|---|---|---|---|---|---|---|---|---|---|
| P1 | **T06 视频质量评价** | LGVQ；Spatial-4、Temporal-36、9视频×4帧、16视频×4帧为独立协议；test 558 | SRCC↑；同时报 KRCC/PLCC↑、RMSE/MAE↓ | **Temporal-36：SRCC 0.8454**；**9×4：SRCC 0.8082、PLCC 0.8131**；**16×4：SRCC 0.8044、PLCC 0.8180**；Spatial-4 候选 SRCC 0.6371 | — | 9×4/16×4 均为六次整场传播，一幅场分别输出 9/16 个 MOS；实验台端到端延迟/吞吐 **未测** | 冻结 Qwen3-VL-2B：Temporal-36 线性头 SRCC 0.7820；448px 五质量词 4/9/16 帧 SRCC **0.7693/0.7745/0.7787**；Spatial-4 SRCC 0.6440 | RTX 5090D、batch 1：原 MP4 到标量 Temporal-36 均值 **1133.494 ms/视频**；448px 五质量词从 Vision block 0 到分数为 **65.433/65.133/86.078 ms**（4/9/16 帧，无显式 warmup、首条计入）；Spatial-4 端到端 **134.058 ms/视频** | — | 16×4 无全局专家坍缩，但帧路由样本变化率仅 8.2%；实测仍需 PCC、SSIM、gain-aligned NMAE、强度比与饱和率 | **16×4 已完成但未达到 0.81（实测 0.8044）；448px 五质量词 baseline 已形成可追溯性能/速度证据。**下一步优先做硬件串扰、端到端速度/功耗和逐级 CCD 一致性 |
| P2 | **T07 商品检索（图搜图）** | ABO；正式子集、gallery/query 划分尚未冻结 | Top-1/Top-5/Top-10、MRR、Recall@K↑ | — | — | — | — | — | — | — | **尚未运行。**先冻结可发表的数据协议和电子 baseline，再做同协议光电模型；不能引用仓库中旧 ABO 文件作为本任务结果 |
| P3 | **T08 商品检索（图搜文）** | ABO；文本字段、候选库和负样本协议尚未冻结 | R@1/R@5/R@10、MRR、median rank↓ | — | — | — | — | — | — | — | **尚未运行。**先明确 image→text 检索单位、prompt 和候选库，再跑电子 baseline 与光电版本 |
| P4 | T01 物品检索 | Caltech101 target-10：train 2625、gallery 30、query 200；单 seed；周期 test 选模 | Top-1/Top-3、MRR↑ | **DC20 光 Router Top-2：Top-1 90.0%、Top-3 96.5%、MRR 0.9344** | — | — | 同协议/激活相位预算匹配 D2NN：Top-1 89.5%；冻结 Qwen embedding：99.5% | — | — | 尚未做统一硬件实测 | **正式复跑完成。**含 20%–30% 未调制分量、偏置 CCD 噪声、±16 px、k 空间/phase-DC；hard-load 0.50 消除未使用专家，但 Language 仍集中；只保留 best+last |
| P5 | T02 关键点检测 | LSP；固定 test 1000；单 seed；周期 test 选模 | PCK@0.2、PCKh@0.5↑；MPE/NME↓ | **DC20 光 Router Top-2：PCK 0.5773、PCKh 0.7363、NME 0.3488** | — | — | 同协议、激活相位预算匹配 D2NN：PCK 0.6751、PCKh 0.8054、NME 0.2736 | — | — | — | **正式复跑完成。**主方法低于 D2NN 0.0978 PCK；当前不继续为低优先级任务堆 run，保留该负结果和最佳相位证据 |
| P6 | T03 显著性分析 | SALICON train2014 10000；val2014 5000 作 public test；无 validation | CC/SIM/NSS/AUC-Judd↑；KLD/MAE↓ | **DC20 光 Router Top-2：CC 0.8291、SIM 0.8063、NSS 0.9283、AUC-Judd 0.7631** | — | — | 参数匹配 D2NN：CC **0.8346**；Frozen Qwen 待5090D | Frozen Qwen 待5090D | Frozen Qwen 待5090D | — | **正式仿真完成。**四专家选择均衡（23.54%/26.80%/23.38%/26.28%），但 D2NN 的 CC 高 0.0055；只保留 best+last 和正式报告 |
| P7 | T04 语义交互 | OpenMoji train 5000/test 1000；四操作各自均衡；无 validation | changed-cell accuracy、edit-grid IoU、object F1、scene exact match↑ | **DC20 光 Router Top-2：changed 0.9795、IoU 0.9340、F1 0.9833、exact 0.8930** | — | — | 参数匹配 D2NN：changed **0.9895**、IoU 0.9813；Frozen Qwen 待5090D | Frozen Qwen 待5090D | Frozen Qwen 待5090D | — | **正式仿真完成。**语言/视觉 Router 均使用 3/4 专家，第4专家未进入 Top-2；该限制已如实记录，D2NN 主指标高 0.0100 |
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
- T06 十六视频×四帧：
  `tasks/t06_video_quality_assessment/reports/paper_results/temporal_multivideo16x4/result.json`
- T06 448px 五质量词电子 baseline：
  `tasks/t06_video_quality_assessment/reports/paper_results/qwen3vl_quality_token_baseline_r448/result.json`
- T06 Spatial 历史候选：
  `../experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/SPATIAL_OPTIMIZATION_RESULT.md`
- T06 baseline 时间：
  `../experiments/qwen3_vl_2b_lgvq_temporal_framecount_timing/PERFORMANCE_TIMING_REPORT.md`
- T01 Caltech101 正式单次对照：
  `tasks/t01_object_retrieval/reports/DC20_RESULTS.md`
- T02 LSP 正式单次对照：
  `tasks/t02_keypoint_detection/reports/dc20_comparison/RESULTS.md`
- T03 正式对照：`tasks/t03_saliency/reports/dc20_comparison/RESULTS.md`
- T04 正式对照：`tasks/t04_semantic_interaction/reports/dc20_comparison/RESULTS.md`
