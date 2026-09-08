# LightGenV2 八任务总清单

更新时间：2026-09-07

本表是老师查看整体进展时的唯一跨任务入口。每个任务固定一行；详细事实和证据仍放在
对应任务目录。`—` 表示尚无合格证据，不表示数值为零。旧工程结果统一标为“历史候选”，
迁移并复核前不能写成 LightGenV2 正式结果。

## 一页总表

| 优先级 | 任务 | 当前数据/协议 | 主指标 | 我们的仿真性能 | 我们的实测性能 | 我们的速度 | Baseline 性能 | Baseline 速度 | 功耗与能耗 | 仿真—实测一致性 | 当前状态与下一闭环 |
|---:|---|---|---|---|---|---|---|---|---|---|---|
| P1 | **T06 视频质量评价** | LGVQ；Spatial-4、Temporal-36、9视频×4帧、16视频×4帧为独立协议；test 558 | SRCC↑；同时报 KRCC/PLCC↑、RMSE/MAE↓ | **Temporal-36：SRCC 0.8454**；**9×4：SRCC 0.8082、PLCC 0.8131**；**16×4：SRCC 0.8044、PLCC 0.8180**；**单视频 Spatial-4：SRCC 0.6393、PLCC 0.6743** | — | RTX 5090D 计算图估算：16×4 一幅场 **28.744 ms/16视频**（1.796 ms/video 折算）；含 6 次物理光场、CCD 后串行电处理、bridge 与任务头；实验台端到端仍未测 | 冻结 Qwen3-VL-2B：Temporal-36 线性头 SRCC 0.7820；448px 五质量词 4/9/16 帧 SRCC **0.7693/0.7745/0.7787**；Spatial-4 SRCC 0.6440 | RTX 5090D：4帧方案二均值 **65.433 ms/video**；16视频顺序固定采用原始证据 **1046.928 ms** | 光学 80.388 W：6 次物理光场 **0.634 J/field**；全临界路径持续上电代理 **2.311 J/16视频**。Qwen 16视频原始实测 **120.680 J**，反算平均功率 **115.271 W**，575 W 上界 601.984 J | 16×4 无全局专家坍缩，但帧路由样本变化率仅 8.2%；实测仍需 PCC、SSIM、gain-aligned NMAE、强度比与饱和率 | **单视频 Spatial-4 已归档；16×4 已完成但未达到 0.81（实测 0.8044）。**分段计时每分量 1000 次、共 16,000 video workload；下一步测实验台端到端和 GPU/整机功率 |
| P2 | **T07 商品检索（图搜图）** | ABO；正式子集、gallery/query 划分尚未冻结 | Top-1/Top-5/Top-10、MRR、Recall@K↑ | — | — | — | — | — | — | — | **尚未运行。**先冻结可发表的数据协议和电子 baseline，再做同协议光电模型；不能引用仓库中旧 ABO 文件作为本任务结果 |
| P3 | **T08 商品检索（图搜文）** | ABO easy100；100 商品；train 4800/test 2400；100 个官方英文标题为固定候选库；每 5 epoch 看完整 test 选 EMA best | R@1/R@5/R@10、MRR、median rank↓ | **DC20 光 Router Top-2：R@1 0.7988、R@5 0.9479、R@10 0.9825、MRR 0.8617**；强均衡版 R@1 0.7983、MRR **0.8676** | — | — | Frozen `Qwen3-VL-Embedding-2B`：**R@1 0.7371、R@5 0.9337、R@10 0.9604、MRR 0.8230** | RTX 5090D、batch 1；首个 Vision block→2048D 归一化→100 标题完整排序：**26.052 ms/图** | active mean **152.50 W**；peak 156.84 W；idle-subtracted **2.209 J/图** | — | **光仿真与电子 baseline 均已完成。**光模型固定 224×224、64D，Qwen 主干冻结；强均衡候选只少命中 1 张 Top-1，但 R@5/R@10/MRR 更高，优先用于硬件部署；硬件速度、功耗和一致性尚未测 |
| P4 | T01 物品检索 | Caltech101 target-10：train 2625、gallery 30、query 200；单 seed；周期 test 选模 | Top-1/Top-3、MRR↑ | **DC20 光 Router Top-2：Top-1 90.0%、Top-3 96.5%、MRR 0.9344** | — | RTX 5090D 计算图估算 **10.061 ms/query**；4 特征 + 2 router | 同协议/激活相位预算匹配 D2NN：Top-1 89.5%；冻结 Qwen embedding：Top-1 **99.5%** | 冻结 Qwen：mean/median/P95 **26.407/25.731/28.985 ms/query** | 光学物理/墙上代理 **0.634/0.809 J/query**；Qwen 实测 **151.252 W、3.994 J/query** | 尚未做统一硬件实测 | **正式复跑完成。**分段计时每分量 1000 次，电残差 0.275 ms 均值可被 1.314 ms 光路覆盖；硬件端到端未测 |
| P5 | T02 关键点检测 | LSP；固定 test 1000；单 seed；周期 test 选模 | PCK@0.2、PCKh@0.5↑；MPE/NME↓ | **DC20 光 Router Top-2：PCK 0.5773、PCKh 0.7363、NME 0.3488** | — | RTX 5090D 计算图估算 **5.537 ms/image**；2 特征 + 1 router | 同协议、激活相位预算匹配 D2NN：PCK 0.6751；冻结完整 Qwen Vision + DeconvPoseHead：PCK **0.7217**、PCKh **0.8846** | 冻结 Qwen+反卷积头：mean/median/P95 **9.504/9.470/9.632 ms/image** | 光学物理/墙上代理 **0.317/0.445 J/image**；Qwen 实测 **140.128 W、1.332 J/image** | — | **正常大模型 baseline 已纠正。**1000 张 test 全量评估；Qwen 原生 Vision blocks 全部执行，只有 1,102,990 参数反卷积读出头可训练；旧 0.5114 来自欠容量双线性头，不再作为正式 baseline |
| P6 | T03 显著性分析 | SALICON train2014 10000；val2014 5000 作 public test；无 validation | CC/SIM/NSS/AUC-Judd↑；KLD/MAE↓ | **DC20 光 Router Top-2：CC 0.8291、SIM 0.8063、NSS 0.9283、AUC-Judd 0.7631** | — | RTX 5090D 计算图估算 **5.654 ms/image**；2 特征 + 1 router | 参数匹配 D2NN：CC **0.8346**；Frozen Qwen+头：CC **0.8811**、SIM 0.8327 | 冻结 Qwen+头：mean/median/P95 **10.176/9.687/12.544 ms/image** | 光学物理/墙上代理 **0.317/0.455 J/image**；Qwen 实测 **117.837 W、1.199 J/image** | — | **正式仿真完成。**分段计时每分量 1000 次；任务头 0.693 ms；硬件端到端未测 |
| P7 | T04 语义交互 | OpenMoji train5000/test1000；四操作均衡；无validation；v1含1条重复输入 | changed-cell、IoU、F1、scene exact↑ | **旧结构复评：changed0.9800、IoU0.9350、F1 0.9837、exact0.8950**；去光changed0.9850、exact0.8760 | — | 本次未测；旧5090D计算图估算10.862ms/sample不等于端到端 | **本次A100：D2NN changed0.9890、exact0.9640；冻结Qwen+训练头在线changed0.5390、exact0.0170** | 本次未测；历史27.166ms/sample需连同原边界引用 | 本次未测，不能沿用旧值当成本轮实测 | — | **三组测试已完成，但主方法/D2NN指令使用完整冻结语言TF缓存，不合规。**CCD还含clip/log；需修正后重训。相位有更新但alpha约5%；兼容投影592898参数可优先精简。全部证据见T04 reports/reproduction/RESULTS_20260908.md |
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

- RTX 5090D 跨任务光学 MoE 分段电处理与冻结 Qwen 汇总：
  `tasks/t06_video_quality_assessment/reports/5090d_moe_and_qwen_baselines_20260907/README.md`
- T06 Temporal-36：
  `tasks/t06_video_quality_assessment/reports/paper_results/temporal36_balanced/result.json`
- T06 九视频×四帧：
  `tasks/t06_video_quality_assessment/reports/paper_results/temporal_multivideo9x4_contentroute/result.json`
- T06 十六视频×四帧：
  `tasks/t06_video_quality_assessment/reports/paper_results/temporal_multivideo16x4/result.json`
- T06 448px 五质量词电子 baseline：
  `tasks/t06_video_quality_assessment/reports/paper_results/qwen3vl_quality_token_baseline_r448/result.json`
- T06 单视频 Spatial-4 正式归档：
  `tasks/t06_video_quality_assessment/reports/paper_results/spatial_single_video4_balanced/result.json`
- T06 baseline 时间：
  `../experiments/qwen3_vl_2b_lgvq_temporal_framecount_timing/PERFORMANCE_TIMING_REPORT.md`
- T01 Caltech101 正式单次对照：
  `tasks/t01_object_retrieval/reports/DC20_RESULTS.md`
- T02 LSP 正式单次对照：
  `tasks/t02_keypoint_detection/reports/dc20_comparison/RESULTS.md`
- T03 正式对照：`tasks/t03_saliency/reports/dc20_comparison/RESULTS.md`
- T04 正式对照：`tasks/t04_semantic_interaction/reports/dc20_comparison/RESULTS.md`
- T08 ABO 图搜文光 Router MoE：
  `tasks/t08_abo_image_text_retrieval/reports/optical_router_moe_20260907/README.md`
