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
| P6 | T03 显著性分析 | SALICON train2014 10000；val2014 5000 作 public test；无 validation；单seed选模 | CC/SIM/NSS/AUC-Judd↑；KLD/MAE↓ | **DC20光Router Top2、alpha≥0.4：CC 0.85134、SIM 0.81768、NSS 0.95899**；无下限CC 0.85213 | — | 本轮未测；旧模型5090D计算图估算5.654 ms/image，不冒充新版端到端 | 新同规格头Frozen Qwen：**CC 0.88968**；旧头复评0.88103、重新训练0.87899；历史D2NN 0.8346 | 新Qwen头未测；旧头历史mean 10.176 ms/image | 本轮未测；旧光学物理/墙上代理0.317/0.455 J；旧Qwen 117.837 W、1.199 J，不沿用作新头结果 | — | **三阶段100epoch完成；四组训练方法续训已启动，结构不变。**alpha为0.4359/0.4422，专家份额22.91–27.51%；学习率覆盖勘误与run证据见[T03复现说明](tasks/t03_saliency/reports/reproduction/TRAINING_REFINEMENT.md) |
| P7 | T04 语义交互 | OpenMoji train5000/test1000；四操作均衡；无validation；旧v1有重复，新v2按源网格+指令去重 | changed-cell、IoU、F1、scene exact↑ | **旧结构复评：changed0.9800、IoU0.9350、F1 0.9837、exact0.8950**；去光changed0.9850、exact0.8760；新结构训练中 | — | 本次未测；旧5090D计算图估算10.862ms/sample不等于端到端 | **旧v1 A100：D2NN changed0.9890、exact0.9640；冻结Qwen+训练头在线changed0.5390、exact0.0170** | 本次未测；历史27.166ms/sample需连同原边界引用 | 本次未测，不能沿用旧值当成本轮实测 | — | **新embedding-only双模态光Router / alpha>0.4已启动100轮训练，另含lean与D2NN。**无语言TF缓存，删无效输出映射及mean/max拼接；CCD仅标量归一化+线性分箱。第3轮alpha约0.594–0.598、相位确有更新；最终性能待测。源码5298f296，说明见T04 reports/reproduction/EMBEDDING_ALPHA40.md；旧指标不可当新架构成绩 |
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

> 2026-09-08补充：上面总表是历史汇总，不代表所有服务器worktree的最新事实。
> T07独立A100分支已有冻结Qwen结果；T08本轮查到一对跨split重复图像；T06旧正式入口有后端SHA冲突。
> 本轮审查详情见下方“2026-09-08 初步审查”，历史数值未被静默重写。

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

## 2026-09-08 初步审查

本轮是源码、服务器数据及历史产物的double check，**没有重新训练或全量推理任何模型，也没有改模型/原始数据**。
本地起始HEAD为 `458b4ab2c6ba16b00a0742c140f677f28dbaa3f9`，服务器首轮检查主目录为
`c313ff807617a75f410246e08d22d5f03b96443b`，两者均有未提交修改；正在开发的T04/T06代码没有被覆盖。
审查读取81份服务器run/报告元数据，重新计算6个主方法best文件SHA；另核验T08全部7200图像的字节SHA、
T04重复输入和LGVQ清单，从历史预测独立重算T07 Hit/Precision/Recall及T08检索指标。
每任务文稿附 `EVIDENCE_20260908.json`，区分本轮检查、原报告数值和未验证事项。
审查期间其他工作仍在提交：证据写入时本地HEAD为 `79bf0f2695949fc5d0353698cfad417c14282f7b`，
最新T04记录已启动v2三组训练；本轮未把其早期进展认定为最终精度复现，服务器CPU测试仍属于较早源码快照。

| 任务 | 一任务一份架构/操作/参数审查稿 | 主要结论 |
|---|---|---|
| T01 | [CHECK_20260908](tasks/t01_object_retrieval/reports/reproduction/CHECK_20260908.md) | 实为10个类别原型排序；MoE仅多命中1/200，Top-3/MRR不占优；warmstart链待补 |
| T02 | [CHECK_20260908](tasks/t02_keypoint_detection/reports/reproduction/CHECK_20260908.md) | MoE低于D2NN；Qwen为完整Vision+监督Deconv头；计时尚未统一到正式二维坐标解码 |
| T03 | [CHECK_20260908](tasks/t03_saliency/reports/reproduction/CHECK_20260908.md) | 旧log版、mean_only续训、高alpha新实验应分开；Qwen头/预算不同；public test参与选择 |
| T04 | [CHECK_20260908](tasks/t04_semantic_interaction/reports/reproduction/CHECK_20260908.md) | v1完整语言TF缓存、真实source_grid、1条跨split重复；v2开发版不能沿用98% |
| T05 | [CHECK_20260908](tasks/t05_video_classification/reports/reproduction/CHECK_20260908.md) | 未发现正式实现与run；仅提供待冻结的实验合同 |
| T06 | [CHECK_20260908](tasks/t06_video_quality_assessment/reports/reproduction/CHECK_20260908.md) | 旧入口完整性校验失败；另含14通道电子质量输入；冷/热计时协议冲突；多视频padding仍有光场输入 |
| T07 | [CHECK_20260908](tasks/t07_abo_image_retrieval/reports/reproduction/CHECK_20260908.md) | 独立A100分支已有480-query冻结baseline，Hit@1=0.952083；归档SHA有65位转录错误 |
| T08 | [CHECK_20260908](tasks/t08_abo_image_text_retrieval/reports/reproduction/CHECK_20260908.md) | 全100商品/spin跨split共享，且1对图像字节重复；裁剪/维度/监督不同；多次5090D记录需按run区分 |

### 必须先处理的共同问题

1. **评估设计**：目前多个任务的test既用于checkpoint选择，也用于架构/超参取舍。诚实披露不等于消除了选择偏差。
   历史值保留 `test used for selection`；要报告独立泛化，应另冻结validation与未参与开发的test，或设计新的外部测试。
   已反复看过的test不能仅改名就变成盲测。多seed应报告所有预先约定运行，不只挑最高seed。
2. **比较对象**：Qwen有零训练embedding、完整视觉主干+监督头、完整多模态主干+监督头及五质量词读出等多种形态。
   必须逐任务命名；“冻结主干”不等于“整个系统没有训练”。D2NN只匹配激活专家相位，不匹配总参数/global/router/传播次数。
3. **光学主张**：光Router包含电子Top-K和读数处理；许多旧CCD链含clip/log1p，且有电子mixer、MLP、读出和手工特征。
   可以是合理光电系统，但不能描述为全光、CCD后纯线性或全部语义由光提取。alpha是混合系数，需配合去光、固定/打乱路由、
   独立电子模型等消融；相位有梯度并不能证明相位对任务必要。
4. **鲁棒性**：T01/T08的DC20训练并不意味着其报告test在20%实测噪声下完成；T06有名义eval未调制强度20%的另一个合同。
   分别列ideal simulation、noisy simulation、hardware direct、hardware finetuned及重复次数。
5. **时间/能耗**：根协议warmup50与T06 dataset-once warmup0不一致。光学临界路径组合估算、每场吞吐折算和真实端到端要分列。
   新视频/未知指令的前端和缓存生成成本需按输入边界纳入；Qwen批处理吞吐也应测试，不能只把batch1时间乘N代表其最佳吞吐。
6. **功率代码与合同不符**：`common/baseline_measurement.py:power_report` 用active采样算术均值乘平均CUDA时延，
   未使用采样时间戳逐窗口积分；active标签常在整个forward前设置，而计时从block0开始，边界也不完全相同。
   默认物理GPU索引0未自动跟随 `CUDA_VISIBLE_DEVICES`；另有A100分支显式修正，不能假定所有历史run均修正。
   10/50ms查询间隔不等于传感器的独立测量分辨率；NVIDIA文档区分平均与瞬时功率，平均值可覆盖上一秒。
   应记录实际卡/驱动支持的字段、UUID、传感器语义及完整时间窗，在稳态足够长负载中测量并积分。
   现有数值可作为原算法的功率/能量代理保留，不能称已完成协议要求的精确逐样本能量实测。
   参见[NVIDIA官方功率字段](https://docs.nvidia.com/deploy/nvidia-smi/index.html#gpu-power-readings)。
7. **版本与发布链**：run commit、dirty patch/源码文件清单、完整父config、模型/数据/缓存/初始化/checkpoint SHA、环境和原命令必须绑定。
   审查commit不是训练commit；当前代码可运行也不证明历史结果出自当前代码。T06应恢复原身份的入口，不能跳过SHA验证。

### 本轮执行检查及限制

服务器`xml`环境，隐藏全部GPU并限制CPU线程，执行T01/T02/T03/T04/T06/T08已有测试：**52 passed、2 failed**，14.53s。
两失败均为T06旧Temporal36/Spatial4 profile后端SHA冲突；完整测试命令和失败项在T06机器证据中。
这些检查验证部分结构/合同，不等于任务精度复现，也不覆盖本地尚未同步的T04 embedding-only新实现。
本地base环境PyTorch用户目录DLL导入出现Windows access violation；未在本轮重装或更改用户环境。

文稿定位为**可追溯审查稿与复现操作草稿**，不是已经完成验证的投稿补充材料。
建议先恢复T06入口和数据去重/版本身份，再冻结比较范围及独立评估协议；随后受控修订模型、完成重训/固定权重复评，
最后汇总硬件端到端、原始预测、误差条与论文文字。原始失败/负差距保留，不能为统一叙事而删改。
