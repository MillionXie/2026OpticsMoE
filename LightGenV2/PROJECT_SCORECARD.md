# LightGenV2 任务版本、性能与测速索引

更新日期：2026-10-07。这里只汇总**主线已登记的版本**，不替另一窗口尚未交付的
OpenMoji新实验选模型。每项任务一行，精确PT/SHA、划分、原run和复现命令从任务页查。

仿真、未微调实拍、电子适配后的实拍分开；同权重去光不是重训电子baseline。
不同容量、划分或微调PT不能拼成“同权重仿真—实测差距”。TEST选模按既有人工授权
标为开发指标，不称独立泛化。本次不训练、不重新评估数据集或测速。

## 当前版本一页表

| 任务/入口 | 登记版本 | 仿真主指标 | 未微调实拍 | 适配/校准实拍 | 必要baseline与对照 | 该版本测速与限制 |
| --- | --- | --- | --- | --- | --- | --- |
| [T01 物品检索](tasks/t01_object_retrieval/README.md) | Caltech101 target-10，DC20正式对照 | Top-1 .9000，Top-3 .9650，MRR .9344 | 尚未核验 | 尚未核验 | matched D2NN Top-1 .8950；冻结Qwen .9950；固定专家相位迁移是另一个历史研究 | 旧DC20计算图估算10.061ms/query，旧Qwen模型核心26.407ms；不是实验台端到端 |
| [T02 关键点](tasks/t02_keypoint_detection/README.md) | 官方LSP三类光学候选分别保留；个人照片另列 | DC20 PCK .5772857；低alpha交付 .7347857；alpha40蒸馏 .7282857 | 官方全量实拍尚未核验 | 个人pilot仅伪标签一致性，不作官方成绩 | DC20 D2NN .6751429；旧Deconv128 Qwen .7217143，Deconv40 .6381429，头容量不可混用 | 旧Deconv128 9.504ms：性能1000/batch8，计时200/batch1，显式预热50次；不套后续光电候选 |
| [T03 显著性](tasks/t03_saliency/README.md) | 指定cross-sample best，SHA起始036bc8ca | CC .8624925082；同权重去光 .84229470 | 5000图三层CC .8597739692 | 指定结果未微调 | 同头Qwen历史100轮 .88968469、50轮 .87483828；早期DC20/D2NN .8291/.8346另保留 | 新指定候选未绑定专属测速；旧DC20计算图5.654ms和旧头Qwen10.176ms不移用于新模型 |
| [T04 OpenMoji robust](tasks/t04_openmoji_robust_ablation/README.md) | 2026-10-02封存rank64 G2/G5；现用新实验保护 | 实验台CPU G2 .9390/G5 .9270；服务器 .9385/.9290另列 | G2 .6185/G5 .6910 | 原decoder .9015/.9180；已有bias校准 .9045/.9305 | G1与G2同原PT；旧DC20 D2NN .9895、旧Qwen .5475不是rank64同容量对照 | rank64专属延迟未测；旧DC20 10.862ms、旧Qwen27.166ms仍为历史证据；TEST选PT/校准属开发指标 |
| [T05 视频分类](tasks/t05_video_classification/README.md) | 尚未开展，协议未定 | — | — | — | — | 不为整理自动建模型或启动训练 |
| [T06 视频质量](tasks/t06_video_quality_assessment/CURRENT_VERSION_20261004.md) | Spatial单视频4帧/1M头与Temporal16视频×4帧，两份独立PT | Spatial SRCC .6710968960；Temporal .8043868643 | Spatial .5786364901；Temporal .7977138739 | Spatial不同划分适配见原报告，不取混合TEST最高数替换 | Temporal-36、9×4、旧Spatial及Frozen Qwen分别保留；原Qwen4帧SRCC .7693 | 旧16×4计算图估算28.744ms/场；原Qwen16顺序视频1046.928ms；非SHS端到端、非新Spatial头的时间 |
| [T07 ABO图搜图](tasks/t07_abo_image_retrieval/README.md) | 用户封存rank72/epoch13，SHA起始25f23260；1600真实图库+800查询 | R@1 .83375；同权重去光 .8000 | R@1 .81125/R@5 .94125/MRR .8694430763 | 未微调已达到既定80%门槛，不改最终PT | 原480-query/120商品中心及其他模型对照保持原协议，不拼到本行 | rank72专属测速尚未核验；800查询曾参与开发，不称独立泛化 |
| [T08 ABO双向](tasks/t08_abo_image_text_retrieval/README.md) | 图搜文性能/均衡两版；文搜图10cm主体+10轮读出独立保留 | 图搜文R@1 .79875/.7983333333；文搜图Hit@1 .86、去光 .73 | 图搜文尚未核验；文搜图 .79 | 文搜图采用版 .85，TRAIN拟合/VAL第9轮选模；全项目TEST曾参与开发 | 图搜文Frozen Qwen动态2048D R@1 .7371；固定64D .5358/动态64D .5979，预处理不同 | 旧图搜文Qwen26.052ms；另有仿真.88计时PT，不是文搜图实拍主体，不能套用 |
| [T09 图文/音文判断](tasks/t09_multimodal_matching/CURRENT_VERSION_20261004.md) | seed17探索，逐层OEO；不是自由问答 | 图文MoE .6927；无CNN左右音文MoE .8501 | 未硬件验证 | — | 对应D2NN .5747/.7324；共享CNN音文 .9423/.9487是另一协议 | 历史D2NN前向布局不同，PT形状兼容不代表精度复现；原测速保留，不混作当前版本时间 |
| [T10 专家扩展](tasks/t10_expert_scaling/CURRENT_VERSION_20261003.md) | 多数据集/N/k/seed矩阵，不选择一个数字代替整组 | 按原四组结果矩阵读取 | 未核验 | — | 参数/孔径/源带宽对照分开；固定478九项旧TEST报告仍缺 | 原全部测速保留；源码或矩阵登记通过不等于全部实验完成 |
| [T11 病理终身学习](tasks/t11_lifelong_optics/CURRENT_VERSION_20261004.md) | 纯光二分类协议与CRC9光电蒸馏分别保留 | 历史纯光全量replay四域验证平均BA .8002；CRC9另查原矩阵 | 非实拍 | — | 对应纯光D2NN全量replay .7742；其他联合/有限记忆/冻结迁移对照保留 | CRC9四域是同批图像的变换，不是四个独立数据集；旧时间不套给CRC9 |
| [T12 图文编辑](tasks/t12_text_to_image/README.md) | 17M小版5b4f与电子适配eeec为不同PT；大版及baseline保留 | 原小版PSNR34.277751dB；适配PT自身仿真28.890612dB | 原5b4f EXP27.551289dB | eeec EXP31.552886dB | 大版31.428599dB；同任务Qwen28 27.260453dB；pix2pix-Turbo20.746391dB | 未有统一跨架构新测速；旧9.96M时间不套17M；不能把34.28与31.55当同权重差距 |
| [T13 时序鲁棒消融](tasks/t13_temporal_robust_training/README.md) | full2250 phase-only15四组，schema4 | 原共同条件SRCC .792741/.785567/.803539/.800796，均为仿真 | 尚未完成新一轮实拍 | — | 导师v2同PT重评 .8022806420，名称0.8044是历史标识 | TEST参与选模，末组best为父epoch0；不强造G5最高、不套T06旧计时 |
| [T16 多模态终身学习](tasks/t16_zero_phase_ccd_lifelong/README.md) | 最终A/B/C/D序列及必要对照；不同于T11 | 最终D四任务宏平均召回率均值：MoE全replay .7838 | 非实拍 | — | D2NN无replay .4877、全replay .7579、每旧任务300记忆 .6202；排序探索 .7862不替换主矩阵 | 本版本计时尚未核定；不借其他模型A100/相机时间 |

## 如何查证及保留历史

### T04 的应用版不是 robust 消融版

上表 T04 的 G2/G5 数值属于实验室 robust 消融；同一任务另有
[应用/布局/电子缩减入口](tasks/t04_semantic_interaction/README.md)，目的为不同物体大小、
分层场景与对外复现。指定 epoch45、电子 expansion=.5、DC30＋小CCD噪声版本，
PT SHA `03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21`：
仿真 changed-cell **.8765**、同权重去光 **.4845**；该精确PT的直接实拍、
微调实拍和专属测速尚未核实。TRAIN5000/TEST1000、TEST选定版本属于开发指标。
本地主线及Linux主线严格CPU加载139项state通过，只证明代码/权重构造兼容，
不代表重测精度或完整独立交付包验收。此前另一应用exp05权重的
.9365仿真/.671直接实拍/.8375适配以及旧标准网格测速仍保留，不能套给指定epoch45。
精确原数据、来源与剩余依赖见任务页和 TASK_REGISTRY 的 `application_*` 字段。

### 索引核验边界

- 本表数值来自上述任务页及其原报告，未在整理时重新计算模型性能。
  发布源码、PT、原数据与机器依赖的覆盖见 [TASK_REGISTRY.json](TASK_REGISTRY.json)；
  `migration_complete=false` 不因本表更新而改变。
- 当前时间只绑定相应版本与边界；“未测”不等于0，也不以CPU测试时长代替模型延迟。
- 旧八任务总表完整保存在Git提交 `4df4a1e5c7d4b57d621a46039010262c0200aee2`
  的同一路径；所有原计时/功率CSV、模型/PT、报告和图未删除或搬走。
- 历史光学/Qwen时间和功耗仍见
  [5090D分段报告](tasks/t06_video_quality_assessment/reports/5090d_moe_and_qwen_baselines_20260907/README.md)、
  [旧跨任务baseline](tasks/t06_video_quality_assessment/reports/qwen5090d_cross_task_baselines_20260906/README.md)。
  光学80.388W及持续上电代理不是完整系统实测能耗；GPU平均功率乘均值时间也不是逐样本整机积分。
- LSP原修正归档的头/PT/report与200条计时、83条功率记录可用
  `python maintenance/storage/check_t02_baseline_archive.py --archive <原tar.gz>`
  只读核验；不会解压、测速或运行模型。
- OpenMoji两个目标不可拼接：封存G2直接较G1下降32.05个百分点，校准后距G1
  3.45个百分点；G5直接下降24.80个百分点，校准后距G1 .85个百分点。
  后者接近1%，但不满足“直接下降30个百分点”这一条件。两者均是TEST开发指标。

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

## 历史实验推进顺序（保留记录，不是当前待执行任务）

以下是早期八任务计划，不能据此重启已经封存的 ABO 或覆盖后续版本。当前整理顺序和未完成项以 `../maintenance/storage/REMAINING_CLEANUP_PLAN_20261003.md` 为准。

1. **T06 硬件闭环**：Temporal-36 优先，随后 Spatial-4。先完成速度/功耗/一致性，再报告
   直接部署和微调后性能；这是目前最接近完整论文表格的一行。
2. **T07 ABO 图搜图**：冻结数据与检索协议，先跑可复现 baseline，再训练光电版本。
3. **T08 ABO 图搜文**：协议必须独立于 T07，不能把分类准确率或图搜图结果代替跨模态检索。
4. T01–T04 只迁移可追溯的正式候选并补齐公平 baseline；T05 等数据集确定后再启动。

## 历史对照与测速证据入口（全部保留）

以下旧对照不自动代表上表的最新模型；对应权重、数据和计时边界必须分别核对。

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
