# T20 RGB/SAR 可重构光学系统：光路适配准备

### 当前运行索引（2026-10-25；未进入光路）

| run | 输入/前端 | 用途与状态 |
| --- | --- | --- |
| `unified_static_native8_20261025` | v1分模态位置、原冻结CLEVR前端 | 三case及best/last完整CPU已结束，精度不足；原数据/权重/逐样本保留 |
| `unified_conditioned_native8_20261025` | v1同布局，先拟合读出再调光学 | 被输入诊断否定后停止自己的controller/trainer；PT可加载及SHA确认后停止，`superseded.json`留证，不是数值异常重启 |
| `unified_sensorneutral_native8_20261025` | v2同一三通道布局、原冻结CLEVR前端 | PID3296324/创建1792916031.45；MoE→D2NN_RGB→D2NN_AB，有限训练和CPU审计继续；不热改数值源码 |
| `frontend_domain_diagnostic_20261025` | 同32128参数前端，在本TRAIN两域预训练 | 辅助诊断完成；临时分类头VAL RGB78.6/SAR72.8，不是光学成绩，不替换T21前端 |
| `domain_frontend_frozen_optics_20261025` | v3同布局、两模型共享上项固定前端 | PID3322892/创建1792916989.63；冻结各自原相位、仅拟合原7840读出，再完整CPU复核的受控对照 |
| `robust_readout_native8_20261025` | 同v3推理合同，TRAIN原图/翻转/原图区域dropout及同前端临时头蒸馏 | PID3349346/创建1792917974.51；仍只原7840读出，两个传感器/模型同规则，完整CPU审计中 |
| `domain_frontend_weightnorm_20261025` | 与v3相比仅增加折回原读出的W范数惩罚 | 三case及best/last完整CPU已完成，原数值门槛均通过；用于隔离高增益读出敏感性 |
| `shared_frontend_joint_native8_20261025` | 同一遥感TRAIN前端与v3输入合同，已有光学/电子共同小步训练 | PID3394881/创建1792919604.18，源码b1ded9d45；三case实际TRAIN反向通过，有限40轮→末端拟合→CPU自动接续 |

完整新增结果（每格固定TEST1000，均非实拍；不同run不能逐格拼PT）：

| case | 原冻结CLEVR前端+修正布局/60轮(v2) | 同规格遥感TRAIN前端+固定旧相位/原读出(v3) |
| --- | --- | --- |
| MoE_AB | RGB73.6/SAR60.7，CPU复核中 | RGB82.1/SAR71.0 |
| D2NN_RGB | 接续训练中 | RGB80.8/SAR33.4 |
| D2NN_AB_control | 队列后继 | RGB79.5/SAR69.4 |

v3三个best/last均已完成严格CPU原图重放，argmax与GPU全部一致；两个D2NN最大概率误差.000632/.000642通过原.002门槛。
**MoE最大概率差.005509未过原门槛**，不能仅因分类一致就宣称全部通过；原报告保留且未放宽门槛。
无标签相位诊断确认MoE global3像素/另一个未活动expert1像素在CPU/GPU sigmoid-floor跨8bit边界，
D2NN也各有少数跨bin像素；这是数值实现边界，不是改曝光或模型结构的理由。
`export_canonical_phase.py`将CPU规定的原灰阶码k重参数化到logit((k+.5)/256)，
所有实际CPU8bit相位码和全部非相位张量逐项保持不变，避免GPU浮点实现把临界值分到相邻bin。
它不做训练/选PT/改预测，不覆盖原best/last；两项含全部256bin及原模型CPU输出精确一致性测试通过。
导出至`releases/predeployment_canonical_20261025`的独立PT/manifest。完整2000条CPU证明与原CPU概率**精确一致**，
但MoE GPU对CPU最大概率差仍.005462，未过旧.002；PID3364989队列已据失败明确停止，未继续D2NN导出，不冒称修复/交付实验室。
随后对最差数值样本的8条批次做无标签诊断，所有相位灰阶码已一致，但router/first/final振幅灰阶分别有1/8/11145处不一致，
最终feature最大差.000399，读出Frobenius范数2609.66/谱范数1301.25，logit差.02063。
因此小量浮点差经过中间振幅量化和高增益读出被放大，不能仅把相位临界像素当作唯一原因；失败导出/所有报告原样保留。
单因素`config_domain_frontend_weightnorm_20261025.json`仅增加W平方范数惩罚，原相位/前端/输入/其余数据及损失保持与v3一致，
结果MoE RGB81.5/SAR70.0、D2NN_RGB81.2/31.7、D2NN_AB81.5/70.8；全部CPU/GPU分类一致，最大概率差.001120/.000571/.000322，通过原门槛。
这说明可以改善数值稳健性，但D2NN的SAR迁移仍不足，公平双域控制也未显示MoE明确优势，不得声称纯架构优势已证明。
原`robust_readout_native8`三case也已完整CPU通过，MoE80.0/70.6、D2NN_RGB80.9/33.8、D2NN_AB79.5/69.4；
没有据TEST挑last替换best或逐格拼接。当前共同训练从这组三个数值稳定、已CPU核验的父PT开始，原源PT均不覆盖。
`train_shared_frontend.py`/`config_shared_frontend_joint_20261025.json`给三个case同样40轮上限/min15/patience12、
optical/head LR1e-4、原读出W范数、TRAIN翻转/原图区域dropout与同前端临时分类头蒸馏；最后固定光学拟合原头30步。
前端及显示gain、12个未活动专家冻结；原4个活动专家/router/global及原7840头可训，不新增任何推理层。
VAL按各自训练域选PT，锁定后完整TEST；父PT须通过原CPU/GPU门槛，e0最佳须披露未采用新训练参数，不为趋势挑差PT。
v3数据/权重/前端/报告等38项已本地SHA备份；清单`runs/preparation_20261025/domain_backup_manifest.json`。
RGB-only D2NN的SAR迁移仍明显不足，v3/v4尚不作为原图趋势完整验收，不为贴合旧柱高选择更差权重或削弱对照。

v2已通过30项测试及三case真实TRAIN反向预检，启动源码`ffe15f91c056162a0e17ef14282e389601e0bd61`。
v3源码`455341c309c57f127bba4d4f1425449c89c9a52a`，新增与输入/CPU边界7项测试通过；
`domain_frontend_probe.py`/`config_domain_frontend_probe_20261025.json`固定40步，仅TRAIN与其水平翻转拟合读出，VAL按各自训练域选一个PT，锁定后一次TEST。
预训练前端SHA `fb7d2d52e8ca94fb92d5ee9b81cabb7c7a573d21e405fe96e973140f6803436d`，
其60轮TRAIN训练、按VAL选epoch51，仅原32128特征参数；原24输出临时存储头只用前10类训练，部署时整头丢弃。
**v3架构与v2一致，但前端权重与预训练数据不同**，不能说只是光路几何适配或仍是原CLEVR权重。
所有模型共享同一新前端；D2NN_RGB的“RGB-only”仅指光学/读出拟合，前端预训练已见两域TRAIN，绝不能称整个系统未见SAR。
保留v2原前端对照以区分该预训练收益；v3未自动替换当前候选，未依据TEST挑某个模型来制造趋势。
T21代码、权重与光路不变；所有本轮GPU工作只用原授权物理1号4090，主训练40%/短诊断20%单进程显存上限，GPU0/A100不碰，未创建定时任务。

原v1三case完整CPU与GPU分类一致，最大概率差分别8.04e-5/1.86e-5/2.39e-5，旧.002门槛通过。
best/last/协议/历史/逐样本/CPU报告及诊断等36项已本地备份逐项SHA复验，清单`runs/preparation_20261025/baseline_backup_manifest.json`。
这只证明原失败基线的复现性，不证明性能/趋势已达标。后续v2/v3仍须分别严格CPU核验和同一批ID完整比较后再决定是否可上机。

## 2026-10-25 跟进：定位适配输入问题，统一单传感器布局（最新）

本轮仍只写T20，不修改T21及其共享的`hardware_aware.py`。旧17µm包及已运行候选全部保留。
第一轮v1锁定MoE TEST为RGB635/1000、SAR354/1000；RGB-only D2NN为641/1000、96/1000，均未恢复旧趋势。
原v1 TRAIN/VAL逐阶段诊断定位了如下问题，证据在`unified_static_native8_20261025/MoE_AB_diagnosis`：

- MoE TRAIN RGB61.87%、SAR41.57%，不是仅测试差；SAR所有样本的最大路由权重都落在同一槽，平均功率92.62%。
- 显示截断比例极低，不能把失败归因于饱和。读出梯度范数远大于相位组；分组裁剪/充分拟合原头是训练优化方向，不是新增网络。
- v1为了重建T21双传感器融合，把单独SAR取VV/VH均值置于右下角，而RGB占其他三个瓦片；
  这对原本的单传感器跨域问题额外引入位置差异并丢掉通道信息。此适配选择不应当成原图的等价输入。
  锁定同一D2NN仅换输入布局、未经训练，SAR验证仅9.7→13.2%，说明不能只换图就宣称问题解决。
  TRAIN拟合的输入线性探针、同一VAL对照为旧SAR布局最高19.0%、三通道统一布局57.9%；
  这些是诊断探针，不能作为光学精度或独立TEST结果。

`sensor_neutral.py`改为RGB与SAR一视同仁：保留原始三通道，均使用原T21 `visual_tiles`及**同一个原冻结CLEVR前端PT**，
同一空间位置、同一128特征条和功率归一化；编码没有sensor/类别/ID参数，不新增CNN、alpha、电残差或读出层。
它不再声称能与T21双模态融合振幅线性重建，但仍严格使用同一个T21 native8物理计算图、16槽4活动、7840无偏置头。
RGB编码与v1逐元素相等已测；SAR前端仅使用原冻结权重提特征，不用EuroSAT重新训练前端。
新输入合同为`T20_sensor_neutral_T21_native8_A_graph_v2`，必须新run/数据身份，不能混用v1未来CCD。

最新训练入口`train_sensor_neutral.py`/`config_sensor_neutral_20261025.json`：三case仍为MoE_AB、D2NN_RGB、D2NN_AB_control，
各从对应原锁定PT热启动，按各自TRAIN重新固定显示增益（无现场曝光/坐标/LUT操作）。
同样最多60轮/min20/patience15/每case7200秒，optical/head LR均.001、分组裁剪、TRAIN水平翻转.5，
MoE只增加按传感器分别统计的弱路由均衡损失。前端及增强前端只从原TRAIN图像生成。
先联合优化光学/原读出，再锁定最佳光学，使用TRAIN及其水平翻转充分拟合**原7840读出**；
协方差预条件仅训练坐标变换，保存时折回原无偏置Linear，不增加推理模块。
v1的conditioned接续先拟合高增益读出再更新光学，早期损失上升且未超过其已存读出候选；保留证据，
不将该方案无限排队，不覆盖原最佳。v2修正为最后才做固定光学读出拟合。
所有case相同优化规则与上限；按各自训练域VAL均值/最小值/CE选一个共享PT，锁定后一次完整TEST，
禁止逐格拼PT、压低D2NN、裁剪到原图柱高或训练TEST。若趋势仍未恢复须如实报告。
`audit_sensor_neutral_CPU.py`在best/last上执行完整CPU重放、原相位/gain/逐样本/CPU-GPU差异核验，不据复核重新选模。
仿真达标和物理pilot前不启动SDK；不接管另一分支的光路或任务。

## 2026-10-25 最新接手范围：新包审计与统一T21后端（覆盖下文旧准备状态）

本对话仅负责T20，T21训练、光路、原公共模块保持只读；特别不得修改T21继承的本任务`hardware_aware.py`。
用户要求核对`../EuroSAT_MoE_D2NN_完整数据代码与训练说明_20261025.zip`，
SHA256 `fa35dd1406d3d3575caf99e7a8d5f9c2abac1327e24e5778fad89e0aedc15f7a`。
`audit_updated_package.py`已校验170项清单、全部10000张像素/标签/ID、三划分的pairID/spatialGroup/pixelSHA无交叉，
并由已有预测完整重算五份模型：RGB→SAR无回放MoE阶段1为81.1/52.1，阶段2为72.2/72.3，
额外联合读出MoE82.9/73.8，RGB-only D2NN84.9/70.1，SAR-only D2NN66.3/74.1；每格TEST1000。
这不是重新推理或实拍。训练/验证/测试每域3000/1000/1000，实际图像齐全；顶层额外乱码文件名README不执行。
清单及审计私有索引`runs/preparation_20261025/bundle_audit.json`；仅DATA六文件按SHA物化/传输，未解包或执行第三方源码。

### 与旧图、T21的实质差异

| 项目 | 新交付包 | 本轮统一候选 / T21实际合同 |
| --- | --- | --- |
| 光学边界 | 已有CCD/OEO，不再纯复光场串联 | 直接复用T21原native量化、CCD/OEO接口 |
| 采样/有效像素/画布 | 17µm /478 /518 | 8µm /1016 /1056 |
| MoE容量 | 固定4槽，RGB2→SAR4活动 | T21相同16槽、A阶段4活动；本静态任务不做2→4阶段扩展 |
| 视觉前端 | 两域预训练93472参数CNN，128×7×7空间编码 | 原T21冻结CLEVR32128特征参数及相同PT；RGB原特征条，SAR原VV/VH平均，不加新CNN |
| 末端 | 非线性/rowLN/RMS→Pool16²→LN→带偏置Linear256→10(2570参数) | 原T21均值归一化Pool28²→无偏置Linear784→10(7840参数) |
| 图1训练问题 | RGB→SAR无回放后另做双域读出校准 | 恢复原图MoE双域联合、RGB-only D2NN；另做D2NN双域同数据公平控制 |

“终生学习没有电子前端”不成立：T21 EuroSAT与CLEVR有同一冻结CNN，其临时分类头丢弃；
Speech/Physical不用该视觉CNN。本轮统一候选不再使用新ZIP的额外CNN、空间128×7×7编码及2570参数读出，
原包/原权重/证据保留，不破坏已有实验。

T21 EuroSAT一条记录是同一地点RGB+SAR联合输入、一个类别标签；本实验是RGB或SAR单传感器输入、分别分类。
`unified_native.py`在T21原四瓦片合同内将未观测模态置零，保留当前传感器并归一化为单位功率；
无类别标签、样本ID或真值进入编码/路由，无按模态切头/指定专家。两单模态振幅相加除sqrt2可重建原T21融合振幅。
因此是相同数据源的不同输入信息问题，不能要求准确率和T21 A阶段逐格相等。
本包TRAIN3000地点/域、VAL1000、TEST1000；T21完整TRAIN15998地点、VAL5465、原TEST5429，
其固定分层TEST1000抽样seed20261024，本包TEST按seed42固定SHA排序；不能因为分母都1000就称是同一批ID。
服务器逐图对应检查进一步通过：本包TRAIN6000、VAL2000、TEST2000均为T21原始31996/10930/10858张的同ID、
同像素/标签/模态子集；两个固定TEST1000地点的交集实际为174，不是同一评估集合。
具体证据`runs/preparation_20261025/T21_data_correspondence.json`；原始数据都只读，未为匹配结果重新抽样。
新包CNN联合见过RGB/SAR且MoE旧任务经历无回放遗忘，是原趋势改变的已知协议差异，尚无证据认定只有某个训练bug。

### 统一候选的有限训练与验收

只新增`unified_native.py`/`train_unified_static.py`及`config_unified_static_20261025.json`，不修改T21/原公共类。
三种case为MoE_AB、D2NN_RGB、D2NN_AB_control；前两项保留旧图不等训练域问题定义，第三项才是同域/同记录更公平对照。
两模型同一现成冻结前端、原生相位/振幅8bit前向、TRAIN固定显示增益标定、原7840无偏置头、初始化规则及seed42，
相位从头训练，不拿改变几何与编码后的旧PT冒称等价微调。所有case相同40轮上限/min10/patience10，
batch8、光学LR3e-4、读出1e-4、WD.001、labelSmooth.02，MoE原弱均衡项.15，无增强/教师/TEST梯度。
MoE_AB和D2NN_AB_control每轮6000条两域平衡；D2NN_RGB每轮RGB3000，验证只据各自训练域，
TEST在锁定best/last后单次完整评估，不挑TEST权重或为低于某条柱高选择差模型。
每case一个共享PT评RGB与SAR，不逐格拼接，真实高分不截断；若新对照仍不符合旧趋势，如实标未恢复。
源码先Git发布/服务器测试和真实TRAIN预检，获准非A100限额共享（40%）后才运行；不抢GPU0/A100或停止他人任务。
目前没有本轮统一候选的最终精度或实拍结果，不启动SDK，不占用T21光路；仿真完成仍需严格CPU/PT/相位与新包/新prefix桥接pilot。
`audit_unified_CPU.py`对每case已锁定best/last作原完整CPU图重放、逐样本、相位与固定gain核验，
GPU差异原样保留（旧0.002概率门槛不放宽），不据CPU结果另选PT或编辑预测。
17项原后端/CCD重放/输入合同/CPU审计边界测试已通过，三种case均完成真实TRAIN反向预检，全部应训参数梯度有限。
有限队列已在服务器实际启动：`runs/simulation/unified_static_native8_20261025`，
原始启动源码commit `d080c21a3e6d0b6b46264fd5e6fc8590911b9a14`（T20数值源码为已发布468fa753版），
父队列PID3215388/进程创建时间1792913177.36；一张4090 UUID
`GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d`、每进程显存上限40%。
依次MoE_AB→D2NN_RGB→D2NN_AB_control，每项完成后独立CPU重放与下一项GPU训练并行；
不接SDK、不重启旧任务。启动/测试/数据传输收据在`runs/preparation_20261025/`，实际进度以各case的status/history为准。
报告口径注意：本版GPU评估JSON中`routing.winner_fraction`字段实际保存各槽获胜**次数**，
不是比例；分析时须除该报告样本数，保留原始JSON不覆盖。该命名问题不影响训练、预测或准确率，运行中不热改数值源码。

---

以下是2026-10-24旧包准备与证据，不能替代上述新包结果。

2026-10-24 用户新授权任务。只负责本目录；不覆盖师弟ZIP，不改原T16或已收官Caltech。
状态：原包数据/预测/SHA已审计；新增8 µm原生网格候选代码。**尚未训练实验室适配权重、标定新任务、启动SDK或取得实拍精度，不能称可直接部署。**
服务器四种native模型（T20/T21×MoE/D2NN）CPU前向/梯度检查通过，仿真强度替代相机输入后的logit重放误差为0；这只是软件边界测试。
原包两模型各8条CPU载入smoke的argmax均与原CSV一致，但概率最大差异约0.0030/0.0012，未宣称逐位复现或重算完整图中成绩。

## 原图数据与公平边界

输入ZIP在上一级：`f图_EuroSAT_RGB_SAR_完整代码与结果_20261023.zip`。
SHA256：`7b9cd19058e2808334f57e23895923b99102e163ca633e3d890fcfed2d31f7de`。
93项文件清单SHA核验通过，逐样本CSV与NPZ的ID、标签、顺序一致。

| 原模型 | 光学训练域 | RGB验证1000 | SAR验证1000 |
|---|---|---:|---:|
| LightGen / MoE_AB（图1） | RGB+SAR | 79.50% | 74.90% |
| D2NN_A_only（图1） | RGB | 74.60% | 63.00% |
| D2NN_AB（包内公平回放参考） | RGB+SAR | 72.60% | 74.70% |
| D2NN_B_only（保留证据） | SAR | 56.10% | 73.40% |

图1不是仅架构单因素比较：MoE见过两域，图中D2NN光学训练只见RGB。
两条虚线baseline的身份尚未核定，不按图片像素反推数字。
这里是选模用的**验证集**，不是独立TEST。包内没有独立TEST图像；已有历史TEST元数据不能替代缺失图像。
两域各1000张，每类100；原混合数组RGB/SAR交替，不可直接截混合前1000冒充每域1000。
TRAIN共6000，两域各3000、每类300；没有更改原划分。

## 原模型与实际光路的差别

- 共同冻结CNN：32/64/128卷积，93,472参数，去掉电子分类头；非负128维固定排成224×224振幅，单位功率。
- MoE：router相位224²、四专家相位224²、global478²；共479,364光学参数。
- D2NN：两层478²完整相位，共456,968参数。
- 532 nm，17 µm采样，每段10 cm；478有效口径，518传播画布。
- 十个32²窗口积分功率直接分类。无电子输出头、无alpha、无电残差。
- **expert/input→global之间保留复光场，没有OEO。**单SLM+相机串行替换为强度重加载会丢相位，不能冒称原模型复现。

现有硬件（只读LAB配置核对）：输入SLM1920×1080、相位SLM1920×1200，均8 µm，当前有效方形1016²。
原478×17 µm=8.126 mm，现有1016×8 µm=8.128 mm，口径近似可匹配；并不解决中间复光场问题。

`hardware_aware.py`是**待训练的新OEO适配计算图**：native8 µm，active1016²，canvas1102²，expert476²、gap64、border43。
每层原生像素传播10 cm；中间为测量强度除均值→softsign→单位功率振幅，不再sqrt。
仍无新增电子头/残差/alpha。原raw相位双线性插值只作为显式warm-start，不是等价变换。
目前仍是连续值候选；实际SLM灰阶量化、每阶段固定TRAIN显示增益/相机响应尚未接入验收。
单位L2振幅不可简单乘255导致几乎全零，也不允许逐图峰值拉满；应固定TRAIN标定的共同增益并纳入训练前向。
沿用硬件LUT前仍须核响应；既有linearVoltage LUT不是已证实的线性相位标定。
`reference/`只导入已审计四份原推理依赖，导入改为包内相对导入；原始ZIP成员SHA见PROVENANCE。

## 实验顺序与数据协议

1. 服务器严格CPU核对原PT，适配计算图smoke/梯度/光功率/阶段替换审计。
2. 按真实8 µm/OEO合同重训，先TRAIN-only；保留原图模型不替换。
3. 新任务标定/桥接/24条TRAIN pilot，记录曝光、坐标、LUT、相位与数据SHA，再采整层数据。
4. 每模型同域同组TRAIN/EVAL连续；上游参数改变才新采下游，不把旧CCD套给新输入。
5. 默认先仅光学适配；冻结CNN与十窗口读出不变。若要电子微调或新增读出，属于额外架构方案，需另行授权。

原图两模型×两域=4000条输入。OEO适配后MoE每输入3次CCD，D2NN2次：**EVAL共10000帧**。
若加D2NN_AB公平参考，另4000帧。初步TRAIN采集提案：MoE两域各500，D2NN_A仅RGB500，共4000帧；未启动、不等同原全TRAIN光学重训预算。
此前同相机Caltech2011帧收据实测约0.54–0.62秒/帧；仅作排期参考，T20必须重新实测速率。
按此参考：EVAL约1.5–1.7小时；加上述TRAIN约2.1–2.4小时，不含标定、训练、传输、故障与必要下游重采。

## 科学与发布边界

目标为硬件结果距离**同一适配PT、同一批ID**仿真不超过6个百分点，不保证，不人为压低更好的实测结果。
原图准确率仅原包参考；OEO适配后先重算仿真。TEST/验证选模须标开发指标。
交付并列原模型同子集仿真、适配后仿真和实测三项；若适配仿真本身下降明显，要披露，不能用更低的新基准掩盖总损失。
增强教师默认只用TRAIN；若教师学过评估样本/其增强，学生也继承评估信息，不能称独立测试。
不把评估ID/标签写进编码、缓存查表或推理分支，不为了拟合柱高削弱某模型。
源码走Git；ZIP、PT、数据、CCD留assets/runs，通过manifest+SHA传输，实验室只接收不推送。

```powershell
python -m LightGenV2.tasks.t20_reconfigurable_rgb_sar.audit_package --zip "LightGenV2/tasks/f图_EuroSAT_RGB_SAR_完整代码与结果_20261023.zip"
# 以下仅CPU仿真；新out必须不存在。默认每域全部1000，--per-domain 4仅smoke。
python -m LightGenV2.tasks.t20_reconfigurable_rgb_sar.evaluate_reference --zip "LightGenV2/tasks/f图_EuroSAT_RGB_SAR_完整代码与结果_20261023.zip" --model MoE_AB --out LightGenV2/tasks/t20_reconfigurable_rgb_sar/runs/reference_cpu
```
