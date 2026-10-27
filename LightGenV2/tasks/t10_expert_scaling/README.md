# T10 专家数量与 Top-k 扩展实验

## 逐样本光学Jacobian固定ridge诊断（2026-10-28）

该固定诊断已完成，耗40.23秒，lambda=3.93616e-5，小于lambda的谱方向目标能量比例为0，
不能把失败归因于依赖极小特征值或继续降低lambda。fit均值空间占原Jacobian能量20.28%，未将其从解中删除。
RMS=.0015线性/实际相关.996、无修复/误伤；.005仍没有fit/观察修复，但STROMA保护组新增3错；
.01 fitS/D修1/7、观察1/1，S观察平均margin下降，S/D/COMPLEX保护新增5/1/0错，线性/实际相关降至.864。
没有满足转移/保护的候选，不执行TEST或追加同解步数。逐错核验发现：.005三处S误伤均为S→COMPLEX，
源模型最强错误类也是COMPLEX，但本诊断保护的是S−DEBRIS间隔；成对约束没有保护完整8类判断。
11份小产物manifest+SHA、NumPy逐组指标及128维ridge解/权重/目标已独立复算，源恢复、GPU3释放；
1.56GB Jacobian留服务器并核SHA，未下载或入Git。报告`reports/kather_n49_jacobian_probe_20261028/final_summary_20261028.json`。
下一设计应修正正确类保护的竞争类，并区分样本相关入口调制与末层改动，不重复当前插值；尚无新TEST提升。

按咨询建议先用既有预测计算统一z=log(p_STROMA/p_DEBRIS)，不混合两个标签相反的margin。
在所选S/D干净TRAIN图中，单S/单D方向RMS=.0015的常量移动能量占比分别94.8%/94.2%，
各拟合/观察/保护组z变化均值近似同号同幅；共同方向仅3.3%。这是所选样本的共享边界移动证据，
不是全TRAIN总体或“所有梯度都只是偏置”的证明。CPU逐组均值/标准差存`reports/kather_n49_jacobian_probe_20261028/class_shift_preflight.json`。

新增`jacobian_probe.py`、`configs/kather_n49_jacobian_probe_20261028.json`，同80TRAIN图/160双视图，
只原最后global相位求导。16张fit+48张guard的双视图形成128行真实光学margin Jacobian，
16张配对观察图不进入Jacobian、目标或解。S/D固定对方为竞争类，COMPLEX按源模型逐视图固定最强错误竞争类；
候选仍按全部八类argmax计实际修复和新增错。Jacobian约1.56GB FP32留服务器，仅解128维系统，不构造参数Hessian。
fit目标max(0,.05−源margin)、guard目标0；两类fit和三类guard各按本组行数归一化，不逐行单位化梯度。
A=sqrt(W)J，t=sqrt(W)b，固定lambda=.01×trace(A A^T)/128，float64求解后Delta=A^T(K+lambda I)^{-1}t。
guard目标0是软约束，不是保证。只解释类均值空间占比，不先删除fit均值而使目标不可达。
记录谱与小于lambda方向上的目标能量、一阶各组收益/误伤、原解RMS；仍只试RMS=.0015/.005/.01三次精确源恢复，
比较一阶与实际margin变化、配对观察转移、干净/D4三类保护误伤和竞争类切换。
没有VAL/TEST前向、不搜索lambda、不做连续优化；拟合成功而观察不转移时不再降低lambda追插值，
只有双侧观察收益兼顾保护才支持另一个有限短训练，当前616目标仍未达成。

## 双向类别梯度与难度配对的TRAIN转移诊断（2026-10-27）

前一expert块比较也已完成：三个共同方向源恢复点耗31.70秒，cos=−.9216，仍高度对立；
RMS=.005拟合S/D修6/6、观察1/3，正确S/D/COMPLEX保护图新增4/1/4错；.01观察修3/7，保护新增6/2/7错。
S观察组平均目标间隔在三个幅度均下降，不能把拟合组进步作为扩大前层训练的理由。
10份小产物manifest+SHA和NumPy逐样本指标已核，两次诊断TRAIN身份逐位一致，源恢复、GPU释放；
报告`reports/kather_n49_expert_tail_transfer_probe_20261028/final_summary_20261028.json`。
此结果只否定本次两个层块的所试类平均方向，没有证明全部光学方向不可用；下一诊断须区分类偏置与样本相关梯度残差，
不能继续同一种投影、加学习率或无依据展开前层训练。当前最好TEST仍600/752，未达616目标，未新增TEST前向。

上述九个源恢复试探已完成、16份产物SHA/NumPy分组逐样本结果已核验，耗32.49秒，源恢复、GPU3释放。
两类末层global修复梯度余弦为−.9391；单类方向能修复同类独立TRAIN观察组，却严重伤害另一类正确图。
归一化共同方向RMS=.005在独立观察S/D组分别修3/1张，但正确S/D/COMPLEX保护组新增2/1/2错；
.01分别修4/3张、保护组新增3/4/5错。没有测试点满足保护条件，不能据此继续同一末层投影训练，
也不能把这些TRAIN数字称作TEST提升。报告`reports/kather_n49_boundary_transfer_probe_20261027/final_summary_20261027.json`。
为比较是否是当前更新层的方向冲突，新增有限profile`configs/kather_n49_expert_tail_transfer_probe_20261028.json`，
同一入口仅把求梯度块切为原`expert_phases.1`（2,458,624参数），冻结原末层global及其他块；
只试共同方向三个同幅度源恢复点。缓存前缀为原前两特征层OEO输出，suffix仍调用原第三专家层、第四global层及探测器，
不改model.py或推理计算图；初微批必须与完整模型逐位一致。所有TRAIN图/配对/D4、源权重和保护观察相同，
先测梯度夹角/独立转移/误伤对比，不直接启动更大范围训练，也不宣称新层必能解决冲突。

受约束更新的修复集中在STROMA，完整开发TEST新增错误却来自DEBRIS，不能继续同一配置加步数。
经“微调方法总结”只读复核，源TRAIN中S→D/D→S在logmargin[-.5,0)的错例分别52/36张；
排除旧各16张拟合图后，分别还有36/20张，足够开展新的互斥诊断。
`boundary_transfer_probe.py`及`configs/kather_n49_boundary_transfer_probe_20261027.json`
分别取新错例按源margin排序的16张、相邻配成8对，每对按固定seed随机分配一张求方向、一张观察，
避免“最近边界拟合、较难样本观察”的难度偏差。原正确STROMA/DEBRIS/COMPLEX每类16张，
各含8张margin(0,.2]和8张(.2,.5]；全部80唯一TRAIN图，各干净+固定D4，共160视图。
只对每方向8张拟合图双视图求修复梯度，观察与正确保护图不进梯度。
记录两类梯度范数/夹角与真实方向导数，比较单修S、单修D及两类分别L2归一化后相加的共同下降方向。
每方向仅RMS=.0015/.005/.01三次源点精确恢复试探，最多九次，固定原E117末层global相位、原完整光学合同，
不训练连续路径、不读取VAL/TEST预测、不扩seed。报告各方向拟合/观察两侧的实际修复、新错、pair-margin和全类margin，
保护图三个类别分别报告。共同方向的一阶下降不保证有限幅度或跨样本收益，只有两侧观察都改善且保护误伤受控，
才支持后续短训练；单侧改善则不重复同一种投影。源权重/前缀SHA、ID互斥与缓存/整模型等价必须核验。

## TRAIN方向冲突实证与受约束更新（2026-10-27）

后续`kather_n49_projected_repair_20261027`已完成：接受2次整池更新，第3方向的3个回溯均不满足预定条件而拒绝，
没有增加预算。TRAIN候选锁定后，完整TRAIN3141→3148正确、VAL589→585、开发TEST600→597；
逐样本TEST纠正0/新增3，全部新增错来自DEBRIS。子集6张修复全部为STROMA，反向DEBRIS错例未修复，
COMPLEX保护组无误伤，独立随机TRAIN观察组正确数不变，说明随机小观察组未检出该边界方向的转移问题。
原模型600继续保留为best，候选last SHA`cee3c264…539c`及其597张正确的预测保留；不能把回退后600当作候选成绩。
14份小产物已manifest+SHA传输并独立复算采样分组/所有5个试探逐视图正确数、752TEST的argmax/cm/finite及源/权重SHA，
PID3155358退出、GPU3释放。全程158.19秒。报告`reports/kather_n49_projected_repair_20261027/final_summary_20261027.json`。
尚未达到616/752目标，不扩种子；下一步须分别检查双向修复梯度和独立TRAIN近边界转移，
并保护原正确DEBRIS与COMPLEX，而不是继续同一配置加步数。已将证据反馈“微调方法总结”只读咨询。

有限方向诊断`kather_n49_phase_repair_probe_retry_20261027`已完成八个试探，耗68.50秒，
128唯一TRAIN图/256视图；16份产物manifest+SHA及分视图逐样本正确数已独立核验，源相位精确恢复、GPU3释放。
Adam方向RMS=.0015修复4/32张干净近边界错例、误伤1/32张COMPLEX；.005修复10张、仍误伤1张；
负修复梯度RMS=.0015修复16张但误伤18张。故末层确有局部修复能力，同时存在类别方向冲突，不能简单加大学习率。
小batch梯度分解也在TRAIN最修复点测得保护梯度与修复方向相反；不能把此小样本统计当作全TRAIN结论。
没有执行新VAL/TEST，当前全模型开发TEST最好仍600/752。初次诊断因输入元组调用错误在首批前退出，
保留失败run与成本，修复后补入口/D4测试，再从源重做，非训练恢复。报告`reports/kather_n49_phase_repair_probe_20261027/final_summary_20261027.json`。

下一入口`projected_repair.py`、profile`configs/kather_n49_projected_repair_20261027.json`，
仍同一TRAIN图/固定双视图，仅训练原最后global相位，不开放前层。每个候选方向先从fresh AdamW预条件方向中，
用最多三轮循环投影去除降低原正确COMPLEX相对STROMA/DEBRIS间隔的一阶分量；它不是精确QP，
最终必须用实际非线性输出的回溯检查保护条件。每步至多试RMS=.0015/.00075/.000375，
相对固定原E117总圆周RMS仍≤.02；最多8次整池更新/24次试探，整个方向拒绝即结束，不空转。
目标包含修复平方margin、COMPLEX间隔保护、父KL、原capture/TV/balance和相位锚；连续接受路径保留Adam状态，
拒绝恢复权重及Adam状态。没有EMA、电子头或TTA。
接受要求：干净修复≥2；COMPLEX干净和原正确D4不新增错；回放干净新增≤1；干净总正确数相对源净增；
修复D4平均margin继续增加。当前总正确数增加，或同正确数下三组NLL之和下降>1e-4，才可接续。
独立TRAIN观察组不进梯度也不用于接受选择。若有接受更新，先锁TRAIN候选，再用原完整模型评3496TRAIN/752VAL及一次开发TEST，
与原600候选按正确数/NLL比较；无接受更新则核源权重复用，不重测源。所有新结果仍属TEST反馈开发，不替换原等预算表。

## 末层边界试验结束与TRAIN方向诊断（2026-10-27）

`kather_n49_boundary_refine_20261027`两组各12段/384步/6144个TRAIN视图已完成。
control开发TEST第4/8/12段为599/596/596张正确，boundary为599/596/595；均选回原E117的600/752，
没有达到单seed保守目标616/752。boundary末次纠正3错却新增8错，COMPLEX净少7张。
两组best SHA与原模型一致，TEST核验复用；全部59份小产物已manifest+SHA下载，逐样本argmax/finite/cm、
采样顺序/身份、路由、冻结前缀及best/last SHA独立核验，父/缓存/worker退出、GPU1/3释放已实查。
报告`reports/kather_n49_boundary_refine_20261027/final_summary_20261027.json`。大fields/PT保留服务器。

同池EMA干净TRAIN准确率变化很小，不能把换池后的准确率升高当作训练收益；日志gradient_norm是段末batch，
不是全池范数。KL/锚定损失标量小、RMS未触上限也不能排除梯度冲突，RMS更不能直接等同输出改动大小。
按“微调方法总结”后续只读建议新增`phase_repair_probe.py`、
`configs/kather_n49_phase_repair_probe_20261027.json`，在GPU3做有限TRAIN诊断，不再盲加正则或启动SAM。
从固定原E117出发，只按TRAIN选择最多各16张近边界STROMA↔DEBRIS错例（log-margin在[-.5,0)，不足则缩小），
32张正确COMPLEX保护图、32张八类回放及32张不进梯度的独立TRAIN观察图，各干净加固定D4视图。
对负修复梯度与Adam预条件方向分别试RMS=.0015/.005/.01/.02，共八个精确恢复源的方向探针，
不等于八条训练路线，不执行VAL/TEST前向，也不把TEST样本加入梯度。
记录修复/新增错、分视图margin与概率变化、末层OEO激活变化；源点及一处TRAIN最修复点另做固定小batch的
repair/COMPLEX保护/KL/capture/TV/锚定梯度范数和夹角诊断。源点保护梯度为零是定义结果。
只有诊断后才能区分局部步幅/方向不足、修复方向冲突或需开放前层；没有预设诊断必能提高性能。

## 冻结前缀的末层边界修复（2026-10-27，采用窗口咨询建议）

六份保留last已在GPU1完成一次有限开发TEST，原90/120、focused颜色/相位/蒸馏、Mixup分别为
592/598/593/594/595/596张正确，均未超过原验证选定模型600/752。19份产物传输SHA与全部逐样本结果已独立核验，
审计PID退出、两卡释放；报告`reports/kather_n49_checkpoint_dev_audit_20261027/final_summary_20261027.json`。
这排除了本次列出的六份末轮权重中存在漏选过线模型的可能，不代表检查过所有历史中间权重。

新模块`boundary_refine.py`、profile`configs/kather_n49_boundary_refine_20261027.json`，
run `runs/simulation/kather_n49_boundary_refine_20261027`。按“微调方法总结”窗口建议，优先只训练最后global相位
3,055,504个参数，冻结router及前三层；推理仍原ScalingOptics N49/k49、4层逐层OEO、原detector、总参数11,078,432。
两组从同一原120轮VALbest E117 EMA（SHA9a6929c…956）初始化，fresh AdamW、weight_decay=0，不冒称恢复原动量。
GPU1为原CE(.02smoothing)+.2倍线性margin对照，GPU3为去掉CE的平方margin边界组。
两组都使用同一full49父模型概率保护及相位锚定，唯一主要监督目标差异为CE+线性hinge与margin-only平方hinge。

边界项为mean(max(0,.2+log p_wrong−log p_y)^2)，使间隔足够的样本不再贡献该监督梯度；
父模型正确且log-margin≥.2时KL权重.5，其余.05，按全部microbatch平均。原capture.005、TV.02、softbalance.01保持；
周期相位锚定50×mean(1−cos(delta))，用等价的100×mean(sin(delta/2)^2)避免小角度消减误差；
每步把相对原父模型的圆周相位RMS投影到≤.02rad。最后一层仍影响全部类别，不能冒称单类相位行。

两组共用三个TRAIN-only池，每池256张、每类8低log-margin＋24轮换回放，每张两个固定D4视图，512优化输入/段。
每四段更换池；每组固定12段、每段32个effective16/micro2 AdamW更新，共384步/6144个优化输入，
两组合计12288输入，约1.76个3496样本全TRAIN等效epoch/组。这里的双视图不是512张不同图像。
LR第1段5e-5、第2段1e-4，再余弦到第12段2e-5；EMA.95。连续live/optimizer路径不因短时持平回滚，
固定父模型KL/相位锚不随checkpoint选择变化。只在第4/8/12段锁定EMA候选后检查开发TEST：正确数优先，
相同时NLL需下降>1e-4才更新best，原600张正确模型保留。实时报告live/EMA VAL与净修复/误伤，
不能用单seed或测试反馈开发成绩声称三seed均值或独立泛化。

缓存是原冻结前缀的float32 OEO实幅度，绑定源权重/输入ID/D4码/代码/数据SHA；两组从同一缓存取相同视图和顺序。
每个512视图池约6.55GB，三个池连同VAL/TEST前缀约38.9GB，留服务器CPU内存映射文件，不传输或进入Git，微批送GPU。
缓存生成器先在GPU3单独完成并退出，之后才启动GPU1/3两个训练worker；不会另起竞争GPU任务。
初始缓存输出须复现原模型；前缀参数SHA逐项保护。最终选中模型用原整模型确认、评全3496 TRAIN/752 VAL，
新权重的最终TEST还需与缓存分数比较；原权重回退则核验复用旧TEST，不重新forward。
路由只取决于冻结的router/原RGB输入，selected route NPZ在ID/labels及冻结身份核验后复用并留收据，不冒充新路由分工。

## 按“微调方法总结”经验修正开发顺序（2026-10-27）

用户明确要求向“微调方法总结”窗口学习。已只读交流并核对该窗口的实际经验：先区分困难类别训练不足与泛化差距，
优先冻结前缀、仅改最后一层光学相位，以margin-only监督、同一full49父模型概率约束及相位锚定保护旧能力；
全参数SAM作为后备，已发布的SAM配方尚未正式启动。该窗口没有修改T10或使用本任务GPU。
其电子分类头单行修补/257维曲率方法不能照搬到T10的3,055,504参数末层global相位，后者仍影响全部八类。
末层缓存只读探针已证明：缓存最后global层输入的OEO实幅度后，原末层及探测器输出与整模型逐位相同。

先用`checkpoint_dev_audit.py`及`configs/kather_n49_checkpoint_dev_audit_20261027.json`，
在GPU1对预先SHA锁定的六份保留last做一次有限TEST开发比较：原90/120轮full49 last、focused颜色/相位/蒸馏last、Mixup last。
run `runs/simulation/kather_n49_checkpoint_dev_audit_20261027`。原600/752模型始终候选，旧VAL选模正式表不覆盖。
这是用户已授权TEST反馈后的checkpoint开发选择，不是新独立测试；不训练、不扩seed，也不增加其他架构或TTA。
若无过线模型，再按咨询建议做小规模末层边界对照，并在预定短段节点直接检查开发TEST正确数/NLL；
不能继续只按VAL留下唯一checkpoint而忽略已授权的开发反馈。所有新流程须另有配置、损失/缓存身份检查及有限预算。

## 硬标签优化与SAM成对对照（2026-10-27）

Mixup与功率渐进阶段已完成，未达到616/752目标；不能把“阶段完成”当作性能目标完成。
纯Mixup实际20轮、20480个TRAIN优化输入，验证选第2轮：TRAIN89.87%、VAL78.86%、TEST595/752=79.12%。
对原120轮600/752的配对分析：纠正5错、新增10错，STROMA/COMPLEX分别净少4/5张。
功率渐进完成32轮、32768个优化输入，其中17轮完全原始49路；末轮VAL45.08%，没有恢复源性能。
其验证选模回退原120轮full49模型，核验相同checkpoint/数据/依赖/预测SHA后复用原600/752=79.79% TEST，
未将过渡模型成绩当作49/49。两项best/last及全部失败轨迹保留；55份最终产物manifest+SHA、
逐样本accuracy/cm/argmax/finite/IDs、所有逐轮采样和五组梯度独立核验通过，全部原父子进程退出、两卡各12MiB/0%。
报告`reports/kather_n49_mixup_curriculum_20261027/final_summary_20261027.json`；
复算入口`summarize_mixup_curriculum.py`。源代码身份仍为`252fed1a07056ab032e2e300b4e3e04f849b7e2e`。

干净TRAIN与VAL同时退化，不支持把所有失败都归因于经典过拟合、再盲目增加输入扰动。
新入口`sharpness_refine.py`、profile`configs/kather_n49_sharpness_refine_20261027.json`，
run `runs/simulation/kather_n49_sharpness_refine_20261027`。只seed17，GPU1硬标签AdamW对照、GPU3 SAM-AdamW。
两组从同一原120轮full49验证best EMA（源实际E117，SHA9a6929c…956）初始化，均fresh AdamW及seeded RNG，
不是精确resume。原始ScalingOptics N49/k49/routerT.05/参数11078432/OEO/数据划分/原D4全部保持；
不使用Mixup、颜色扰动、功率过渡、教师、新头或TTA。原CE标签平滑.02、margin.2、capture.005、TV.02及softbalance.01不变。

每组固定24个1024样本round，每类轮换128、4轮覆盖3496TRAIN；无三轮早停。最多49152个优化输入，
约7.03个全TRAIN等效epoch/组。两组相同数据/增强/学习率：expert预热1e-4→3e-4，再余弦至3e-5，global.5/router.1。
SAM依据[原论文](https://arxiv.org/abs/2010.01412)，在完整effective16梯度累积后按全体相位参数的
全局L2半径rho=.05rad作一次梯度方向扰动，对同一增强batch再次前向/反向，逐张量精确恢复原参数后再执行
裁剪AdamW及EMA更新。rho不是每像素噪声标准差。与硬标签对照唯一优化差异是SAM的上升/第二遍反向；
SAM因此增加一倍前反向计算，不能把相同图像数冒称相同计算预算，也不保证提升。

每轮512 TRAIN探针及752完整VAL，每8轮完整3496 TRAIN评分；分别记录clean/perturbed损失、sharpness差值、
实际扰动范数、恢复校验、五组梯度/学习率/耗时及采样IDs。仅VALaccuracy/NLL选择checkpoint，原best保留候选。
每组训练及选模结束、GPU1训练槽空闲后立即锁定并TEST一次，SAM仍在GPU3训练时允许先测硬标签对照；
不会再为等待更慢的一组而推迟已完成候选的TEST。相同原权重继续核验复用旧TEST。
用户授权的TEST开发比较及616门槛不变；仍不是独立确认性泛化或等预算Top-k表，不自动跑27/37。
本次成对对照用于检验优化邻域稳定性假设，不能把方法名称或梯度正常当作有效的证据。

## 追加目标：前一Top-k至少＋1个百分点（2026-10-27，用户新授权）

用户核定未达标继续利用TEST反馈做微调。本轮TEST可参与两个配方的开发比较，明确是已观察测试集的超参开发，
不冒称独立泛化或等预算消融；仍仅TRAIN样本进入梯度。原结果、全部失败路径/预算、best/last永久保留。
上一轮N49/k49 seed17选定TEST79.12%（595/752），确实未达标。同seed90轮k24为80.32%（604/752），
＋1pp需至少612/752=81.38%；旧k24三seed均值80.90%，＋1pp约81.90%。本轮单seed开发使用更保守的
**616/752=81.91%**门槛；单seed过线不等于已证明三seed均值过线，暂不扩seed27/37。

新入口`mixup_curriculum.py`、profile`configs/kather_n49_mixup_curriculum_20261027.json`，
run `runs/simulation/kather_n49_mixup_curriculum_20261027`。只用GPU1/3各一路，UUID与上一轮相同。
对比两种训练路径：原120轮full49 VALbest＋Mixup，及同seed90轮k24 VALbest＋输入功率连续过渡＋Mixup。
直接把k24权重改为k49，未训练VAL从80.98%降至18.62%；k28/32/36/40/44分别68.09/62.63/55.32/40.43/29.65%。
因此不能直接移植并把稀疏模型成绩冒充full49。本次过渡的必要性是待验证假设，未承诺过渡能保住准确率。
原始诊断`runs/smoke/kather_n49_curriculum_transition_20261027/receipt.json`，参数11078432，未读新TEST。

过渡定义：归一化top24入口功率p24与原full49概率p49按(1−t)*p24+t*p49混合，再sqrt转振幅，
t随前16轮余弦从0升至1，第16轮起完全调用原ScalingOptics k49路由。t>0全部49入口功率为正；
t=0仅作原k24初始化复现。任何t<1的checkpoint不得参与full49选模/测试，最终TEST只用原始ScalingOptics N49/k49，
无额外推理头、教师、TTA、学习参数或永久偏置。训练期使用与原实现同定义的dense-amplitude STE。
共享model.py/train.py/plan.py/OEO不修改，专用factory仅用于过渡训练，t=0/1需实际输出复现检查。

Mixup按[原论文](https://arxiv.org/abs/1710.09412)在TRAIN配对图像及标签上做凸组合，alpha0.2；
每个effective batch16的图像经各自确定性D4后混合，标签CE和margin按同lambda加权，原capture/TV/soft均衡总权重不变。
不再采用上一轮50%困难样本反复追逐；每轮各类均匀轮换128张，共1024唯一样本，每4轮覆盖全部3496TRAIN。
新AdamW/seeded RNG从源选定EMA初始化，不能称恢复旧动量的精确resume。专家LR3轮预热1e-4→3e-4，
global0.5倍/router0.1倍，余弦至3e-5；原EMA0.95、labelsmoothing.02、TV.02、softbalance.01、capture.005、margin.2。

每配方最多32聚焦round，总65536 TRAIN图像曝光。每轮512 TRAIN探针＋752VAL，每8轮保存完整TRAIN评分，
每分钟保存批次/梯度/功率比例；完整采样IDs逐轮写sampling_history.jsonl，修复上一轮中间难度缓存覆盖导致历史回放不可完全追溯的局限。
只在完全full49阶段累计early-stop观察，至少12个full49 round后才开始8轮无accuracy/NLL纪录改善早停；
过渡模型不能因稀疏VAL高而选入。原120 full49 best始终作回退候选。各配方结束后按VAL锁各自full49 checkpoint，
每个候选TEST一次（权重完全相同的原候选核验后复用），再按用户授权的TEST开发成绩比较配方及616门槛。
本阶段是有限方法比较，不承诺必达标；后续调整须有训练/验证/开发TEST证据、明确新增预算和独立profile。


## TRAIN难例回放与评估降频（2026-10-27，用户追加要求）

### 本轮最终结果：三方法完成并测试，性能目标未达到

三组均仅seed17，按预定最少10聚焦round后耐心6停止；颜色16轮、颜色＋相位噪声17轮、筛选蒸馏20轮，
合计53个1024图像聚焦round、54272次TRAIN图像曝光。此前已完成的短试验与extended训练成本另存，不隐藏。
验证accuracy优先、NLL平局选择蒸馏EMA逻辑E144（该新阶段第14round，计数不等于训练144个全量epoch）；
其他两方案未读TEST。全量TRAIN指标来自3496张最终评估，区别于逐轮512张TRAIN探针。

| 方法 | 完成聚焦round | 验证选中round | 全量TRAIN | VAL | TEST |
|---|---:|---:|---:|---:|---:|
| 颜色增强 | 16 | 12 | 90.16% | 78.46% | 未选，不读取 |
| 颜色＋相位噪声 | 17 | 12 | 90.07% | 78.46% | 未选，不读取 |
| 颜色＋筛选蒸馏 | 20 | 14 | 90.10% | 78.59% | **79.12%（595/752）** |

已下载65份最终JSON/NPZ/记录，所有传输manifest SHA及三方法best/last SHA核验通过。
对TEST752个样本独立重算accuracy、混淆矩阵、argmax、有限分数、跨版本相同IDs/labels，均通过。
验证比原120seed17多对2张，TEST却比原120的79.79%（600/752）少对5张（−0.66pp）；
配对错例分析为纠正3错、同时新增8错，主要净损失为COMPLEX类6张。相同seed的90轮49/k24为80.32%（604/752），
本轮低1.20pp；相同seed的90轮49/k49也是79.12%，但不是同一权重或同一逐样本预测。
此前90轮49/k24三seed均值80.90%±样本SD0.50%仍单列为旧结果，本单seed开发不能报告三seed均值SD或当作等预算消融。

训练采样提高了难例密度和轮效率，但当前没有得到净TEST收益；全k路由hard全1是定义，soft功率仍接近均分，
未证实专家分工改善。圆周SWA已按VAL比较，最终选择EMA。当前不扩其他seed、不按TEST结果再追加预算。
三方法所有训练/评估进程与父调度器均退出，GPU1/3实查各12MiB/0%，无本任务显存占用；
`final_independent_server_audit.json`与`release_check.json`留证，旧原始权重、失败数据和全部结果保留。

证据与复算入口：
- `reports/kather_n49_focused_adaptation_20261027/final_summary_20261027.json`
- `reports/kather_n49_focused_adaptation_20261027/final_convergence_20261027.png`
- `python -m LightGenV2.tasks.t10_expert_scaling.summarize_focused --snapshot <downloaded_final_snapshot> --report <report_directory>`


启动核验：旧extended父子进程均已退出，颜色两路从完整E127、蒸馏从E130接续；三路初始VAL与源记录复现，RNG恢复。
TRAIN-only采样审计：1024样本中保留约311–318个当前错例（全TRAIN约344–349错），约30.4–31.1%样本为当前错例，
而均匀全TRAIN约9.8–10.0%；类计数128、ID唯一及与VAL/TEST不相交均由服务器和下载后NumPy独立核验。
蒸馏首聚焦轮实测训练96.75秒＋TRAIN探针10.32秒＋VAL15.14秒，合计122.20秒；旧3496样本全量轮中位约421秒。
这比较的是1024与3496图像的不同训练轮，不能称相同计算提速3.4倍；初始化评分另70.43秒、每5轮刷新及第10轮后SWA另有开销。
首聚焦轮VAL77.93%仍低于源78.32%，当前只证明针对难例的计算利用和轮耗时改善，尚未证明精度提升。
证据：`reports/kather_n49_focused_adaptation_20261027/initial_error_speed_analysis_20261027.json`；源码发布`1d9cdd8b3`。

用户要求分析已观察TEST的错例、减少不必要的全量TRAIN重复并加快速度。对齐seed17旧TEST预测：
49/k49累计120轮为600/752，49/k24累计90轮为604/752；教师可纠正学生48个错误，学生也能纠正
教师44个错误，二者共同错104个。学生152错中118个来自TUMOR/STROMA/COMPLEX/DEBRIS四类；
STROMA→DEBRIS22、TUMOR→MUCOSA16、STROMA→COMPLEX13、DEBRIS→STROMA13。
这些是用户授权的已观察TEST开发诊断，不是独立确认性分析；绝不把TEST样本加入TRAIN，也不按TEST挑训练样本。

新入口`focused_adaptation.py`、profile`configs/kather_n49_focused_adaptation_20261027.json`，
run `runs/simulation/kather_n49_focused_adaptation_20261027`。只继续seed17三方法，GPU1蒸馏、GPU3颜色/相位噪声，
资源与原光学推理合同保持。从活动`extended_adaptation`的下一个完整checkpoint边界交接：先核PID及启动身份，
冻结旧调度器防重复，等待各worker写完新last，保留live/EMA/AdamW/完整RNG及源best，再退出旧父子进程、启动新worker。
独立有界SIGCONT保护处理交接异常；旧run、全部权重和历史不删除。完整checkpoint后到冻结之间极少量下一批工作可能未保存，
如实记录，不冒充任意批次精确恢复。

按当前EMA在TRAIN上的clean正确类margin排序，每类选64个低margin难例＋64个轮换均匀回放，共1024个唯一样本/聚焦轮。
固定类平衡；保留容易样本回放而不是永久删除已学会样本；每5轮重新评分刷新难例，避免持续追逐陈旧或噪声样本。
这是有意改变TRAIN采样目标的难例优先开发，不能称为未改变原训练分布的无偏全量训练。
每轮只评固定512张类平衡TRAIN探针，完整3496张TRAIN仅在初始化、每5轮刷新及最终评估；752张VAL仍逐轮全量选模。
训练micro2/effective16保持；评估batch4只改变计算分批。分别记录训练/探针/VAL/刷新耗时，不能把探针准确率冒充全量TRAIN。

每个方法最多20个1024样本聚焦轮，且不超过原20轮适应预算剩余的TRAIN图像曝光数；旧完整epoch与新聚焦round分别报告。
新阶段前10聚焦轮不累计早停，之后连续6轮VALaccuracy/NLL均无纪录改善才停。20round约等于5.86个全TRAIN epoch，
不能写成追加20个全量epoch。LR新阶段3轮预热1e-5→3e-5→5e-5再余弦至5e-6，global/router倍率不变；
蒸馏权重已预热到1并保持，仍仅对正确且更自信的冻结同seed教师TRAIN输出做筛选KL。
第10聚焦轮起比较圆周SWA候选，推理仍原49/k49、无教师/新电子头/TTA；单seed试验不报告三seed均值SD。
三方法完成后只按VAL选择并锁权重，再TEST或核验复用张量相同的旧预测。训练预算、TEST已观察、实际来源及评估降频均必须披露。


## 单seed充分适应续训（2026-10-27，用户最新要求）

用户明确指出三轮筛选过短，要求先在一个seed上找到有效方法，再跑其他seed，并追加GPU1。
因此此前三轮早停只能视为短期筛选，没有充分证据认定蒸馏或增强方法普遍无效。
新profile `configs/kather_n49_extended_adaptation_20261027.json`与入口
`extended_adaptation.py`替代旧的小预算筛选；run为
`runs/simulation/kather_n49_extended_adaptation_20261027`。仅seed17，暂不启动seed27/37。

三路为颜色增强、颜色＋相位噪声、颜色＋正确性/置信度筛选蒸馏；前两路在GPU3并行，
蒸馏在GPU1，UUID分别固定为`GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd`与
`GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d`，不用其他卡。
先等待旧短试验完成选模及释放，再按SHA锁定各自last/best/metadata/history。
颜色两路接续`kather_n49_training_robustness_20261027`，蒸馏接续
`kather_n49_fullk_refine_20261027/distill_s17`，恢复last的live、EMA与AdamW状态，保留源best候选，
不反复从起点重训。源没有完整RNG检查点，续训按seed初始化随机状态，不冒称位级精确resume；
新last保存后续RNG状态。源记录永久保留。

每个方法累计最多20轮适应训练，已完成的3–4轮计入；新阶段前10轮不累计早停，
之后连续6轮VALaccuracy与NLL均不刷新各自最佳才停。若源有3轮，最早累计19轮才可能早停，
不会再因开始三轮的波动结束。专家LR先用新阶段3轮从1e-5预热至5e-5，随后余弦到5e-6；
global为0.5倍、router为0.1倍。batch16/micro2，原监督/margin/capture/TV/balance保持。
三项均使用同定义的轻度颜色增强；仅phase_color启用TRAIN相位噪声0.02 rad。
蒸馏教师仍为同seed的90轮49/k24 VAL best，仅处理同一增强TRAIN视图；KL权重在新阶段3轮
从0.3逐步增至1，保持类别温度2及原筛选，避免辅助信号长期过弱。

新阶段第10轮起，额外比较EMA相位的跨轮圆周平均候选（先累积单位复数相量再取角度，
不直接平均跨越±π的相位）。只用VAL比较EMA、圆周平均与继承的best；模型结构和参数数目不变。
每轮记录干净TRAIN/VAL、逐组梯度、蒸馏覆盖率/损失、两个候选及耗时，每分钟记录批次进度。
三个方法全部完成后按VALaccuracy/NLL选方并锁权重，再对选定权重TEST；若张量与原模型
相同则核验后复用原TEST并写收据，不重复forward。

这是一轮新的、用户要求扩大适应预算的方法开发：此前自设24个epoch-job筛选上限不再适用。
三方法各20轮上限共60个epoch-job，其中源训练3–4轮/方法复用，最多追加51个；旧筛选和降温
费用另列，不能省略。蒸馏从旧三轮方法切换为颜色增强＋较大KL权重，属于新阶段，不能将整个
轨迹冒充一份从零开始的统一协议。推理仍为原49/k49、原router温度0.05、干净单视图，无教师、
额外分类头或测试增强，eval关闭相位噪声。当前仅单seed开发，不据此报告三seed均值/标准差，
也不混入原等预算Top-k消融表；不按TEST是否超过49/24选方、停止或追加预算。

## N49/k49训练扰动正则对照（2026-10-27）

分阶段优化与筛选蒸馏均在3轮后未得到VAL收益，训练NLL下降而验证NLL上升，
提示需要检验泛化正则，而不是继续同一局部小步优化。新profile
`configs/kather_n49_training_robustness_20261027.json`复用`fullk_refine.py`，
run `runs/simulation/kather_n49_training_robustness_20261027`，只在GPU3最多两路。
两项都从原120轮同seed的VAL best EMA初始化live/EMA，重置AdamW，推理保持原49/k49。

- `color`：保留原D4增强，额外在TRAIN对RGB三个通道分别应用0.9–1.1的随机指数，
  等价于逐通道光密度缩放；保持0/1边界及白背景不变，不声称进行了H&E染色分离。
- `phase_color`：相同颜色增强，另启用模型已有的TRAIN相位噪声，标准差0.02 rad；
  每个特征层相位调制前采样扰动。评估时`model.eval()`关闭噪声，VAL/TEST均使用原干净输入。

两项均从首轮联合优化，专家LR1e-4、global0.5倍、router0.1倍，4轮余弦到各自0.2倍；
batch16/micro2、原分类/margin/capture/TV/soft balance不变，关闭教师。
与前一组比较时，应将优化器设置和增强变化一并记入开发配置，不能单独归因为颜色或噪声。
每项最多4轮、连续3轮accuracy与NLL均不刷新最佳则早停；源best始终参与验证选模。
先seed17两项验证选方，仅有至少4/752张VAL改善再rollout至seed27/37各最多4轮。
该profile最多16个epoch-job，加前一组实际6个，本轮方法开发累计最多22个，不超过24个上限；
此前降温试验成本另计并完整保留。TEST只对锁定的选定方法进行，源模型胜出则核验后复用旧TEST。
这是改变训练增强的额外预算开发结果，推理未增加教师、电子分类头、参数或测试增强。

## N49/k49训练方法对照（2026-10-27）

用户要求主动改进训练方法。两路渐进降温在追加第4轮（逻辑124）仍未超过初始VAL78.32%，
较激进一路为76.46%，另一为77.66%；已保留全部best/last/history并结束，未测TEST。
收据为该run的`stopped_for_method_redesign.json`。这说明本次降温未提供收益证据，
不能把入口功率更不均匀本身当作改进。

在固定8张TRAIN图像（每类一张、教师学生同一增强视图）上分解分类、capture、TV、balance梯度：
TV及balance远小于分类梯度；未发现它们压倒分类的证据。无筛选的蒸馏在部分学生已学好的样本上
出现很大的相反方向梯度，因此采用有正确性及置信度筛选的蒸馏。该小样本梯度检查是机制诊断，
不是整个数据集的统计结论；原始收据`runs/smoke/kather_fullk_loss_gradient_20261027/receipt.json`保留。

新入口`fullk_refine.py`、profile `configs/kather_n49_fullk_refine_20261027.json`，
run `runs/simulation/kather_n49_fullk_refine_20261027`。两项均保持原`ScalingOptics` N49/k49
推理结构、路由温度0.05、原损失及数据/增强不变，从同seed累计120轮VAL best的EMA权重初始化
live与EMA，重置AdamW，不是精确续训。仅训练策略作成对比较：

- `staged`：前2轮冻结router，先适应专家及全局相位；以后解冻router。专家学习率5e-5，
  global为其0.5倍，router为其0.1倍；6轮余弦降至各自初值0.2倍，避免同时大幅扰动入口与光场。
- `distill`：完全相同的分阶段优化，另用同seed的90轮49/k24 VAL best作冻结教师，
  教师只处理相同增强的TRAIN图像。只有教师argmax正确且正确类概率高于学生时才计算该样本的KL；
  类别蒸馏温度2、权重0.3，按整个microbatch求均值。该温度不改变光学router。
  VAL/TEST推理只使用49/k49学生，教师、额外分类头和新增可训练参数均不参与推理。

先seed17两路并行，各最多6轮；若连续3轮VALaccuracy和NLL均未刷新历史最佳，提前停止并保留
源模型候选。每轮记录完整TRAIN/VAL、逐组梯度、两项损失、蒸馏样本比例和耗时；每分钟另存
optimizer batch进度。按VALaccuracy选best、NLL平局。两个方案均结束后按VAL选择，再锁定并TEST；
若选中源模型且推理/张量完全一致，则复用已核验的旧TEST，另存收据，不重复forward。
仅较源seed17 VAL多对至少4/752张时，将同一方法用于seed27/37各最多6轮，总预算最多24个
新epoch-job。后续各项选模完成即TEST，不按TEST挑方法或追加预算。教师及学生来源均按SHA锁定。
新结果是已观察旧TEST后的额外预算方法开发，不能直接替换原等预算Top-k消融。
仅GPU3最多两项计算，结束核子进程与显存释放；此前60/90/120结果全部保留。

本组两项均已按规则在追加第3轮停止，source初始VAL78.32%/NLL0.621844，
末轮staged的TRAIN89.90%/VAL78.19%/VAL NLL0.622740，
distill的TRAIN89.90%/VAL78.19%/VAL NLL0.623143。
各组解冻router后梯度均有限非零；蒸馏覆盖约25%–26%的训练样本，辅助损失约为主损失2.3%，
因此不能将无收益简单解释为无梯度或蒸馏完全未启用。
二者验证best都保留原模型（真实来源选定E117），未推广另外两个seed。
按预定VAL规则选方后，逐张量确认与原模型一致，复用原600/752=79.79%的TEST，未重复forward；
未选中的distill未读取TEST。31份JSON/路由NPZ/预测已下载、SHA和逐样本结果独立核验，
本组全部进程退出、GPU3分配释放；后续扰动正则另建run记录。
完整摘要见`reports/kather_n49_fullk_refine_20261027/final_summary_20261027.json`，
没有获得新的TEST提升，也不报告单seed标准差。

## N49/k49路由温度微调试验（2026-10-27）

用户在120轮两个已完成seed的TEST之后再次授权GPU3微调。原120轮及seed37训练保持不变，
新试验只修改T10新增入口`route_temperature_tune.py`，不热改正在使用的model/train/diagnose/evaluator。
观察到seed17/27选定权重的验证样本平均有效专家数约47.69/47.76，功率最高24路合计
仅占54.32%/54.27%；全49路入口功率较均匀。这是温度微调的假设依据，不是已证实的性能瓶颈。

直接降温的初始验证准确率降至33.91%/27.26%，旧试验已保全并停止，未将失败试验冒充完成；
收据见旧run的`stopped_for_initial_distribution_shift.json`。据此改为逐步降温，
不热改活动进程，旧源码身份保留在Git `b79f0d8c7`。
固定profile：`configs/kather_n49_temperature_anneal_20261027.json`。保持N=k=49、相位参数形状、
RGB输入/OEO/数据划分与增强不变；原路由softmax温度0.05，分别在12轮内线性降至0.02和0.01，
通过softmax(log(p)*0.05/T)等价重标定入口功率，场振幅取其平方根。49个专家始终保留，
不使用TTA、电子分类头或新增可训练参数；温度改变了路由前向合同，应作为单独开发配置。

先用seed17从同一120轮验证best的EMA权重做两项各12轮试验，GPU3最多两路。
新AdamW优化器，live及EMA均从选定EMA初始化；不是精确续训，旧Adam动量不沿用。
初始模型及第一轮保持T=0.05，先核对初始VAL复现原结果（容许最多1/752浮点边界差异），
后续每轮缓慢改变输入功率；best checkpoint保存该时刻的实际温度，推理按选中温度加载。
选方固定的是目标温度与12轮降温流程；各seed仍按自己的VAL选择权重及其配套温度。
lr=2e-4余弦至2e-5，batch16/micro2，原label smoothing0.02、phase TV0.02、
soft balance0.01、capture0.005及margin权重0.2不变。逻辑epoch标121–132，
实际来源权重选定epoch另记，不能把来源E117冒充完整连续132轮优化轨迹。

两档按验证accuracy选方案，NLL打破平局，在读取新TEST前保存recipe_selection。
仅当较原seed17验证结果至少多判对4/752张（0.53个百分点）时，固定该方案再微调seed27/37，
每项仍12轮；seed37来源须完成原120轮并锁定best SHA。总优化预算最多48个epoch-job，
不根据TEST是否超过49/k24改变预算、挑seed或继续追加。仅验证选定的温度方案评TEST，
未选中的候选保留完整验证记录。若验证门槛未达到，不启动额外两seed。

每项保存best/last、初始VAL、训练/验证曲线、梯度和selected train/val逐样本路由，
完成项立即评TEST。评估使用温度模型factory和源码/数据/权重身份，不调用会忽略温度的旧模型加载器。
新run为`runs/simulation/kather_n49_temperature_anneal_20261027`；旧直接降温run
`kather_n49_temperature_pilot_20261027`保持原样。仅GPU3，结束核自己的进程及显存释放。
这是观察旧TEST后的超参开发，不与原Top-k等预算表混用，也不保证超过49/k24。

## N49/k49从90轮追加至120轮（2026-10-27）

用户授权仅GPU3继续优化N49/k49，以超过前面的结果为开发目标；不保证达标，
不把TEST是否超过其他k作为选模、终止或追加预算的条件。原60轮及90轮结果保持不变。
上一轮N49/k49三个seed最后阶段验证NLL仍下降，但验证accuracy已趋于平台：
seed17/27/37第80至90轮VAL分别76.60→76.86%、78.32→78.19%、77.53→77.53%。
因此先采用一次有限的学习率重启，不能仅凭现有曲线预期一定再提升2个百分点。

配置`configs/kather_n49k49_continue90to120_20261027.json`；三个seed17/27/37均从
对应90轮last追加30轮至120，lr=3e-4余弦降至3e-5。保留live模型、Adam动量和EMA，
架构、数据划分/增强、batch16/microbatch2、margin权重0.2、capture0.005和原soft均衡不变。
随机状态按原seed初始化，epoch依赖的采样沿累计epoch计算；不是恢复全部RNG的位级续训。
按验证accuracy选best、NLL打破平局，包含90轮best作为候选；按用户最新要求，
每项完成验证选模并锁定后立即TEST，不等待其他seed。
这是额外预算、观察过原TEST后的开发续训，不能作为新的独立确认性结果或等预算Top-k消融。

入口复用`expert_tail_schedule.py`，只扩展来源身份审计以接受配置明确声明、锁定best/last SHA
且history达到90轮的来源；未声明时仍要求原60轮，原历史配置的行为不变。
GPU3 UUID固定`GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd`，最多两项并行，禁止其他GPU；
调度结束或异常时仅回收本任务子进程并写release_check。新run为
`runs/simulation/kather_n49k49_continue90to120_20261027`，不得复写上一轮目录。
用户随后追加授权GPU1（UUID `GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d`，
PCI `1B:00.0`）启动待跑seed37。沿用`expert_tail_expand.py`接管同一个run的队列，
GPU3的seed17/27保留原PID、权重、优化器、EMA和进度；seed37在GPU1从原90轮来源启动。
不修改原profile文件，资源授权变更通过handoff snapshot/ack另记；总训练仍为三个seed各30轮。
现控制器身份以`status.json.scheduler_pid`及`handoff_ack.json`为准，`plan.json.pid`为旧父进程。
训练源固定`d7270f1de`，控制器单独记录发布commit，后续监测不把旧父进程退出误判为故障。
两卡每卡最多两项本任务计算，只启动尚未运行的原待跑项；所有seed训练参数保持一致。
用户随后要求完成项即时测试：seed17/27已在GPU3顺序评估，未暂停GPU1的seed37。
最终控制器仍保存三模型总锁；通过checkpoint/data/model/evaluator/预测SHA校验后
复用已完成TEST，另存reuse_receipt，不重复forward或覆盖首次收据。预算与选模规则不变。

三个seed均已完成累计120轮及752样本TEST，逐文件SHA、accuracy、混淆矩阵、
分数argmax和跨run样本身份均已独立核验。下表对应各自验证选定的同一权重。

| Seed | 选定epoch | Train(%) | Validation(%) | TEST(%) | 同seed 90轮49/49 TEST(%) | 同seed 90轮49/24 TEST(%) |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 117 | 89.85 | 78.32 | 79.79 | 79.12 | 80.32 |
| 27 | 109 | 89.59 | 79.65 | 80.19 | 78.86 | 81.12 |
| 37 | 119 | 90.27 | 79.39 | 77.93 | 78.46 | 81.25 |

| 配置 | Train(%) | Validation(%) | TEST(%) |
|---|---:|---:|---:|
| 49/49，90轮 | 87.69 ± 0.13 | 77.70 ± 0.67 | 78.81 ± 0.33 |
| 49/49，120轮 | 89.90 ± 0.35 | 79.12 ± 0.70 | 79.30 ± 1.21 |
| 49/24，90轮 | 90.06 ± 1.68 | 80.36 ± 0.55 | 80.90 ± 0.50 |

均为三个seed的均值±样本标准差。追加预算使49/49的TEST均值提高0.49pp，
仍比90轮49/24低1.60pp；seed37自身TEST下降0.53pp，不能用两个改善seed替代完整统计。
训练与验证仍相差约10.78pp，未证明继续堆叠轮数可以消除泛化落差。
最终机器可读结果见`reports/kather_n49k49_continue90to120_20261027/final_summary_20261027.json`；
之前的n=2快照`partial_test_20261027.json`保留为历史，不能作为最终三seed表。
67份JSON/路由NPZ/逐样本预测已下载，按`final_artifact_manifest_20261027.json`逐份校验SHA，
独立复算三个seed分别600/603/586个正确TEST样本；best/last权重SHA已在服务器核验。
seed17/27既有TEST经身份核验复用，未重复forward；seed37由原控制器完成首次TEST。
`release_check.json`及额外身份收据确认本120轮任务的训练/评估子进程均退出，GPU1已无本任务分配。
GPU3后续方法开发与原120轮结果分开登记；原120轮数据、权重和结果完整保留。

## 五组尾部微调最终结果（2026-10-27）

15项全部完成：五个配置各seed17/27/37，从原60轮last追加30轮，累计90轮。
按验证accuracy选best、NLL打破平局并包含原best候选；每项锁定后评TEST。
各TEST含752张；逐样本预测、混淆矩阵、分数argmax、样本身份及传输SHA已独立核验。
下表为三seed均值 ± 样本标准差，单位%；Train/Validation/TEST对应同一选中权重。

| N/k | Train | Validation | TEST | 原60轮TEST | 提升(pp) |
|---|---:|---:|---:|---:|---:|
| 4/4 | 78.52 ± 0.48 | 73.85 ± 0.43 | 71.90 ± 0.28 | 69.73 ± 0.28 | +2.17 |
| 16/16 | 82.61 ± 0.60 | 75.66 ± 1.09 | 75.53 ± 0.46 | 73.89 ± 0.63 | +1.64 |
| 25/25 | 86.92 ± 0.69 | 76.99 ± 0.35 | 78.50 ± 0.89 | 76.46 ± 1.38 | +2.04 |
| 49/24 | 90.06 ± 1.68 | 80.36 ± 0.55 | 80.90 ± 0.50 | 80.41 ± 0.38 | +0.49 |
| 49/49 | 87.69 ± 0.13 | 77.70 ± 0.67 | 78.81 ± 0.33 | 76.64 ± 0.31 | +2.17 |

| N/k | Seed | 选中epoch | Train(%) | Validation(%) | TEST(%) |
|---|---:|---:|---:|---:|---:|
| 4/4 | 17 | 76 | 78.89 | 73.54 | 72.21 |
| 4/4 | 27 | 82 | 77.97 | 73.67 | 71.81 |
| 4/4 | 37 | 82 | 78.69 | 74.34 | 71.68 |
| 16/16 | 17 | 81 | 83.12 | 75.93 | 76.06 |
| 16/16 | 27 | 75 | 81.95 | 76.60 | 75.27 |
| 16/16 | 37 | 81 | 82.75 | 74.47 | 75.27 |
| 25/25 | 17 | 79 | 86.16 | 77.39 | 79.52 |
| 25/25 | 27 | 89 | 87.50 | 76.86 | 78.06 |
| 25/25 | 37 | 90 | 87.10 | 76.73 | 77.93 |
| 49/24 | 17 | 61 | 88.19 | 80.98 | 80.32 |
| 49/24 | 27 | 79 | 91.45 | 80.19 | 81.12 |
| 49/24 | 37 | 71 | 90.53 | 79.92 | 81.25 |
| 49/49 | 17 | 84 | 87.56 | 76.99 | 79.12 |
| 49/49 | 27 | 83 | 87.70 | 78.32 | 78.86 |
| 49/49 | 37 | 87 | 87.81 | 77.79 | 78.46 |

五组TEST均值均有提高，但没有得到随k增大必然提高的结果。N25/k25为78.50%，
仍低于原60轮k12的80.23%；同为90轮的N49/k49为78.81%，低于k24的80.90%。
追加训练缓解了部分优化不足，不能据此断言已解决过拟合或证明单调关系。
本轮是在观察原TEST后的开发跟进，且仅尾部配置增加预算，必须与原60轮等预算表分开呈现。
没有根据TEST挑选seed、继续追加训练或替换验证选中的权重，D2NN及原表完整保留。

15项30轮history中的loss和梯度均有限。完整训练/验证曲线、selected train/val逐样本
hard/soft路由及TEST预测保留在原run；全k时hard占比全1由定义决定，不能证明均衡。
N49/k24三个seed在TEST分别出现654/662/651种专家组合，选择随样本变化。
15份best SHA各不相同，best/last共30份权重继续保留服务器，身份见
`final_checkpoint_manifest_20261027.json`。

最终机器可读结果：[final_summary_20261027.json](reports/expert_tail_followup_20261026/final_summary_20261027.json)。
含15项逐seed指标、五组Train/Validation/TEST均值和样本标准差、专家负载及权重/预测SHA。
此前11项阶段快照仍保留在`partial_test_20261027.json`，不作为最终表。
本地已下载267份JSON/路由NPZ/逐样本预测并通过`final_artifacts_manifest_20261027.json`；
数据传输包另有SHA，PT不进入Git或本地传输包。

控制器结束后核验自己的所有训练/评估子进程退出，GPU1/3无本任务显存占用，
收据为run下`release_check.json`及`final_gpu_release_audit_20261027.json`；未操作其他任务。
原run为`runs/simulation/kather_expert_tail_followup_20261026`。

完成项即时TEST与最终复用：`evaluate_test.py --cache-locked-result`对同一权重、
数据、模型和评估代码校验身份后复用现有逐样本结果，另存reuse_receipt，
不重复forward或覆盖首次TEST收据。最终调度复用了14份已测结果，首次评估剩余N25/k25 seed37。
四次短暂停自己的训练均已恢复，之后两批使用GPU3空闲计算槽位无需暂停；
收据`early_test_execution_20261027_001.json`至`006.json`保留。
这替代下文早期“全部15项锁定后才统一TEST”的时间安排，训练参数及预算未改。

## 五组尾部微调扩展至GPU1/3（2026-10-26）

用户追加授权GPU1，最多两项并行；GPU3原两项继续，合计最多四项。
新调度`expert_tail_expand.py`接管同一个run：先冻结并核验原父调度器，
保存完整队列快照和来源身份，然后仅结束旧父调度器，收养原训练PID。
原训练进程不重启、不重新载入权重、不更改学习率或剩余轮数，禁止重复启动同一输出。
新增4项纯CPU检查覆盖15项任务无重无漏、来源/输出替换拒绝及训练预算/TEST延后不变。
原训练源保持`cecdf7091`且不热改，新增控制器单独记录发布commit和交接收据。
两卡各最多两项，启动前空闲显存阈值6000MiB；只操作本任务进程，最后释放两卡。
仍为原15项、每项追加30轮、全部模型锁定后统一TEST，科学设置完全不变。

## Kather专家数量实验：五组尾部追加微调（2026-10-26）

用户指定GPU3，允许与已有任务共卡；只处理(N,k)=(4,4)、(16,16)、(25,25)、
(49,24)、(49,49)，每组seed17/27/37，共15项。从截图所对应原60轮run的last继续，
不复用此前N4累计120轮结果，以保证这15项微调追加预算相同。
原run、D2NN和其他k均保留。新结果作为额外训练的开发结果单列，不能混入原60轮表
冒称等预算消融，也不能预先保证单调性或以TEST是否达到单调作为停止条件。

配置`configs/expert_tail_followup_20261026.json`，入口`expert_tail_schedule.py`：
每项追加30轮，累计90轮，Adam状态和EMA沿用原last；学习率5e-4余弦降至5e-5，
分类log概率margin=0.2、权重0.2，capture权重0.005；其余结构、划分、增强、
batch16/microbatch2和原soft均衡约束不变。按验证accuracy选best，NLL打破平局，
并包含原best作为候选。全部15项模型锁定后才统一评估TEST。
保存训练/验证曲线、hard/soft专家占比及逐样本路由，最终汇总三seed均值和样本标准差。

GPU3最多两项并行，启动前至少保留6000MiB空闲空间；不停止其他任务。
完成或异常清理仅处理本调度器自己的子进程，写释放收据。
run为`runs/simulation/kather_expert_tail_followup_20261026`。
服务器主目录有其他任务未提交修改，本轮不整仓快进；T10文件只从已发布Git对象
逐路径恢复并核验字节一致，训练记录发布代码commit与host HEAD两种身份。
整仓HEAD一致性仍须由相关任务完成主线收敛，不能声称本轮已统一整个服务器工作区。

## OrganCMNIST单seed统一50轮结果（2026-10-23）

用户将本轮上限改为50轮；四组实际均完成50轮，按前50轮内验证accuracy选择best，
验证NLL打破平局，全部锁定后评估官方TEST 8216张。保留原80轮余弦学习率曲线，
本结果是80轮计划在第50轮截停，不是重新采用50轮余弦曲线。
数据划分、方向保留增强及其余配置沿用下节profile；seed固定17，无重复实验标准差。

| Top-k | 完成轮数 | 选定epoch | Train(%) | Validation(%) | TEST(%) |
|---|---:|---:|---:|---:|---:|
| 1 | 50 | 49 | 89.79 | 90.26 | 75.97 |
| 2 | 50 | 40 | 95.36 | 93.56 | 80.55 |
| 3 | 50 | 48 | 97.39 | 94.57 | 82.35 |
| 4 | 50 | 48 | 98.12 | 95.11 | 82.64 |

三列均使用各组同一份best权重，Train评估关闭增强和训练噪声。k4超过72%开发目标，
但VAL与TEST仍相差12.47个百分点，不能声称完全解决跨划分泛化问题。
本轮增强、权重及正则同时改变，不能把提升全部归因于某一项。
这是查看旧TEST后的开发重训结果，不冒称新的独立确认性验证。

TEST每个专家被选中的样本占比（四项之和为k）：
k1=[24.38,27.69,26.68,21.25]%；k2=[70.46,10.00,19.54,100.00]%；
k3=[97.61,25.96,76.42,100.00]%；k4=[100,100,100,100]%。
k2/k3负载仍不均匀，k4全部选中由定义决定，不能据此声称路由均衡。

机器可读汇总：[summary_50.json](reports/organ_orientation_regularized_s17_20261023/summary_50.json)。
run仍为`runs/simulation/organ_orientation_regularized_s17_20261023`，原metadata/profile中80轮
计划不覆盖，截停变更见`cap50_control.json`及`summary_50.json`。
服务器保留50轮history、best/last、逐样本TEST预测和选模锁；本地同步history、预测、
指标和锁，权重仍留服务器。结果与预测已独立复算，传输清单含SHA256。
四个训练进程均已退出，GPU1/3已释放，见`release_check.json`。

## OrganCMNIST单seed方向保留重训（2026-10-23）

用户目标：只做seed17，统一重训k1/2/3/4，k4 TEST以72%为开发目标，
不更换数据划分、不按TEST调参或选模型。旧结果继续保留；72%不是已达成结果。
原120轮均值TRAIN79.53%、VAL77.72%、TEST63.21%，不能只凭该落差断言
经典训练过拟合已被证实；同时存在方向增强不合适及跨split泛化问题。

统一profile：`configs/organ_orientation_regularized_s17.json`。
四组均从随机初始化训练，保持原224专家/478有效面、17um计算网格、8um设备映射、
四特征层、逐层OEO、灰度重复三通道后沿用原编码及完整11类，原train/val/test身份不变。
禁用水平/垂直翻转和90度旋转；使用±8度旋转、±4%平移、0.9–1.1缩放、
灰度对比度/亮度/gamma扰动。标签含左右股骨/肾脏/肺，方向破坏有风险；
本轮不声称仅凭代码阅读就已证明这是TEST落差唯一原因。

标签平滑0.05、capture权重0.005、相位TV0.03、soft均衡0.01、训练期相位噪声
标准差0.02rad、EMA0.95；相位噪声在eval严格关闭，不改变checkpoint参数形状。
类别权重采用训练频数逆平方根并归一化，使训练样本平均权重为1。
microbatch=effective batch=16，soft均衡因此在16样本上计算，四个k完全一致。
学习率0.002余弦衰减，最多80轮，至少40轮后允许15轮无验证改善早停。
按验证accuracy选best，验证NLL打破平局；每5轮记录完整训练指标。
这里只改变训练及增强设置，没有增加可训练电子旁路或CNN。

入口`organ_retrain_schedule.py`，新run为
`runs/simulation/organ_orientation_regularized_s17_20261023`。
最多三张空闲卡、每卡最多两项，跳过GPU6；先完成四项并锁定全部checkpoint，再评TEST。
保存原始命令/profile SHA/数据SHA/Git身份、best/last、逐样本预测、
专家hard负载和soft概率及释放收据。最终仅报告一个seed，无重复实验标准差。
这是已观察TEST后的开发重训，不冒称新的独立确认性验证。

## k4累计120轮三seed最终结果（2026-10-23）

Kather2016与OrganCMNIST各seed17/27/37全部完成；实际末轮均为120，
模型按验证accuracy选取，NLL打破平局，随后锁定checkpoint评估TEST。
所有6份best文件实物SHA与训练、TEST报告逐一一致；均值和样本标准差已独立重算。
本任务所有子进程已退出，释放收据确认正常结束，没有继续追加训练。

| 数据集 | 训练准确率(%) | 验证准确率(%) | TEST准确率(%) |
|---|---:|---:|---:|
| Kather2016，k4 | 78.97 ± 0.36 | 74.16 ± 0.20 | 71.90 ± 0.94 |
| OrganCMNIST，k4 | 79.53 ± 0.30 | 77.72 ± 0.55 | 63.21 ± 0.58 |

| 数据集 | Seed | Train(%) | Validation(%) | TEST(%) | 验证选择epoch |
|---|---:|---:|---:|---:|---:|
| Kather2016 | 17 | 79.15 | 73.94 | 72.74 | 106 |
| Kather2016 | 27 | 78.55 | 74.20 | 70.88 | 107 |
| Kather2016 | 37 | 79.20 | 74.34 | 72.07 | 115 |
| OrganCMNIST | 17 | 79.67 | 78.09 | 63.44 | 106 |
| OrganCMNIST | 27 | 79.19 | 77.97 | 62.55 | 107 |
| OrganCMNIST | 37 | 79.74 | 77.09 | 63.63 | 113 |

只统计本次累计120轮的k4，不把未补齐同预算的其他k列写成120轮结果。
每次累计120=原训练60+原目标续训30+margin微调30；选定epoch低于120正常，
不是提前终止。仍属已观察TEST后调整训练目标的开发结果，不是新的独立确认性比较。
Organ验证与TEST约14.51个百分点的落差仍存在，不能称已解决其泛化问题。
机器可读统计及逐seed身份见
[`k4_120_summary.json`](reports/topk_diagnosis_20261023/k4_120_summary.json)。
原结果、best/last、逐样本TEST预测保留，未覆盖或删除。

## 2026-10-23：两任务k4三seed微调重复

Kather seed17累计120轮已完成，验证accuracy选定epoch106：
TRAIN79.15%、VAL73.94%、TEST72.74%，同seed固定k3为77.20/74.34/71.28%。
这只是一seed开发结果，不能替换旧表三seed的69.73±0.28%。

用户要求第一/第三行k4重复三seed，统一seed17/27/37、累计120轮：
原60轮→原目标extra30（lr2e-4→2e-5，验证NLL选模）→margin30
（lr5e-4→5e-5，margin0.2、权重0.2，capture权重0.005，验证accuracy选模）。
复用Kather17的两阶段及Organ17的第一阶段，剩余9个30轮阶段需要训练。
按最新授权三张GPU分别固定seed17/27/37，同卡两个数据集并行运行
`k4_repeat_schedule.py`；每任务只补剩余60轮，内部30+30阶段按源结果依赖调度，
已有同身份结果复用。run为
`runs/simulation/k4_three_seed_20261023`，每阶段退出后释放，整个队列完成/异常有释放收据。
结果自动汇总到run下`table_summary.json`，不足2seed时标准差记null，不能填0。
最新统计只包含两任务k4累计120轮的逐seed、训练/验证/TEST均值和标准差，不再输出60轮表。
这是基于已查看TEST的开发微调重复，不构成统一预算的严格top-k因果消融；
不以TEST选择权重、seed或训练停止点。

## 2026-10-23：k4分类准确率微调（单次有界开发试验）

Kather N4k4 seed17从extra30的last90出发，额外30轮、lr5e-4降至5e-5，
原CE/soft均衡/TV保持，capture权重0.02降到0.005，追加0.2倍分类margin项：
`max(0, 0.2 + max_wrong(log p) - log p_true)`。目标是检验偏重正确类别排序能否
改善accuracy，同时保留capture指标观察光学功率代价。光学结构、参数量不变。
使用验证accuracy选best，以验证NLL打破平局，候选包含原best。
不承诺k4必胜；只做这一次30轮试验，不在TEST上反复搜索或以TEST触发停止。
运行GPU0，产物为`runs/simulation/kather_k4_margin_20261023`。
比较对象为同seed17无新增负载均衡的k3（TEST71.28%、VAL74.34%），不把三seed均值混入。
涉及已观察TEST后的开发诊断，不能冒称新的独立确认性结果。

## 2026-10-23：Kather k3入口路由负载约束诊断

旧soft均衡权重0.01仍保留：原项约束microbatch2的soft概率均值，不能保证
hard top-k实际使用均衡。无额外约束续训后的Kather k3在验证集始终选择同一组三专家，
平均soft概率约[0.202,0.138,0.271,0.389]，故不能宣称已学会动态专家分工。

新增可选`--hard-balance-weight`：定义`f_e=mean_batch(mask_e)`、
`p_e=mean_batch(probability_e)`，附加`lambda*N/k*sum(stopgrad(f_e)*p_e)`。
这是使用实际选择频率的可导代理，不是对离散mask直接求导；均匀使用时基线为1。
每次optimizer step单独重算同一增强图像的router，以完整effective batch16统计，
不会把16样本全光学图保留在显存。原分类/捕获/TV/soft均衡项保持。

预先固定两个权重0.1/1.0，仅Kather N4k3 seed17，从原last60继续30轮，
与既有extra30相同数据顺序、增强、学习率和优化器/EMA状态配对。
不根据TEST选择权重；报告路由是否恢复变化及验证性能权衡，不预设均衡提高准确率。
保留每轮hard负载/soft概率/选择集合数、辅助项及分类router梯度范数，
末轮与验证选定best均保存逐样本路由记录和类别分配。
若原best仍胜出，明确报告选定模型仍可能坍塌，不能以last路由替代best的性能。
新run为`runs/simulation/kather_k3_load_balance_20261023`，
由`diagnose_topk_schedule --balance-sweep`在最多两张空闲GPU运行，跳过GPU6。
这些属于已查看TEST后的开发诊断。

## 2026-10-23：固定4专家 Top-k 收敛诊断

针对已有固定478三任务表，重新核对36份训练/history/TEST报告。均为60轮；
原选模使用验证集macro-NLL最低的EMA模型，不是验证或TEST准确率最高的轮次。
三seed的Kather k3/k4训练准确率均值为76.33/75.65%，验证73.49/71.81%，
TEST71.01/69.73%；最佳轮次分别60/59/58与60/59/59，支持进一步检查优化预算，
不足以诊断严重过拟合。Organ k3/k4训练74.35/75.09%，验证76.51/76.39%，
TEST61.85/61.38%；验证与测试的较大落差不能归因于k本身。
原history只保存训练目标及验证指标，没有逐轮训练准确率。

当前路由恰好选择k个端口并重新归一化总输入功率；相干整场传播及逐层OEO
会改变多专家组合，固定光学router也未保证可将额外专家概率任意逼近零。
因此不能假定k4模型函数集合严格包含k3，也不能预设准确率必须单调。
均衡项实际在每个microbatch=2上计算，再累积成16样本的优化batch，
不是直接对16样本聚合后计算；本次为保留对比身份，不修改此损失。

诊断入口`diagnose_topk.py`从原last模型、Adam状态和EMA续训，原best参与验证选模，
新目录保存best/last及身份，不改动旧结果。seed17下Kather/Organ的k3和k4
各追加30轮，统一学习率2e-4余弦下降至2e-5；几何、增强、损失及EMA不变。
逐轮记录验证指标，每5轮记录完整训练指标，并在续训前对比live/EMA。
仅在锁定验证macro-NLL最优checkpoint后执行一次TEST；由于问题来自已查看
的TEST结果，新增结果标记为开发诊断，不能宣称新的独立确认性性能。
`diagnose_topk_schedule.py`最多占两张空闲卡、跳过GPU6，子进程退出后写释放记录。
新run位于主工程`tasks/t10_expert_scaling/runs/simulation/topk_diagnosis_20261023`。
历史摘要与收敛图保存在`reports/topk_diagnosis_20261023/`。

## 当前入口：2026-10-03工程整理

请先读[实际源码、四组结果及保留边界](CURRENT_VERSION_20261003.md)。
实际运行工程为 `t10_corrected_20260918`，不是下文已过期的校准PID或GPU队列。
原HEAD之外9份实际运行源码已另存Git恢复快照；现有结果、baseline和全部测速保留。
以下内容完整保留作历史协议，不代表今天正在训练，也不得直接照旧命令抢占GPU。

## 2026-09-18：校准完成，扫描运行中

12/12校准完成；按三架构验证macro-NLL的预定共同深度规则，选4层、学习率0.002。
相同4层的最佳checkpoint验证准确率：MoE 73.01%、总参数匹配D2NN 63.83%、等孔径D2NN 63.43%。
MoE对应训练准确率80.92%，探测窗捕获功率仅0.2725%；尚不能仅凭准确率判定光学实现成熟。
6层MoE验证75.00%不用于与4层D2NN混合比较。本轮未读取测试集指标。
证据：`kather_calibration_s17_17um_uuid_20260917`各run的`result.json`。

`kather_spaced_scan_gpu012_s17`已进入4专家k=1/2/3训练，后接k=4及每个N的三种D2NN对照，
总计39新run+3校准复用。运行GPU0/1/2，协调器878608；以下旧的校准运行中描述仅供追溯。
中期路由检查：N4k2两专家被选择比例约96%/99%，另两专家约3%/1%；
soft概率接近均衡不能证明hard top-k负载均衡，须随最终结果报告此项，不预设专家数增加必然提升。

输入审计澄清：当前`encode_rgb`先把各原始RGB通道插值到目标边长的一半，再拼[R,G;B,mean]。
MoE通道112²，D2NN通道S/2²；因此两者原始信息带宽并未严格统一，D2NN可能保留更多细节。
不应把当前协议称为“相同112²源信息”；后续匹配带宽对照需新profile，不能静默修改本轮结果。

路由数据保存：每个MoE结果的`result.json`和逐轮`history.json`均保存验证集的`route_load`（hard top-k选择比例）、
`route_probability`（soft概率均值）、`distinct_selected_sets`以及验证指标；`audit_routes.py`可将所有run导出为JSON/CSV，
并计算死专家数、最大/最小hard负载和soft概率熵。soft概率均衡不等于hard选择均衡，论文必须同时报告两者。
当前损失中的`router_balance_weight=0.01`只约束batch内soft概率均值，不能直接阻止hard专家坍塌；增加该权重可能降低分类准确率，
因此后续只对固定配置做无约束/当前约束/增强约束的成对消融，不覆盖已有结果。

OEO核查：当前实现为`intensity -> mean normalization -> spatial LayerNorm -> ReLU -> t/(1+t) -> unit-power`。
LayerNorm输出严格包含正负值，ReLU会把负半轴置零；这不是数值bug，但会造成半数左右的归一化响应没有梯度通过，需单独做激活消融。
图文/音文工程的`relu_softsign`实现是`softsign(ReLU(z))`，只把正半轴映射改成平滑有界函数，并没有恢复负半轴。
因此它不能被表述为“完全取消ReLU截断”。当前正式扫描仍保持原OEO以保证可比；后续新增profile比较原`ReLU+t/(1+t)`、`ReLU+Softsign`，
必要时再设计物理上非负的平滑替代，所有版本同时应用于MoE和D2NN。

架构冻结前的关键检查：MoE输入先编码为224×224单幅振幅；224×224路由相位经传播后读取49个24×24端口，
得到每个样本的soft概率，再做top-k选择和功率归一化，复制到专家阵列。每个特征循环依次执行专家相位+传播+OEO、
全局相位+传播+OEO；最后传播到8个分类探测窗。没有CNN、Qwen或电子残差分类支路。D2NN使用同一RGB编码函数但目标边长不同，
当前仍存在源信息带宽不完全一致的问题，必须在正式结论中单独标注。

10×10专家阵列本身没有版面限制，边长为2510逻辑像素。此前代码把路由器错误地固定为7×7=49个端口，
这是实现缺陷而非架构限制，已在最新代码中改为按N自动生成端口：N=100时为10×10、312×312路由面；N≤49仍保持224×224旧布局。
路由器相位参数、传播网格和D2NN参数匹配均随路由面更新，100专家仍需单独记录显存与耗时。

输入公平性修正：D2NN不应把224×224源场限制在大相位面的中央，否则外围可训练相位没有输入覆盖。
正式协议采用各架构的全孔径输入：MoE每个224×224专家接收同一224×224 RGB编码源并复制到阵列，D2NN将同一RGB图像编码后铺满自己的连续相位面。
两者都不额外使用电子特征或残差支路；公平性通过相同原图、波长、采样、传播距离、层数、训练预算和参数匹配控制。

确认性实验使用seed=17/27/37，至少覆盖Kather2016和DeepWeeds；N=100使用top-k=[1,10,25,50,75,100]，
其余N沿用既定稀疏—中间—密集取值。当前正在运行的旧seed17队列不修改，新的动态路由代码用于后续新profile。

DeepWeeds仍未训练。新增CPU候选审计入口：
`python -m LightGenV2.tasks.t10_expert_scaling.audit_deepweeds --data <deepweeds_fold0> --out <task/runs/smoke/audit_run>`。
输出跨split dHash候选、接近拍摄时间的分组统计与候选图；候选并不等于已确认同一植株。
审计不会自动把training_ready改为true，也不更改官方划分。

DeepWeeds全量CPU审计已完成：54对跨split dHash≤4候选；样例可见近乎相同的连续拍摄画面。
证据run：`runs/smoke/deepweeds_scene_audit_20260918`，源码c90bd143。
后续采用独立`deepweeds_date_camera_grouped_s17_v1`：按拍摄日期+相机分组，再合并54对候选涉及的组；
StratifiedGroupKFold(5, shuffle=True, random_state=17)，固定fold0测试、fold1验证、其余训练，
不搜索seed/fold，不用模型表现挑划分。组是场景代理而非真实植株身份，不声称完全生物学独立。
`prepare_deepweeds_grouped.py --data <official-cache> --audit <audit-run> --out <separate-cache>`
验证原缓存hash、标签、图像ID完整覆盖、组互斥、候选不再跨split、各类非空后生成新缓存及清单；
官方缓存保持training_ready=false，只有新分组缓存允许训练。

## 最新执行约定：GPU0/1/2、4专家全k、专家+global匹配

用户追加要求已执行。当前实际训练数据只有**Kather2016，5000张、8类组织图像**，
train3496/val752/test752；test仍不参与选模。DeepWeeds17509张9类只完成下载和精确去重，尚未训练。

本任务后续固定GPU0/1/2，不再排GPU6。GPU6原子进程863246已退出，未完成run保留在原校准目录
`interrupted_gpu6/`；0、1原进程849900、828346被新协调器接管，继续原训练，GPU2重跑被中断的一组。
协调器PID878607，扫描等待PID878608；实际进程以status.json为准。
新执行工作树`/DATA/DATA1/guest3/t10_gpu012_20260917`，源码`affa905f`。
新扫描`runs/simulation/kather_spaced_scan_gpu012_s17`，39个新run+3个校准复用，共42个比较点。
校准结束自动接续，GPU UUID绑定，完成后检查子进程退出和显存释放。

Top-k：4→[1,2,3,4]，9→[1,3,5,9]，16→[1,4,8,16]，25→[1,6,12,25]，
36→[1,9,18,36]，49→[1,12,24,49]。以下旧章节的32/33组、GPU6与k=1/2/4均属历史约定。

每个N增加`d2nn_expert_global`：匹配J×(N×224²+G²)，不计Router；
原`d2nn_total_parameter`匹配上述参数加224² Router，仍保留；`d2nn_same_aperture`为等孔径对照。
新增版本沿用同N的共同深度及总参数D2NN已选学习率，不额外使用test调参。
边长取最近可容纳偶数，实际相位参数匹配误差<0.5%，写入metadata.geometry。
CPU队列检查通过：39个唯一新配置，N4覆盖全部k，六个N均有新增D2NN版本，不启动额外GPU。

## 当前队列调整（2026-09-17）

校准仍在运行，已完成首组3/12；三张物理GPU 0/1/6均有训练进程。
原队列没有自动衔接扫描，现已补齐：独立工作树`/DATA/DATA1/guest3/t10_scan_20260917`，
源码`f025f467`，协调进程启动PID814553。它在CPU上等待校准完成及释放记录，
随后按validation macro-NLL选择共同深度/各架构学习率，并自动使用同三张卡。
扫描run为`runs/simulation/kather_spaced_scan_s17_20260917`，共32次新训练加3个校准复用结果。
CPU模拟队列检查已通过（唯一配置、选模、复用、无GPU启动）；不是新性能结果。

最新top-k取值替代下文原设计表及`reports/design`旧矩阵：
4→[1,2,4]；9→[1,3,5,9]；16→[1,4,8,16]；25→[1,6,12,25]；
36→[1,9,18,36]；49→[1,12,24,49]。
实际执行清单由`schedule.py --mode scan`生成，包含23个MoE点及12个D2NN点，
其中9专家dense MoE和两种D2NN复用校准已选结果。输入/相位尺寸、损失、60epoch与seed17不变。
所有子进程在单次训练结束退出，整个队列完成/异常时写释放检查。

DeepWeeds图像已经下载并核验存档hash，17509张256² RGB，官方划分无跨split精确重复；
场景/近重复审计尚未完成，`training_ready=false`，尚未启动该数据训练。
不能将Kather扫描队列描述为两个任务全部已经自动排完。

当前状态：**已通过CUDA结构/梯度检查，seed17的Kather验证校准正在三张GPU上训练。**
运行：`runs/simulation/kather_calibration_s17_17um_uuid_20260917`，训练源码`752d2995`。
GPU按物理UUID绑定0/1/6；最多三张，子进程逐run退出，队列结束/异常时检查释放。
当前12个配置用于N=9的共同深度与学习率校准，不是完整专家数矩阵的最终成绩。
DeepWeeds仍在数据获取/审计阶段；原始PBC未启动。详细证据见[复现入口](reports/reproduction/README.md)。
本轮研究仿真性能，不以 SLM 版面限制 N。用户最新指定输入逻辑像素17 μm、相位设备8 μm、专家间隔30逻辑px；
专家采用正式 T01/T02/T08 的 **224×224 px**，不沿用病理 demo 的 146×146。
尺寸依据：T08 `reports/optical_router_moe_20260907/main/config.yaml` 的
`expert_size=224, expert_pitch=254`。仿真计算网格为17 μm；8 μm相位SLM通过保持物理尺寸的最近邻映射承载权重，
依据同配置的 `hardware.phase_mask.mapping=physical_pitch_nearest`。224逻辑像素映射为476设备像素，
并非独立学习476²个权重。新协议ID为 `raw_rgb_fullfield_oeo_17um_phase8um_v2`，此前全8 μm设计未启动训练。

推荐两个主任务：Kather2016 组织分类和 DeepWeeds 杂草分类；原始 PBC 血细胞分类为第三项扩展。
三者的数据许可均核查为 CC BY 4.0，依据见 [数据登记](dataset/registry.json)。
它们是三个应用域的分类任务；联合训练用于研究跨域专家分工，不等于验证了分类、检索、回归等不同任务形式。

| 专家 N | 排布 | 单专家 | 有效面边长 G | 17 μm 对应边长 | Top-k |
|---:|---|---|---:|---:|---|
| 4 | 2×2 | 224×224 | 478 | 8.126 mm | 1、2、4 |
| 9 | 3×3 | 224×224 | 732 | 12.444 mm | 1、2、3、4、5、9 |
| 16 | 4×4 | 224×224 | 986 | 16.762 mm | 1、2、4、8、16 |
| 25 | 5×5 | 224×224 | 1240 | 21.080 mm | 1、2、4、7、13、25 |
| 36 | 6×6 | 224×224 | 1494 | 25.398 mm | 1、2、4、9、18、36 |
| 49 | 7×7 | 224×224 | 1748 | 29.716 mm | 1、2、4、13、25、49 |

`G = sqrt(N)×224 + (sqrt(N)−1)×30`。传播画布暂定 `G+40`，FFT 的数值补零另外计算；
这些尺寸都不作硬件可部署性承诺。固定 k 比较与固定 k/N 比较必须同时看。

## 架构和比较

**此前病理 MoE 有 global phase。** 六层是三个“专家层 + 全局层”循环，而不是每个专家六层后直接求平均。
专家输出为复光场；全局相位与传播使不同区域混合，最后由类别探测窗口的能量读出分类。
旧实现入口是理想图像复制分光；局部专家传播和全局传播并非同一计算域，不能直接等同于完整自由空间仿真。

本轮模型采用以下新仿真合同，正在验证前后向与数值传播：

```text
RGB → 固定空间编码 [B,1,224,224]
    → 光学 Router（224² phase → ASM → 固定区域积分 → Top-k）
    → 总入口功率为1的理想分光，投送到 N 个224²区域
    → [N块专家 phase → 整场 ASM → OEO
       → G² global phase → 整场 ASM → OEO] × 3
    → 整场传播 → 固定类别探测窗口 → 能量归一化 → 类别

D2NN+OEO：同一输入编码 → 覆盖其有效相位面的上采样
    → [dense phase → 整场 ASM → 同定义 OEO] × 6
    → 同一探测器和判决
```

默认两者**每个特征衍射层后都有 OEO**，无可训练电子旁路、Qwen 或 CNN。OEO 中有电子归一化和非线性，
不能写成全光无源系统。Router 额外有一次光学测量，也单独计入参数与成本。
六层是当前起始选择，不声称在224专家新合同下已证明最优；仅在固定 N=9 做小规模验证选深度，
不展开 N×k×层数大网格。

三组主对照：MoE+OEO、**总相位参数量匹配**的 D2NN+OEO、**有效孔径相同**的 D2NN+OEO。
另外固定 global 面为1748²，在 N=4/16/49 上补充对照，区分专家数量与全局层扩容的作用。
不沿用“只匹配激活专家参数”却声称总参数相等的口径。

## 实验顺序

1. 先完成数据分组/去重、数值传播收敛、张量覆盖、功率守恒、Top-1 路由梯度检查。
2. seed17，两个主任务先跑 N=4/9/16 的 k=1/4/N 及两种 D2NN，共 **28 个配置**；小规模校准另计。
3. 协议锁定后补齐 N=4–49、固定 k 和固定比例扫描，两个任务共 **88 个配置（包含前述28个）**。
   D2NN 不随 k 重复训练。完整矩阵一致时才可复用 pilot，否则重训。
4. 固定 global 的容量控制共18个配置；在 N=9/25 上进一步比较两任务联合训练与单任务训练，
   检查专家是否按任务/类别分工。路由消融、联合训练、噪声检查另计，详见方案。
5. 本轮初扫仍只有 seed17。稳定结论建议后续对预先选定的代表点补3个配对种子；当前未安排这些训练。

不是要求结果随 N 或 k 单调增加，也不预设 MoE 必须胜出。要回答的是：性能何时饱和、最佳稀疏度是否变化、
收益来自专家分工还是孔径/参数增加，以及大 N 是否出现训练不足或过拟合。

## 文件和入口

- [详细实验协议](docs/PROTOCOL.md)：输入、OEO、路由、损失、公平性与数据划分。
- [设计配置](configs/study.json)：机器可读设计参数；**不是已有训练入口可加载的配置**。
- [训练入口](train.py)读取设计配置并记录实际参数；[三卡队列](schedule.py)当前提供calibration阶段。
- [主实验矩阵](reports/design/main_matrix.csv)、[几何/参数表](reports/design/geometry.csv)。
- [复现入口](reports/reproduction/README.md)：本任务唯一证据入口，当前无新性能数值。

从仓库根目录运行（仅标准库，不使用GPU）：

```bash
python LightGenV2/tasks/t10_expert_scaling/plan.py
```

脚本校验专家尺寸/间隔、路由探测区域不重叠、k 合法、入口功率、D2NN 参数匹配误差，
并生成计划 CSV/JSON；不会自动启动实验。今后训练写入本任务 `runs/smoke/` 或 `runs/simulation/`。
