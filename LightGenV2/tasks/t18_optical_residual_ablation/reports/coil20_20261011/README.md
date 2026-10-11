# COIL20：快速物体任务与两层门槛

用户要求自然图像、类别>8、两层70%且更快。完整30类叶片有界预检30轮未过70%，
保存其负结果和历史，不扩展它的深层套件；只额外考察一次完整作者COIL20物体任务。
[作者发布](https://www1.cs.columbia.edu/CAVE/software/softlib/coil-20.php)的processed
archive实际20个已知实体×72个视角，1440张128灰度照片，全部原生图与物体保留。
作者公开研究下载，页面／archive未附明确CC许可，不冒称CC BY，不再发布原图。
它是已知物体在留出视角上的识别，不是20个多实例语义类别／未见物体个体泛化。
完整数据身份在[data_identity.json](data_identity.json)。

每相邻6视角（30°）成块，再合并精确原始／缓存像素重复，实际240连通组，
seed20261011按每物体组数70／15／15取整：train960／val120／test360，
各类48／6／18，真实图像比例66.7／8.3／25.0%。未拆块／挑图，也没有逐图随机拆邻近角度。
块边界近邻及采集依赖仍可能存在，评估范围是图像／留出视角。
作者原生去背景／裁边灰度图同值复制RGB、bicubic150和原固定RGB振幅编码；
无本项目新增CNN／特征／分割／挑容易类别。20个固定32×32 CCD窗三行8／8／4，
九专家原ASM／全场OEO，相干rho只expert/global、router无rho，无电子学习head。

## 两层30轮门槛已完成并CPU审计

run `coil20_L2_s17_b4_batched_pilot_20261011`，训练commit
`2b4c28f9319e8a1b52ff33a7147684f7de4e97e5`，67实际／发布来源匹配。
20:12:18启动，20:17:57两个GPU子均结束且实际释放，两臂各30轮，test始终0。
同seed17初始化、449848参数／11相位；micro4专家batch保留相干残差，
有效batch16加权含尾／辅助项，每macro一次clip／AdamW／EMA。两臂所有30轮
order／augmentation哈希相同，config唯rho不同。每轮只验证EMA balanced NLL最低
首次选best，两个best均第30轮，不按accuracy反选。train为同EMA全train未增强。

| rho | train准确率／宏召回 | val准确率／宏召回 | train NLL | val NLL | 平均秒／轮 | 总秒 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 94.79% | 84.17% | 0.60112 | 0.76773 | 10.91 | 330.79 |
| 0.3 | 94.27% | 85.00% | 0.80422 | 0.92882 | 10.61 | 321.89 |

两层速度及验证accuracy／macro≥70%门槛均通过，不能据此保证test≥70%。
残差验证accuracy高0.83pp，但NLL较差，不宣称所有指标优势／单种子稳定收益。
train／val差10.625／9.271pp，需要关注；两者验证NLL持续下降、best在最后一轮，
没有晚期损失恶化或很早best的证据，不能称严重过拟合，也不能声称已完全收敛。
数据集选择用了有界验证探索，不能宣称代表任意自然图像任务／独立实体泛化。
CPU只读保存结果／PT／history／CSV／缓存trainval标签核验全部通过，不重复光模型推理：
最低NLL首次、best／last／config／data／67来源、所有相位梯度、预测ID／标签／argmax／
support／CM／NLL／macro及配对身份。[审计](pilot_audit.json)、[指标CSV](pilot_summary.csv)、
[曲线CSV](curve_data.csv)、[PNG](learning_curves_L2.png)、[SVG](learning_curves_L2.svg)、
[PDF](learning_curves_L2.pdf)均为真实保存结果，单seed无误差条，没有test曲线。

## 历史：20:27深层数值／速度预检

[新数据预检](deeper_execution_probe.json)只用train8及新鲜seed17参数，无正式拟合／test。
4／8层micro4两rho过原门槛；16层micro4两rho未过（maxgrad L2 3.216%／5.889%，
rho.3 maxprob5.342e-6），保留失败不放宽。16层micro1专家batch通过，初始prob／grad差0。
该检查不是任意权重逐像素精确证明。短程加50%余量投影16层188–189秒略高180秒门槛，
不能当真实整轮，正在GPU4进行两轮端到端计时`coil20_L16_b1_full_epoch_timing_20261011`，
commit `29a542ace`，PID645299，20:27:35启动，只一个GPU子；此计时不是正式16层臂。
首轮含完整train／val172.22秒，普通轮／均摊还待实测，test0。控制器只增加depth／预算
参数化，光模型与优化算法不变；完成旧两层来源仍按metadata的原commit审计，不能HEAD冒充。
待真实速度门槛和预算核验后再展开2／4／8／16两rho；全八臂来源／数据／配置／方案／PT
锁定后获选每臂首次单次test，不能提前只给L2测test。原数据／原ZIP／PT／预测不进Git。
完整契约见[协议](../../COIL20_PILOT_PROTOCOL_20261011.md)。

## 已接续四深度有界比较（20:46:11启动）

16层两轮端到端计时已完成并释放GPU：首轮172.224秒（含全train），普通轮136.895秒；
按每5轮完整train评估均摊的投影143.961秒，最终额外选中train评估另计。
完整[计时收据](L16_throughput_receipt.json)保存两轮／来源／参数和实际退出证据，
不把计时两轮当正式16层臂，也不把380.18秒总时长直接除二当每轮时长。

CPU队列`coil20_s17_L2L4L8L16_val_fast_20261011`父654702，实际源码commit
`b8bbe7ccf`；CPU无CUDA，最多GPU1／2／4各一个子进程。复用已完成的两层两臂
30轮以省时，其它六臂从头seed17各完整30轮（--full-budget、每臂5400秒安全上限）。
4／8层生产micro4，16层micro1，专家batch均保留相干残差；同深度两rho相同模式／
有效batch16／样本加权尾及辅助项／一次clip Adam EMA。同层配置仅rho不同。
全部每轮val、EMA最低balanced NLL首次选best，未增强全train每5轮和端点。
ReduceLROnPlateau只看val调整lr，不用test选权重或调训练。
两层完成来源是旧2b版本，其它六臂使用只新增depth／预算参数化的29a控制器；
各metadata实际67文件及发布blob分别核验，不能冒称全八臂控制器来源完全相同。
光模型、编码、detector、原优化算法未改；每深度仅作单seed探索，不称纯深度因果。

20:52实际核验：GPU1 L16rho0 PID654708、GPU2 L16rho.3 PID654709、
GPU4 L8rho0 PID654710，全67实际／发布来源匹配，重新GPU smoke通过；
随后以current_processes.json核对PID，不重复启动。显存1170／1020／1338MiB，
各3GiB+800MiB。16层普通E2为143.55／140.20秒、8层普通约20秒，
首轮30轮训练暂估22:00前后结束，非最终test／拟合调整完成时间保证。
源／负载变化需实测修正。8层rho0 E15 val82.50%；16层各E2只是早期，
不根据chance附近的早期精度判容量或宣称残差最终收益。

## 已完成4／8层与16层拟合诊断（21:44）

4／8层各两rho完整30轮、验证最低NLL首次均为E30，test尚未读取。保存结果CPU核验通过：
最低轮／PT／config／各原commit发布blob／data／梯度及逐样本ID、标签、argmax、支持、CM、NLL、macro；
每对初始化／全部30轮order及augmentation一致，配置仅rho不同。GPU4四个fit子实际退出。

|层数|rho|train / val accuracy|val balanced NLL|总fit秒|
|---|---|---|---|---|
|4|0|98.4375% / 100.0000%|.39155049|397.091|
|4|.3|98.1250% / 94.1667%|.79273651|399.631|
|8|0|99.0625% / 97.5000%|.38328143|660.078|
|8|.3|97.7083% / 88.3333%|1.65757573|668.493|

真实保存数据：L4[审计](L4/pilot_audit.json)、[CSV](L4/pilot_summary.csv)、[曲线CSV](L4/curve_data.csv)、
[PNG](L4/learning_curves_L4.png)、[SVG](L4/learning_curves_L4.svg)、[PDF](L4/learning_curves_L4.pdf)；
L8[审计](L8/pilot_audit.json)、[CSV](L8/pilot_summary.csv)、[曲线CSV](L8/curve_data.csv)、
[PNG](L8/learning_curves_L8.png)、[SVG](L8/learning_curves_L8.svg)、[PDF](L8/learning_curves_L8.pdf)。
PNG视觉检查通过，单seed无误差条、没有test曲线。100%仅120个已知物体留出视角，不是未见实体泛化。
4／8各最后十轮验证NLL仍改善，存在未充分收敛可能，不能因高accuracy假称已收敛。

16层尚未完成：rho0 E22 val83.33%；rho.3 E23 val11.67%、E20 train8.96%，
长期低train／val是学习停滞风险，不能归因为过拟合或直接推断容量不足。
新鲜seed17、四个预指定train标签的[独立诊断](L16_train_diagnostic/diagnostic.json)
没有加载在训／完成PT、没有训练更新／val／test。[来源](L16_train_diagnostic/metadata.json)
与[完成收据](L16_train_diagnostic/status.json)保留，GPU4诊断子退出释放。
原初始化rho.3分类梯度范数76.95、加权capture3.51、smooth1.61e-5，分类梯度并非消失；
epsilon贡献约2.66e-10，不支持epsilon淹没预测解释。不能外推到全部样本／训练权重。
两rho统一raw初始化范围1→.1，rho.3四样本最大预测差异.00707→.49262，
但初始NLL2.940→4.312更高；支持有限候选检查，不证明因果、准确率收益或优化已经修复。
只预登记16层一次配对10轮新profile，待base全部结束再发布10轮入口并启动；保留base30，
整个pair按两rho最佳验证NLL均值选择，同分base，任何结果仍如实报告预算及未收敛风险。
不改模型／rho／数据／类别，不无限追加。完整规则在[协议](../../COIL20_PILOT_PROTOCOL_20261011.md)。

此队列严格只fit，结束生成8臂fit_manifest，实际释放GPU并停在
fit_complete_awaiting_validation_review。然后只读保存结果做CPU来源／PT／指标及
过拟合／欠拟合／未收敛审查，必要调整必须有限、成对、使用新profile并保留原结果；
不能为70%反换权重／无限追加。所有方案选择及pair／selection／global lock
完成后，获选每臂EMA才可首次单次test。当前所有test均0，正式最终结果尚未完成。
