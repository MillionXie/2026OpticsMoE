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

## 深层只做了数值／速度预检，尚无正式精度

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
