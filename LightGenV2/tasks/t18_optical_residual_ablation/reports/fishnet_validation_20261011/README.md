# FishNet：验证选模重跑（2026-10-11）

用户11:00要求暂时不按test选权重。新run `fishnet_val_s17_e100_gpu1_20261011`，
入口`fishnet_ablation.py`，profile `fishnet_validation_profile.json`。
已于北京时间11:07:35启动，父PID305617，后续子PID以current_process.json为准。
训练源码main `ce7edf074`已发布并同步服务器，Linux11项CPU检查通过；
Windows9过／2项原生Torch检查因环境限制跳过。不改共享环境。
六层rho0已于北京时间14:05:48完成100轮并通过CPU证据核验；其子PID305989已退出。
六层rho0.3已自动接续，初始子PID446585，后续以current_process.json为准。
尚未完成的组不填写最终性能数字，不提前作残差配对结论。

六组顺序为六层rho0／rho0.3、四层rho0／rho0.3、两层rho0／rho0.3，仅GPU1串行。
各组从头seed17、完整100轮，同原初始化、原光学模型、lr／EMA／损失／增强与有效batch16。
每轮验证EMA balanced NLL最小选best，精确同分保留第一次，不因更高accuracy替换。
训练不读取test；100轮后先核验PT、配置、来源、缓存并写selection_lock，随后对该EMA
仅评估一次test。不会测试raw／last或按test成绩改变采用决定。

旧test-selected四个完成组的真正验证最优轮72／97／97／97均未保留权重；原来只有
test-best和last，因此无法诚实地从旧PT补选。这是重跑的原因。
旧run、开发成绩、逐样本结果、best／last及全部Mango结果均保留原身份。

FishNet v1原3013张、八原生鱼虾类别、CC BY4.0，train2106／val448／test459。
数据和分组不变，HEIC全部保留。新profile严格引用原准备profile SHA
`17cbc0264fd2c633f4ba8aef5aeaed6cc8c69ee2c25273370e7da841c1112229`，
缓存SHA `6257ad87af9e23e573890aa69a3ecb5e1ba6e8afd93b52fb69fec5339ffca267`。
唯一协议变更是选模和测试时机，模型、rho、类、数据、训练设置均未改变。

这是**验证选模的探索性评估**。同一test在旧run已经多次观测，不能声称独立泛化。
图像级划分不代表个体独立；Ruhi Fish只有9张test，必须报告macro recall和每类支持。
单种子无误差条，负结果保留，不预设残差应优于无残差。

完成后将从保存指标／预测／PT做CPU审计，报告train／val／test、NLL、macro和获选轮，
真实PNG／SVG／PDF／CSV以及配对初始化／顺序／增强／数据身份，不重复推理已测test。

## 六层无残差：验证选模获选结果

第72轮EMA是全部100轮验证balanced NLL最低的首次出现（0.2526981831）。
第97轮验证accuracy为92.86%，高于获选轮92.19%，但balanced NLL较高，未替换权重。
选模不按accuracy，也没有根据下列test成绩反换旧test-selected权重。

| 划分 | 准确率（正确／支持） | 宏平均召回 | NLL | balanced NLL |
|---|---:|---:|---:|---:|
| train | 98.91%（2083／2106） | 97.82% | 0.070366 | 0.091061 |
| val | 92.19%（413／448） | 90.51% | 0.211962 | 0.252698 |
| test | 89.98%（413／459） | 86.05% | 0.263383 | 0.357830 |

| 类别 | train支持 | val支持 | test支持 | test召回 |
|---|---:|---:|---:|---:|
| Mola fish | 362 | 77 | 78 | 96.15% |
| Pabda fish | 304 | 65 | 66 | 100.00% |
| Prawns | 158 | 34 | 35 | 77.14% |
| Puti fish | 384 | 82 | 83 | 87.95% |
| Ruhi Fish | 37 | 7 | 9 | 88.89% |
| Shing mach | 418 | 89 | 91 | 94.51% |
| Shrimps | 85 | 18 | 19 | 57.89% |
| Tengra | 358 | 76 | 78 | 85.90% |

`selection_lock.json`于北京时间14:05:12写入，记录验证获选PT、配置、来源、数据SHA及
锁定前test评估0次；随后保存一次test预测，最终`test_evaluations=1`、
`test_used_for_selection=false`。100条history仅含train周期评估／val，没有逐轮test。
CPU从保存预测重算三划分accuracy／macro／NLL／混淆矩阵，核对原manifest的唯一ID及顺序、
缓存标签、概率argmax和类别支持；未做新的光学推理或重复test。
best／last配置及来源身份、完整100轮、数据SHA及60个发布Git blob／运行文件身份均通过。
31张可训练相位共1,329,544参数；其它固定buffer不计作可训练参数或Linear。
best SHA256：`5452b9d4c25e1e18d509502418b9e2d72838fb820916d173af34d0efc67878db`。

训练末轮train99.24%／val92.19%，但这是末轮EMA诊断，不替代第72轮获选PT。
全组10671.6秒，末20轮均104.2秒（含周期评估）。实际用时受共享GPU／CPU竞争影响，
剩余深度耗时仍待实测；六层残差组尚未完成。
验证／test的CCD捕获比例为63.49%／62.61%，只表示探测窗占输出整场的比例，
不等于整模型光透过率，也不是残差因果证据。

完整[CPU审计JSON](L6_rho0.0_audit.json)、[PNG学习曲线](learning_curves_L6.png)、
[可编辑SVG](learning_curves_L6.svg)、[PDF](learning_curves_L6.pdf)、
[逐轮数值CSV](curve_data.csv)、[获选结果CSV](stage_summary.csv)已保存。
图只画train／val曲线；黑点是完成验证锁定后一次test，横坐标标注获选epoch，
不表示该轮曾访问test，不绘制不存在的逐轮test曲线。单种子不加误差条。
