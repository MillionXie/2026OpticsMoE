# FishNet：验证选模重跑（2026-10-11）

用户11:00要求暂时不按test选权重。新run `fishnet_val_s17_e100_gpu1_20261011`，
入口`fishnet_ablation.py`，profile `fishnet_validation_profile.json`。
源码通过检查、发布main并同步服务器后启动；未开始前不填写性能数字。

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
