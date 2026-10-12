# 筛选20类正则实验：12层完整100轮的训练/验证审查

两rho各完成100轮，原GPU子1151379/1151536退出，队列自动在GPU1/2接续4层。
本页只读取保存的PT、history、预测CSV和train/val标签；没有重新模型推理，也没有test。
运行：`caltech_selected20_s17_L2L4L12L16_reg100_4gpu_20261012`；训练Git `a477d916c`。

|rho|验证NLL选中轮|选中train accuracy|选中val accuracy|train宏召回|val宏召回|val balanced NLL|train-val差|
|---|---:|---:|---:|---:|---:|---:|---:|
|0|100|69.02%|54.77%|67.54%|51.95%|1.807827533|14.25pp|
|0.3|91|80.90%|60.78%|80.11%|57.94%|1.500334718|20.13pp|

残差组在本配对选中结果中，验证accuracy高6.01pp、宏召回高5.99pp，balanced NLL更低。
训练精度提升11.88pp，比验证增益大，泛化差距仍然明显。两组验证70%门槛都未达到。
残差组末轮train84.77%/val59.01%，差25.76pp；末轮不能替换NLL选中的E91。

最后十轮平均val balanced NLL相对前十轮改善4.782%/2.292%；E90→E100训练balanced NLL
分别改善10.749%/15.486%。无残差best在预算末端，优化仍未充分收敛。残差后期验证曲线
趋于平台，而train继续改善、差距扩大，存在过拟合迹象；但最后窗口均值仍改善，不能说
验证全面持续恶化，也不能声称已经充分收敛。曲线须同时看accuracy与NLL。

总时间17035.04/17144.27秒，平均每轮169.58/170.64秒（包含定期train评估）。
两臂只按验证EMA balanced NLL最低首次选模，共同100轮，没有追加轮次或候选。

CPU核验通过：100轮/best/last/config/data身份，70份实际源码和训练Git blob，61相位
每轮有限非零梯度，同初始化、100轮order/augmentation/MixUp配对、配置仅rho不同；
独立重生全部MixUp hash，包括12样本宏尾。train/val保存预测的ID、标签、argmax、
完整混淆矩阵、support、accuracy/macro/NLL均核对。两臂test0，全run test_started0。
没有提前创建任何selection/global/test锁。

[真实曲线PNG](learning_curves_L12.png)、[SVG](learning_curves_L12.svg)、
[PDF](learning_curves_L12.pdf)、[曲线CSV](curve_data.csv)、[汇总CSV](pilot_summary.csv)、
[保存结果CPU审核](pilot_audit.json)、[完整审查收据](fit_review.json)。图已实际视觉检查。

验证筛选及旧test已观测，使本任务属于探索性开发；单seed无误差条，不能证明稳定或
机制因果的残差收益。新类别/支持/正则共同改变，不能把与旧20类的差值归因于某项正则。
16层尚未完成，不能提前据此断言深度优劣；全部八臂完成审核并全锁后才获选EMA各一次test。
