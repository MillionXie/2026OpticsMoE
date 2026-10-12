# 筛选20类正则实验：4层完整100轮的训练/验证审查

两rho各完成100轮，原GPU子1386782/1388777退出，队列自动在GPU1/2接续2层。
本页只读取保存的PT、history、预测CSV和train/val标签，没有模型复推或test。
运行：`caltech_selected20_s17_L2L4L12L16_reg100_4gpu_20261012`；训练Git `a477d916c`。

|rho|验证NLL选中轮|选中train accuracy|选中val accuracy|train宏召回|val宏召回|val balanced NLL|train-val差|
|---|---:|---:|---:|---:|---:|---:|---:|
|0|99|61.81%|49.82%|59.89%|47.65%|1.879633985|11.98pp|
|0.3|100|76.90%|61.84%|75.59%|59.26%|1.527726232|15.06pp|

残差组验证accuracy高12.01pp、宏召回高11.61pp、balanced NLL更低；训练accuracy
提升15.09pp，仍大于验证增益，泛化差距没有消失。两组验证70%门槛都未达到。
无残差末轮train62.46%/val50.18%不能替换NLL选中的E99。

最后十轮平均val balanced NLL相对前十轮改善3.377%/5.450%；E90→E100训练balanced NLL
改善6.197%/11.719%。两组best都在预算末端，train/val NLL仍下降，不能声称充分收敛。
存在训练/验证差距，但没有验证持续恶化的证据，不能仅凭差距归因为严重过拟合。
曲线同时显示残差的训练与验证优势；单seed仍不能证明稳定或机制因果收益。

总时间1864.87/1877.00秒，平均每轮18.27/18.45秒（含周期train评估）。两rho同micro4、
有效16，共同100轮，按验证EMA balanced NLL最低首次选模，没有追加候选或轮次。

CPU核验通过：100轮/best/last/config/data身份，70份实际源码和训练Git blob，21相位
每轮有限非零梯度，同初始化、100轮order/augmentation/MixUp配对、配置仅rho不同；
独立重生全部MixUp hash，包括12样本宏尾。train/val保存预测ID、标签、argmax、完整
混淆矩阵、support、accuracy/macro/NLL均核对；两臂test0，全run test_started0，未提前锁定。

[真实曲线PNG](learning_curves_L4.png)、[SVG](learning_curves_L4.svg)、[PDF](learning_curves_L4.pdf)、
[曲线CSV](curve_data.csv)、[汇总CSV](pilot_summary.csv)、[CPU审核](pilot_audit.json)、
[完整审查收据](fit_review.json)。图已实际视觉检查。

新类别/支持/正则共同改变，不能与旧20类作受控比较或归因某一正则；本任务经验证筛选
且旧test已观测，属于探索性开发。其余深度未全部完成，测试仍封存；全8完成CPU审核及
全锁后才各获选EMA首次一次test，无单seed误差条。
