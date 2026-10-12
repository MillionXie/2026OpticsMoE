# 筛选20类正则实验：2层完整100轮的训练/验证审查

两rho各完成100轮，原GPU子1416439/1418565已退出，实际GPU1/2无项目进程。
本页只读取保存的PT、history、预测CSV和train/val标签，没有模型复推或test。
运行：`caltech_selected20_s17_L2L4L12L16_reg100_4gpu_20261012`；训练Git `a477d916c`。

|rho|验证NLL选中轮|选中train accuracy|选中val accuracy|train宏召回|val宏召回|val balanced NLL|train-val差|
|---|---:|---:|---:|---:|---:|---:|---:|
|0|100|53.57%|45.58%|52.47%|44.49%|2.021882625|7.99pp|
|0.3|100|57.80%|51.59%|56.74%|49.92%|1.838665527|6.21pp|

残差组验证accuracy高6.01pp、宏召回高5.44pp、balanced NLL更低；训练accuracy高4.23pp，
选中train-val差较小。两组验证70%门槛都未达到，不能把小差距误作高性能或已收敛。

最后十轮平均val balanced NLL相对前十轮改善2.005%/3.040%；E90→E100训练balanced NLL
改善2.684%/5.071%。两组best都在预算末轮，train/val NLL继续下降；训练精度仍低，
更符合拟合/优化尚不足的迹象，不能主要归因为严重过拟合或仅凭低train断定容量上限。
没有验证持续恶化的证据，也不声称充分收敛。既定100轮完成，不自动扩轮追分。

总时间1566.28/1618.39秒，平均每轮15.57/16.10秒（含周期train评估）。两rho同micro4、
有效16，共同100轮，按验证EMA balanced NLL最低首次选模，未追加候选或轮次。

CPU核验通过：100轮/best/last/config/data身份，70份实际源码和训练Git blob，11相位
每轮有限非零梯度，同初始化、100轮order/augmentation/MixUp配对、配置仅rho不同；
独立重生全部MixUp hash，包括12样本宏尾。train/val保存预测ID、标签、argmax、完整
混淆矩阵、support、accuracy/macro/NLL均核对。两臂test0，全run test_started0，无提前锁。

[真实曲线PNG](learning_curves_L2.png)、[SVG](learning_curves_L2.svg)、[PDF](learning_curves_L2.pdf)、
[曲线CSV](curve_data.csv)、[汇总CSV](pilot_summary.csv)、[CPU审核](pilot_audit.json)、
[完整审查收据](fit_review.json)。图已实际视觉检查。

本任务经验证筛选且旧test已观测，属于探索性开发；新类别/支持/正则共同改变，不能
与旧20类作受控比较或归因某一正则。单seed无误差条，不保证跨种子稳定残差收益。
16层尚未完成，测试继续封存；全8完成CPU审核及全锁后才各获选EMA首次一次test。
