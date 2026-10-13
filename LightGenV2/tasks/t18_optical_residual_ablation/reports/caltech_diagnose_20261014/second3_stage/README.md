# 输入、路由和全局场共同40轮消融（部分阶段）

同一旧20类、main4/rho.3、seed17和初始化、样本顺序、仿射、macro16/micro1、EMA.95、fixed lr.002。
新三组均完成40轮、最低首次val balancedNLL均选E40；不读取test，不建立正式选模锁。
基线来自已经完成的前三组审核，没有重新推理。

| 因素 | Train accuracy | Val accuracy | Val macro recall | Val balanced NLL | Train−val gap | Val零召回类 |
|---|---:|---:|---:|---:|---:|---:|
| 基线8bit＋rowLN＋hardTop2 |33.75%|28.98%|34.17%|2.405106|4.77pp|7|
| RGB固定拼图输入 |45.77%|44.17%|42.76%|2.207855|1.60pp|2|
| 四专家连续路由，保留输入条件权重 |46.87%|47.00%|48.17%|2.041299|−0.13pp|2|
| expert→global保留478整场，不先缩成224再复制 |43.88%|45.94%|45.11%|2.124360|−2.06pp|4|

三个独立因素分别使验证准确率相对基线增加15.19、18.02、16.96个百分点，
宏召回增加8.59、14.00、10.94个百分点，bNLL均改善。此前关闭量化只增加1.41pp。
这为当前预算下输入编码、稀疏路由、expert→global重编码的信息处理均限制学习提供证据；
不能将其中一个因素认定为全部原因，也不能将三项增益相加预测组合精度。
固定初始化单次40轮并不代表充分收敛或跨种子因果稳定性。

RGB改变固定R/G/B/mean拼图输入而非增加可学习前端；dense保留四个随输入变化的L2归一化
路由权重，不是均匀四专家。preserve_global仅在下一层global时保留478场，expert转换仍224。
三项都保留真实8bit联合振幅/相位量化，不以浮点不可部署模型冒充修复。
仅推理切旧3层已训练权重为dense失败，与这里fresh dense重新训练不同，不能混用结论。

逐类仍未通过正式门槛。RGB验证helicopter/kangaroo为0；dense验证Leopards/watch为0，
crayfish只有10%；preserve_global验证crab/crayfish/helicopter/kangaroo为0。
旧Faces/Motorbikes/airplanes在dense下分别45%/90%/75%，保留全局场50%/95%/75%，
说明此前类失效并非仅由图片类别本身决定，不能靠无限删除类别掩盖结构问题。
新20类已经人工指定准备并冻结，后续仍需在新train/val逐类验证，不重新按test筛选。

三组train、val曲线仍上升，NLL仍下降；40轮train约44–47%，未达到充分拟合。
小gap/负gap不证明过拟合已解决，有限验证样本和训练/验证support不同也影响总体gap。
后续必须结合更长的有界验证预算、宏召回、逐类持续失效和路由占用，不能仅看train提升。
75%最终test目标尚未验收，正式六组尚未启动。待最后MixUp.2/.6/legacy同预算完成后，
执行全九CPU审核，再决定修正和最多三个4层正则候选；不提前测试或扩搜索预算。

[真实accuracy/NLL曲线](second3_train_validation_curves.png)、[SVG](second3_train_validation_curves.svg)
蓝线为每轮val，黑点/虚线为真实train端点，红星为val NLL选中EMA；统一坐标，不填造train数据。
[逐类验证召回及support](second3_validation_class_recall.png)、[SVG](second3_validation_class_recall.svg)。
两张PNG已实际视觉检查，标题、刻度和E40星标可读无裁切；无test点或单seed误差条。
[曲线CSV](curve_points.csv)、[选中汇总](selected_summary.csv)、[train/val各类support/recall](class_support_recall.csv)。

[独立新三组CPU审核](cpu_audit.json)、[保存历史及fit结果SHA/曲线证据](saved_curve_evidence.json)。
原三个GPU子3100550/3101484/3106048实际退出，完整40/PT/config/data/源当前及原Gitblob、
全部梯度/Adam3440步、独立重生40order/affine/MixUp含尾12、预测ID标签argmaxCMsupport宏NLL通过。
model_forward0/test_arrays0/test_evaluations0；父与最后三组仍在运行，不是全九屏障。
121运行源仍e4cb1400fdaed5d4a1867770474e91b0ddc19617，实际GPU仅2/5/6，原权重/数据/旧结果保留。
