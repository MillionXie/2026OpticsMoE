# 前三组共同40轮消融（部分阶段，未完成全九组）

旧同一20类，main4/rho.3，seed17、相同初始化/样本顺序/仿射，macro16/micro1，
fixed lr.002，EMA.95；除目标因素其余科学配置一致。三组最低首次val balancedNLL均选E40。
旧完成prefix完整接续、总40轮，不重播；没有test或正式选模锁。

| 因素 | Train accuracy | Val accuracy | Val macro recall | Val balanced NLL | Train−val gap | Val零召回类 |
|---|---:|---:|---:|---:|---:|---:|
| 基线8bit＋rowLN |33.75%|28.98%|34.17%|2.405106|4.77pp|7|
| 关闭振幅和相位量化，仅诊断不可部署 |34.69%|30.39%|35.08%|2.385514|4.31pp|7|
| 8bit＋整面LayerNorm |33.82%|32.16%|32.00%|2.482380|1.66pp|1|

关闭量化验证准确率仅+1.41pp，仍七类0召回，没有在这40轮预算内出现整体拟合恢复。
不能据此排除量化的后期/交互影响，也不能直接将不可部署浮点模型作为最终方案。
整面LN验证总体+3.18pp，train没有明显提升；恢复Faces65%、Motorbikes80%等召回，
但旧较好类别如trilobite67%→8%、umbrella55%→9%，macro−2.17pp，bNLL反而增加.077274。
只有airplanes验证召回为0，但另外8类仍<.2；不能以少零召回或小gap宣称过拟合已解决。
三组train也只有约34%，NLL仍下降，属于有限共同预算的初步诊断，不称充分收敛。
当前没有单一因素把模型恢复到满足75%目标的表现，须等待其余六组共同预算结果。

[真实accuracy/NLL曲线](first3_train_validation_curves.png)、[可编辑SVG](first3_train_validation_curves.svg)
蓝线为每轮验证，黑虚线/点为实际保存train端点（第1/每5轮），红星为val NLL选中EMA。
曲线不填造逐轮train数据，没有test点、单seed无误差条；三个面板使用相同accuracy/NLL坐标。
[逐类验证召回图](first3_validation_class_recall.png)、[可编辑SVG](first3_validation_class_recall.svg)
同时展示真实val support。PNG已实际视觉检查，文本/星标无裁切，布局可读。
[曲线CSV](curve_points.csv)、[选中汇总](selected_summary.csv)、[train/val各类support和recall](class_support_recall.csv)。

[CPU保存审核](cpu_audit.json)、[原历史及fit结果SHA和曲线证据](saved_curve_evidence.json)。
三组完整40/PT/config/原Gitblob/data/预测ID标签argmaxCMsupportmacroNLL、全部相位梯度、
Adam40×86步、独立重生全部40order/affine/MixUp含尾12通过，原17/17/12轮prefix逐项一致。
model_forward0/test_arrays0/test_evaluations0，原三个GPU子已实际退出，队列正常自动接续其他组。
这是完成三臂的partial CPU审查，父与其余组仍在运行，**全九屏障尚未满足**，不得启动正式方案/test。
121训练源仍e4cb1400fdaed5d4a1867770474e91b0ddc19617冻结，GPU仅2/5/6，不改旧源/数据/结果。
