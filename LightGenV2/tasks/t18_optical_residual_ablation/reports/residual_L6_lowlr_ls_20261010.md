# 六层小学习率与标签平滑：目标未达到

用户要求尽快优化六层30%残差至89.56%以上。本轮从同一EMA best epoch14
父权重出发（原一次测试87.08%），run `mango_rho03_L6_lowlr_ls_20261010`，
训练commit c06ba28b8。两候选仅验证训练，lr0.0001余弦至0.00001、EMA0.99，
最多50轮/minimum15/patience12、原增强和光路；lowlr_ls额外将label smoothing
从0.02增至0.05。数据、rho、模型、router无残差、OEO/CCD全部不变。

| 候选 | 所选epoch/完成轮数 | 训练accuracy | 验证accuracy | 验证宏平均召回 | 验证balanced NLL |
|---|---:|---:|---:|---:|---:|
| lowlr | 9/21 | 97.29% | 88.48% | 88.63% | 0.320529 |
| lowlr_ls | 1/15 | 97.29% | 88.24% | 88.37% | 0.322030 |
| 父EMA | 原14/26 | 97.24% | 88.73% | 88.86% | 0.320628 |

预先固定采用条件：验证NLL改善且宏平均召回不低于父，两者均不满足。
lowlr的NLL只降低约0.000099，召回下降约0.231pp；lowlr_ls两项变差。
均拒绝、**均未读取测试集**。没有改用测试来挑候选，也没有为了目标改指标/划分。
最新六层继续保留父EMA及原测试87.08%；89.56%目标**未达到**。
测试418条，若达到89.56%以上至少需375条正确（89.71%），当前父364条，
尚差净增加11条；这是目标换算，不能用于逐样本测试调参。

末轮验证NLL分别0.322946/0.326979，高于各自best和父，因此早停有依据。
训练−验证差8.81/9.05pp，小学习率及更强标签平滑没有缩小到可用收益；
不能因此证明没有其它有效的抗过拟合方法，更不能称已确定架构上限。
目前继续反复降低学习率、从同一收敛权重热启动缺少新增诊断依据。
若再继续，训练期教师蒸馏、从较早checkpoint重新规定正则计划等属于更不同的
训练策略；须提前固定验证选择标准和时间预算，不能因测试目标而事后换标准。

固定无残差2/4/6层74.40/82.30/87.56%，残差最新75.36/84.21/87.08%不变，
2/4层和无残差没有重训或重测。本研究为仅残差追加预算的单种子探索，不是
同预算严格消融；既有负结果全部保留。

检查：源/config/父SHA已按发布Git blob审核，相位梯度有限且非零，
selection_decision记录eligible为空、test_read=false，run无test_predictions.csv。
GPU1/4训练均结束，项目GPU全部释放，未追加无限训练。

[完整证据](residual_L6_lowlr_ls_20261010.json)；
[PNG曲线](residual_L6_lowlr_ls_figures_20261010/validation_curves.png)；
[SVG](residual_L6_lowlr_ls_figures_20261010/validation_curves.svg)；
[PDF](residual_L6_lowlr_ls_figures_20261010/validation_curves.pdf)；
[CSV](residual_L6_lowlr_ls_figures_20261010/source_data.csv)。单种子无误差条。
