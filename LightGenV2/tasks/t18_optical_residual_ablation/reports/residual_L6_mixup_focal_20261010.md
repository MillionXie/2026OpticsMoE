# 六层残差：MixUp／focal没有突破开发最高值

用户继续优化授权下，run `mango_rho03_L6_mixup_focal_20261010`，训练commit af8aedaef。
两候选从同一`smooth`第8轮父权重（测试开发87.80%）出发，seed17，lr0.00015
余弦、EMA0.99、相位平滑0.05、收光loss0.2；最多30轮、至少10轮、patience10。
训练梯度只用train，验证NLL选best，原父以epoch0参与；推理模型/光路/数据不变。
MixUp使用Beta(0.1,0.1)及同系数混合振幅和soft label；focal gamma1关注低置信训练目标。

GPU1在启动前被其它项目占用，因此两候选顺序使用物理GPU4，未共用显存。
实际MixUp第10轮、focal第11轮早停，全部权重/曲线/失败证据保留。
CPU验证了gamma0分类损失、MixUp线性soft目标和梯度有限非零；光学训练的全部
相位梯度也有限非零，源/config/父PT SHA检查通过。

| 方法 | 验证选定轮 | 完成轮数 | best EMA开发test | last raw开发test | last EMA开发test |
|---|---:|---:|---:|---:|---:|
| 原父 | 原8 | 原20 | 87.80% | — | — |
| MixUp | 0（保留父） | 10 | 87.80%（同权重复用） | 86.84% | 87.08% |
| focal | 1 | 11 | 87.80% | 87.56% | 87.56% |

按事先明确的test accuracy最高、同分最低test balanced NLL比较7个保存状态。
获选仍是原父`smooth`第8轮best EMA，367/418正确；**89.56%以上目标仍未达到**。
这是用户授权的测试选模开发指标，不是独立泛化或同预算消融；原无残差87.56%、
2/4层残差75.36/84.21均未重训或重选。

MixUp后期验证NLL由起点0.320538升至0.333186；focal首轮降至0.319396，
但后期恶化，末轮状态没有获得更高开发准确率。不能把这两种方法当成有效提升，
也不能把本轮负结果外推为所有MixUp/focal设置必然无效。继续同起点反复小步
训练缺少收益；后续改用训练期教师蒸馏时，应先检查教师验证优势，不能让较弱教师
误导光网络，电子教师不进入最终光学推理。

[完整记录](residual_L6_mixup_focal_20261010.json)、
[PNG曲线](residual_L6_mixup_focal_figures_20261010/validation_curves.png)、
[SVG](residual_L6_mixup_focal_figures_20261010/validation_curves.svg)、
[PDF](residual_L6_mixup_focal_figures_20261010/validation_curves.pdf)、
[CSV](residual_L6_mixup_focal_figures_20261010/source_data.csv)。单种子无误差条。
