# 六层30%残差：第二轮续训结果

用户明确要求再继续六层。从上一轮选定EMA权重接续，原测试87.56%，
验证balanced NLL0.329087。只使用用户指定GPU4，其他深度与无残差均未重训。
run `mango_rho03_L6_continue50_round2_20261010`，训练commit b149b3e88。
最多50轮、minimum15、patience12、min_delta0.0005，AdamW/EMA重启，
lr0.0006余弦，augmentation epoch偏移150，输入/数据/rho/光路/损失不变。

实际23轮触发早停，选第11轮EMA：验证balanced NLL0.324428，低于父0.329087。
末轮NLL0.324117比所选略低，但改善幅度0.000310未达到预设min_delta，
不会覆盖已选checkpoint；故是按预设改进阈值选模，非精确数值最小。
选模SHA锁定后仅测试一次，未按测试选择，父checkpoint未重复测试。

| checkpoint | 训练accuracy | 验证accuracy | 测试accuracy | 验证balanced NLL | 测试balanced NLL |
|---|---:|---:|---:|---:|---:|
| 上一轮六层父权重 | 96.92% | 88.24% | 87.56% | 0.32909 | 0.39334 |
| 本轮第11轮EMA | 96.98% | 87.99% | 86.84% | 0.32443 | 0.38681 |

本轮测试净少正确3/418条，accuracy下降0.72pp，但测试NLL降低。
验证NLL与accuracy不等价，不能将这些指标变化合称“性能全面提高”。
本轮训练−验证差8.99pp，训练已近97%；再加训练收益趋缓，存在泛化差，
不能据此唯一诊断为过拟合。所选历史和当前权重均保留，不用已观察的测试
反过来换选父模型，也不把本轮称为超过基线或确定理论上限。

| 主干 | 固定无残差测试 | 最新验证选定30%残差测试 | 差值 |
|---|---:|---:|---:|
| 2 | 74.40% | 75.36%（未重测） | +0.96pp |
| 4 | 82.30% | 84.21%（未重测） | +1.91pp |
| 6 | 87.56% | 86.84% | −0.72pp |

这是只残差获得额外训练预算的单种子探索，不是等预算严格消融。
不再自动追加无限续训。如果后续继续，建议优先在训练/验证上检验增强或
正则，而不是继续延长同一优化过程；相干相位面泄漏也不等同于Transformer完整块
的恒等捷径。当前结果不能证明LLM部署抗噪或残差随深度增强收益。

检查：checkpoint源与config身份、锁定SHA、408/418条验证/测试预测，
所有已记录相位梯度有限且非零；分析读取保存结果，不重复测试推理。
全部项目训练/测试进程结束，GPU释放。源码/报告仅main Git同步；旧证据保留。

[完整证据](residual_L6_round2_20261010.json)；
[展示PNG](residual_L6_round2_figures_20261010/depth_accuracy.png)；
[SVG](residual_L6_round2_figures_20261010/depth_accuracy.svg)；
[PDF](residual_L6_round2_figures_20261010/depth_accuracy.pdf)；
[曲线](residual_L6_round2_figures_20261010/learning_curves.png)；
[CSV](residual_L6_round2_figures_20261010/source_data.csv)。
