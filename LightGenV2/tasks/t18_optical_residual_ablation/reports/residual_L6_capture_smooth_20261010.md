# 六层残差：收光与平滑正则候选未获采用

run `mango_rho03_L6_capture_smooth_20261010`，训练commit fad46834a。
两候选从同一EMA父best epoch14接续：父验证balanced NLL0.320627689，
宏平均召回88.8571%，历史一次测试87.08%。共同lr0.0003/EMA0.99、原增强、
最多50轮/minimum15/patience12，输入/光路/模型/rho0.3全部不变。
无残差基线及2/4层没有重训或重测。

| 候选 | 唯一目标项改变 | 所选epoch/完成轮数 | 训练accuracy | 验证accuracy | 验证宏平均召回 | 验证balanced NLL |
|---|---|---:|---:|---:|---:|---:|
| capture | 收光loss权重0.2→0.05 | 3/15 | 97.39% | 88.97% | 89.09% | 0.321226 |
| smooth | 相位平滑权重0.02→0.05 | 8/20 | 97.39% | 88.73% | 88.82% | 0.320538 |
| 父EMA | 两项原值 | 原14/26 | 97.24% | 88.73% | 88.86% | 0.320628 |

预定采用条件是验证NLL改善且宏平均召回不低于父；没有候选同时满足，
因此**均拒绝，均不测试**，继续保留父模型。没有根据已观察测试反选历史87.56%。
capture的召回提高约0.23pp，但NLL升高约0.00060；smooth的NLL下降仅0.00009，
召回下降约0.038pp，也未达到父比较的min_delta0.0005实质改进阈值。

末轮验证NLL：capture0.325225，smooth0.324904，均高于各自best及父，
训练高而验证提升不足，继续延长这两种配置缺少依据。两组训练−验证accuracy
差8.42/8.67pp；目前不能把泛化差唯一归因于某个辅助项。
捕获率parent59.99%，capture56.25%，smooth60.11%。降低收光权重确实降低了
该窗口能量占比，但没有得到全面分类改善，不能据此断言“收光loss是主要原因”。
这里是窗口/全场能量比例，不是总光效率。

最新结果不变：固定无残差2/4/6层74.40/82.30/87.56%，验证选定残差
75.36/84.21/87.08%。额外预算的单种子探索，不是同预算严格消融；未确定上限。
此轮不自动无限追加。若进一步研究，应考虑预先约定验证accuracy/召回与NLL
的多指标目标，或训练期教师蒸馏等新的方法；不能看到某次测试后临时换选模准则。
新的计算图或光学合同改变需要另行确认，本轮未引入这些改变。

保存了两组best/last、metadata、全部曲线与失败结果。相位梯度有限非零，
源/父SHA检查通过，run无test_predictions.csv，selection_decision记录eligible为空。
项目GPU1/4全部释放；只读保存结果分析，未增加任何测试推理。

[完整验证证据](residual_L6_capture_smooth_20261010.json)；
[PNG](residual_L6_capture_smooth_figures_20261010/validation_curves.png)；
[SVG](residual_L6_capture_smooth_figures_20261010/validation_curves.svg)；
[PDF](residual_L6_capture_smooth_figures_20261010/validation_curves.pdf)；
[CSV](residual_L6_capture_smooth_figures_20261010/source_data.csv)。单种子无误差条。
