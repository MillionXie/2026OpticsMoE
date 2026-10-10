# FishNet八分类：阶段结果（2026-10-11）

run `fishnet_s17_e100_gpu1_checked_20261011`。本表仅收录完成100轮且已核验的组；
其它组继续使用同一队列、数据、相位初始化、增强与预算，不能用未完成成绩比较残差。
旧Mango最终开发88.04%及全部历史结果、权重与失败记录保留。

| 主干层数 | 未调制振幅系数 | 获选轮／状态 | 训练准确率 | 验证准确率 | test开发准确率 | test宏平均召回 | test NLL／balanced NLL |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 0 | 65／EMA | 98.62% | 91.96% | 91.72%（421/459） | 87.86% | 0.25833／0.36398 |

六层rho0完整100轮结束于北京时间04:15:53，原PID11951已经退出。
六层rho0.3由同一调度自动启动，当前PID97852、物理GPU1 UUID `e8837b85…`，
本项目仍仅一个GPU训练子进程。两者均microbatch2累计有效batch16、Torch上限3GiB；
原无残差组近期每轮约98秒（包含周期评估），全组约2小时51分26秒。
单卡速度受其他项目临时占用影响；六层配对预计北京时间约07:00前后完成，
四层和两层的耗时仍须实测，不把初始15–20小时估计当作保证。

## 保存证据核验

- [完成组CPU审计](L6_rho0.0_audit.json)：100轮history、200个raw/EMA开发候选按
  accuracy最高／同分balanced NLL最低选择，选中第65轮EMA。
- best PT SHA256 `417371c955fe81e704ab8b7f8b56a88d22fefd3e1f48f48ad3ab241ba0a84c51`；
  PT配置、数据SHA、选中轮次与来源一致，58个运行源码文件及发布Git blob身份均通过。
- 全部train/val/test保存预测按manifest原顺序、唯一ID、八类支持、argmax、混淆矩阵、
  accuracy、macro recall与NLL核对；无新增GPU推理或重复测试。
- 31张可训练相位共1,329,544参数；PT还含固定温度、相位偏置及专家输出增益buffer，
  不把buffer数量算成可训练参数。无CNN或Linear权重；100轮相位梯度全部有限非零。
- 配对组完成后再核验双方initial SHA、100轮orders/transforms、参数量与数据身份。

## 曲线诊断与限制

获选权重训练−验证准确率差约6.66个百分点。后期训练准确率升至99.24%，
验证准确率约92%附近波动，验证balanced NLL在约第60轮后趋稳；
这提示继续拟合训练数据的收益有限，不能据此断言光学结构失效。
第100轮raw开发准确率也为91.72%，但balanced NLL更高，按既定同分规则保留第65轮EMA。

类别不平衡使整体accuracy高于macro recall；Shrimps开发召回11/19=57.89%，
Ruhi Fish开发支持只有9条。最终须保留每类支持与混淆矩阵，不能只报整体91.72%。
所有开发指标按用户授权逐轮test选模，**test-selected DEVELOPMENT，非独立泛化**。
单种子、图像级划分，不能保证个体独立或残差收益；当前尚无完整配对结论。

[学习曲线PNG](learning_curves_L6.png)／[可编辑SVG](learning_curves_L6.svg)／
[PDF](learning_curves_L6.pdf)／[逐轮CSV](curve_data.csv)／[完成组表格CSV](stage_summary.csv)。
图与CSV直接来自CPU审计的保存指标，单种子不加误差条。
