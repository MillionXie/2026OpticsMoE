# BloodMNIST：逐层 OEO、三种子和输入覆盖对照

本轮完成 48 组新增训练，复用此前六组 seed17 权重，共评估 54 个模型。下表为官方测试集 3,421 张图像上的准确率，报告 seed17/27/37 的均值±样本标准差，单位为百分比。D2NN 主对照将输入上采样至整个相位孔径；旧的小输入设置单独列出。没有电子分类前端或残差旁路，OEO 在每个主干衍射层传播后执行。

## 结果

| 模型 | 2 层 | 4 层 | 6 层 |
|---|---:|---:|---:|
| MoE，无 OEO | 57.47±0.75 | 66.61±0.78 | 72.39±0.43 |
| MoE＋逐层 OEO | 71.58±1.23 | 81.17±0.85 | 84.48±0.60 |
| 全孔径 D2NN，无 OEO | 46.17±1.55 | 51.25±1.30 | 51.02±1.48 |
| 全孔径 D2NN＋逐层 OEO | 68.14±0.43 | 79.64±0.50 | 83.79±0.42 |
| 原小输入 D2NN，无 OEO | 43.83±0.98 | 53.16±1.11 | 51.48±1.41 |
| 原小输入 D2NN＋逐层 OEO | 61.47±0.27 | 74.16±0.63 | 79.50±0.76 |

全孔径主对照下，MoE＋OEO 的均值在三种深度最高，但六层配对差值仅 **0.69±0.91 个百分点**；三个种子差值依次为 −0.18、+1.64、+0.61。因此不能宣称六层存在稳定的大幅领先。六层无 OEO MoE 比无 OEO D2NN 高 21.37 个百分点，但比 D2NN＋OEO 低 11.40 个百分点，**不支持“无 OEO MoE≈D2NN＋OEO”**。

六层 D2NN＋OEO 的输入铺满相位面后，均值从 79.50% 升至 83.79%，提高 4.29 个百分点。此前约五个百分点的 MoE 优势很大一部分随此几何对照消失。输入上采样也改变空间尺度，不能将该提升全部归因于参数照明比例；但它说明旧的小输入对照不足以支撑严格的架构优势结论。无 OEO D2NN 未出现同样的上采样收益，完整非单调结果保留。

训练集多数类为中性粒细胞，恒猜该类的测试准确率为 19.47%，均匀随机猜测的期望为 12.5%。这与旧 Adrenal 二分类中“恒猜正常就有七成多”不同。排除精确像素重复后，3,415 张测试图上的六层结果分别为：MoE 72.45±0.43%、MoE＋OEO 84.52±0.63%、D2NN 51.08±1.48%、D2NN＋OEO 83.83±0.43%。

## 训练是否有效、是否过拟合

所选六层 MoE＋OEO 权重的训练−验证准确率差为 1.72±0.14 个百分点；D2NN＋OEO 为 1.58±0.10。两者验证−测试差分别为 1.80±0.24、1.51±0.10 个百分点。结合完整曲线，本轮没有旧实验那种明显的训练／验证分离，不能据此保证没有过拟合或没有同源图像偏差。

所有选定光学相位模块都检出了分类损失梯度和非零参数变化。逐像素照明、分类梯度覆盖及相位饱和统计保存在 `phase_audit.csv`；这不等于每个像素贡献相同。无 OEO D2NN 的训练、验证准确率都偏低且波动较大，更符合当前表示与优化设置下的学习受限，不能只归因于过拟合。

验证集路由诊断覆盖 18 个 MoE 模型，保存了每张图像的九路概率和实际入口功率。六层 MoE＋OEO 的路由相位 RMS 变化为 0.294–0.315，功率分配随输入变化；按 `1/Σ_j power_j²` 定义的平均有效支路数为 5.78–6.20。该数值衡量功率集中程度，不是激活数量；实现仍计算九条支路。无 OEO 六层的功率更集中，有效支路数为 2.33–3.05。类条件路由差异是描述性证据，不能代替固定路由／重新训练消融来证明专家分工的因果收益。

## 数据、训练与公平性范围

数据采用官方 BloodMNIST 28×28 RGB 文件，CC BY 4.0。训练去重并排除与验证集精确相同的图像后为 11,948 张，验证 1,712 张。固定 RGB 编码、归一化、九路功率约束、输入张量、相位孔径、光学参数、损失和指标定义见[完整技术说明](OEO_METHODS_20260917.md)。原 seed17 测试成绩此前已见过，本轮不作为全新隐藏测试；没有根据新增测试结果调参。

各模型联合训练全部相位，采用同一 lr=0.002、batch=16、最多30轮、EMA=0.95、标签平滑0.02、捕获率损失权重0.2、相位平滑权重0.02；按验证平衡 NLL 选权重。各模型的样本顺序和增强参数按种子配对并核验。逐层 OEO 为整面光强 LayerNorm→ReLU→Softsign→零相位振幅重编码。参数数量相近不意味着物理面积、路由光路或系统能耗相同；理想无损九端口与局部独立传播仍为仿真假设。

训练实际使用 4090 与 A100，设备 UUID 有记录。统一在 4090 上复评时，A100 训练的 `d2nn_L2_seed27` 未通过验证指标完全相等检查，原失败目录和日志保留，当时未读测试集。随后将全部 54 个锁定模型放回各自原训练 GPU，验证指标和每个验证分数均精确重放成功，再统一进入测试评估。没有放宽容差、修改权重或重新选模。

## 图与原始数值

- [深度—准确率曲线](figures_20260917/bloodmnist_results/depth_accuracy.png)、[全部种子点／均值／中位数](figures_20260917/bloodmnist_results/seed_distribution.png)。
- [输入覆盖对照](figures_20260917/bloodmnist_results/d2nn_input_coverage_control.png)、[OEO 配对增益](figures_20260917/bloodmnist_results/oeo_paired_gain.png)。
- [完整训练／验证曲线](figures_20260917/bloodmnist_results/paired_train_validation_curves.png)、[六层混淆矩阵](figures_20260917/bloodmnist_results/confusion_L6.png)。
- [真实测试图像及预测](figures_20260917/bloodmnist_results/examples/prediction_examples.png)：固定六层 seed17，按类别和正确／错误条件取字典序第一个 ID，不挑选最佳种子或最漂亮样本；分数不称为校准置信度。
- `figures_20260917/bloodmnist_results/` 内同时提供 PDF/SVG、逐种子指标、配对差值、曲线数据和独立复核清单；混淆矩阵类别 0–7 依次为嗜碱、嗜酸、成红、未成熟粒、淋巴、单核、中性粒、血小板。
- 路由图及逐图分配：`runs/smoke/bloodmnist_selected_routing_20260917/`。真实训练图、编码和照明示意见 `figures_20260917/bloodmnist_inputs/`。

三个点不用于拟合小提琴密度，也不据此作显著性或患者泛化声明。标准差只反映当前固定划分上的训练变化，并混有设备浮点路径影响。

## 复现身份与操作

主 run：`runs/simulation/bloodmnist_oeo_multiseed_20260917`；复用来源：`runs/simulation/bloodmnist_dedup_s17_20260916`。训练协调器 commit 为 `7492f9305710995cfe282d5189e99b23abc4da26`；正式原设备复评 commit 为 `e17ca0b32ce823eda4748e58ac04edab5e06e95b`。后续文档提交不改变锁定训练源码；每个 job 的实际 commit、命令、配置和源码 SHA256 均在 metadata 中。

环境：Python 3.11.15、PyTorch 2.6.0+cu124、CUDA 12.4、cuDNN 90100、NumPy 1.26.4、scikit-learn 1.9.0、matplotlib 3.8.4。训练完整命令见[预声明协议](../../reproduction/BLOODMNIST_MULTISEED_PROTOCOL.md)及主 run 的 `metadata.json`。评估故障恢复与 CPU 分析使用以下入口，参数中的 RUN/DATA/MAP 指向该轮记录：

```bash
python reproduction/evaluate_native_device.py --run RUN --data DATA --dataset bloodmnist --gpu-map MAP
python reproduction/analyze_oeo_suite.py --run RUN --task-root TASK --out EMPTY_OUTPUT --dataset bloodmnist
python reproduction/plot_prediction_examples.py --run RUN --data DATA --dataset bloodmnist --out EMPTY_EXAMPLE_OUTPUT
python reproduction/audit_selected_routing.py --run RUN --data DATA --dataset bloodmnist --out EMPTY_ROUTING_OUTPUT
```

原设备恢复入口仅用于尚无最终结果、且已检查过原始评估失败的锁定 run；不可用来替换已完成评估。CPU 分析不重新训练或选择模型。

- 数据 SHA256：`062023e186f537e26b3c21ea3b2614ddfc475e8a14825dfd20663bbb7e37bddc`。
- 选模锁 SHA256：`62473a2e60471a734ab95d4ebbaa74236e57f89c47d6f82a74881b4734530c72`。
- 最终 results.json SHA256：`693bc205d29e1d15bd9d6476f1f143175a0fb9012d4960dffdc5d959cbec3748`。
- 54 份权重、逐样本测试分数及历史 SHA256 见图表目录的 `independent_verification.json`。独立复核通过选中轮数、所有预测指标、数据配对、权重身份、分类梯度与相位变化检查。
- 正式预测与相位审计在主 run 的 `evaluation_native_device/`；旧的失败 `evaluation/` 保留，不作为最终成绩。`evaluation_directory.json` 明确指向正式目录。
