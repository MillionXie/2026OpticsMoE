# Kather2016输入覆盖修正版：seed17训练与测试

按用户要求仅完成seed17的2/4/6层、四架构，共12份权重。其他种子训练已停止，日志和权重保留，不参与本表。训练/验证/测试分别3496/752/752张，八类均衡、图像级划分。输入与模型见[Kather覆盖说明](KATHER_FOLLOWUP_20260917.md)。

统一候选`lower_capture`由六层seed17四架构的平均验证balanced NLL选定（0.931774，另一候选1.058120），没有按架构差值或测试集选择。捕获损失权重0.02，其余训练配置与60轮协议一致。每个模型按验证损失选EMA权重，训练和测试柱均使用同一份权重、关闭训练增强。所有12份权重在原训练GPU上重放训练/验证预测，最大分数误差为0，再读取测试集。NumPy独立复核36份逐样本预测，指标一致、三划分ID不交叉。

## 准确率（%）

各单元格为训练／测试；不是多种子均值。

| 模型 | 2层 | 4层 | 6层 |
|---|---:|---:|---:|
| MoE | 61.76／58.24 | 66.59／63.16 | 68.88／64.63 |
| D2NN | 49.54／43.48 | 48.97／40.82 | 50.37／41.62 |
| MoE＋逐层OEO | 76.40／70.21 | 87.27／79.12 | 92.82／80.85 |
| D2NN＋逐层OEO | 73.77／66.89 | 78.63／71.14 | 80.21／71.14 |

MoE＋OEO相对D2NN＋OEO测试优势为3.32、7.98、9.71个百分点。无OEO的MoE仍低于D2NN＋OEO，未达到希望的跨OEO排序。D2NN＋OEO四/六层测试准确率相同；D2NN无OEO不单调，未人为平滑或改选权重。

MoE＋OEO训练−测试差为6.19、8.15、11.97个百分点，存在随深度增大的泛化差距；六层测试仍较四层提高1.73个百分点，所以不能仅凭柱状图断言“增加层数使泛化性能变差”，也不能说已消除过拟合。无OEO的MoE差距为3.51、3.43、4.25个百分点。测试集曾在旧版本中被观察，本轮不是全新盲测；单种子没有种子不确定性估计，图中不画误差棒。

降低捕获损失会影响光能分布：MoE＋OEO六层测试探测窗口捕获率仅1.69%，D2NN＋OEO为6.71%。捕获率指八窗口能量占末端全场能量的比例，不是整机光电效率。分类性能提升不能当成光能利用效率提升。

## 图片与证据

[2层柱状图](figures_20260917/seed17_train_test/train_test_L2.png)、[4层柱状图](figures_20260917/seed17_train_test/train_test_L4.png)、[6层柱状图](figures_20260917/seed17_train_test/train_test_L6.png)、[三图合并](figures_20260917/seed17_train_test/train_test_all_depths.png)。同目录提供PDF、SVG、`metrics.csv`及`independent_verification.json`。

测试run：`runs/simulation/kather2016_coverage_seed17_test_20260917`。原六层权重来自`kather2016_coverage_pilot_20260917`，2/4层来自`kather2016_coverage_multiseed_20260917`。目录中的“multiseed”是原计划名称，不代表本轮完成多种子。

评估源码commit：`64a268f9`；绘图/独立验证commit：`a9039f2e`。协议锁SHA256：`72f700af10d8cb062621b1600f91c22ed946710b10ccad3870f0d48cfc4af1b7`；结果SHA256：`41e736f14e9ef9c44a7011039cf19eaee1e1387d1d61f7ff3879d92e107def40`。每份权重SHA256、源码哈希、数据SHA256、环境与原始GPU身份保存在该run的`protocol_lock.json`及逐模型`lock.json`、`result.json`。

```bash
python LightGenV2/demo_check/reproduction/plot_kather_seed17.py --run LightGenV2/demo_check/runs/simulation/kather2016_coverage_seed17_test_20260917 --out LightGenV2/demo_check/reports/reproduction/figures_20260917/seed17_train_test
```

输出目录须不存在。全部评估和优先调度进程已退出，检查时服务器无GPU计算进程；未继续补种子或根据本次测试结果调参。
