# Adrenal输入、损失与深度重训核查

## 结构的明确结论

当前病例分类模型没有电子残差支路，没有α，没有Qwen，没有RGB三通道，也没有patch/token编码。
原始肾上腺三维形状掩膜28×28×28除255，沿最后一个空间轴平均成单通道28×28，
双三次抗混叠插值为100×100并截断到[0,1]，张量为`B×1×100×100`，作为振幅加载。
这是形状分类；并非原CT灰度影像分类。原始分割的3D信息已在输入准备阶段压为2D。

残差支路是另一条并行路径，其输出与主路径结果相加；病例模型不存在这种电子旁路。
OEO开启时，电子处理是主路径中的强度探测→无可训练仿射的LayerNorm→ReLU+Softsign→零相位振幅重编码。
OEO关闭时中间保留复场。路由探测和控制仍有电子处理，所以“无电子残差”与“无任何电子操作”并不等价。
可训练参数只包含相位；两层、四层、六层是独立从头训练，并在每次迭代中联合反传所有层。

## 训练究竟优化什么

CCD模拟输出是500×500强度图I。两类各有一个32×32窗口，标签y的目标图T_y在对应窗口内为1，
在其余所有位置为0。先按整场总能量把I缩放至目标总能量1024，再计算逐像素平方误差的平均：

`I_normalized = I × 1024 / (Σ_all_pixels I + 1e-8)`

`loss = 100 × mean((I_normalized − T_y)²)`

因此所谓“光斑MSE”就是让实际亮度图接近指定亮度图；MSE是均方误差，不是另一个模型。
它同时推动光落到正确位置、窗口内部亮度接近目标、背景变暗。它包含分类监督，并非错误或无关目标。
最终分类却只计算两个窗口积分E0/E1的比例，忽略各自内部光斑形状和窗口外亮度。
同为正确窗口80%、错误窗口20%，均匀光斑与单点尖峰会给出相同类别分数，却具有不同MSE。
所以MSE改善与分类改善不等价。现在先复现原损失；后续可对比窗口NLL，但不能只优化能量比而不看捕获效率。

## 阈值和AUROC分开理解

当前阳性分数为`s=(E1+ε)/(E0+E1+2ε)`，不是经过校准的真实患病概率。
代码用`s>t`判增生、否则判正常；t就是阈值，原始t=0.5。
例如一个正常样本得0.1、一个增生样本得0.3，t=0.5时都判正常，t=0.2时都能判对，分数完全未改变。
调阈值仅修改判决规则；准确率变化本身不是“学到东西”的证据，完全无区分力的分数也能随阈值改变准确率。

AUROC考察的是排序：随机选一例增生和一例正常，增生分数更高的比例，同分计半。
两层MoE+Softsign原测试69×229=15,801个跨类配对中，11,402对排序正确、无同分，
故AUROC=11402/15801=0.7215998987。它说明这批测试分数存在区分能力；不是72.16%的分类准确率，
也不是对任意新数据性能的保证，更不能据此证明训练充分。要严格归因于训练，还应与初始化/未训练模型比较。
这里测试分数范围0.02331—0.37826，全部低于0.5，因此原阈值使298例全判正常。
“阈值失效”更准确的说法是：默认0.5阈值在该分数尺度上导致零阳性预测；不代表阈值算法存在代码错误。
0.5本身也不是普遍错误：若分数是已校准的后验概率且两类误判代价相同，0.5有相应决策依据。
当前没有这样的校准证据；若更看重少数类检出或平衡准确率，应先声明目标再在验证集选阈值。
更换MSE为NLL也不保证消除多数类预测，因为类别不平衡、输入信息和可实现映射仍然存在。

原权重验证集选阈值约0.2004后，测试检出20/69增生，误报21/229正常，准确率76.51%，
AUROC仍是0.7216。完整阈值审计见[固定权重复评](ADRENAL_THRESHOLD_AUDIT_20260916.md)。
交互示意位于`../explanations/threshold-demo.html`，其中4个样本为明确标注的教学数据，不是实验样本。
定义参考[scikit-learn阈值文档](https://scikit-learn.org/stable/modules/classification_threshold.html)与
[AUROC文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)。

## 深度复现与训练诊断

两层此前已完整重训四组，本次补齐四层、六层各四组，均为seed17、50轮。
测试AUROC（本轮重训后按验证集选模）：

| 模型 | 2层 | 4层 | 6层 |
|---|---:|---:|---:|
| MoE＋Softsign OEO | 0.721600 | 0.701348 | 0.715524 |
| MoE，无中间OEO | 0.671983 | 0.684514 | 0.681223 |
| D2NN＋Softsign OEO | 0.656351 | 0.638567 | 0.632618 |
| D2NN，无中间OEO | 0.556863 | 0.619644 | 0.600658 |

与师弟原seed17固定0.5记录比较，两层四组AUROC一致，四层最大差0.0001899且最佳轮次一致。
六层MoE＋OEO的最佳轮次从原50变为47，AUROC由0.715081变为0.715524；
六层D2NN＋OEO由原50变为49，AUROC由0.634327变为0.632618。
其余两组六层AUROC和轮次一致。12组最大AUROC差0.0017088，因此是趋势接近复现，不能声称全部精确相同。
原环境RTX5090/PyTorch2.8+cu128、本轮RTX4090/PyTorch2.6+cu124；环境与浮点差异可能影响深层
分数排序和近似并列的验证选模，未通过同环境交叉实验将差异唯一归因于某个组件。
本轮未调整原1e-6 AUROC选模容差，也未为了匹配原轮次改选checkpoint。

固定0.5下，仅4层MoE无OEO及6层两组MoE出现少量阳性预测：准确率分别77.52%、77.85%、77.18%，
阳性召回分别2/69、8/69、2/69；其余9组仍全判正常，准确率76.85%。深度增加没有自动解决少数类检出。

**训练不是完全没动。**所有12组各完成7450次更新，50轮训练顺序哈希一致，150个相位参数张量均更新。
在同一批4正常/4增生训练图上，初始化、best、last共450个相位张量状态的梯度均有限且非零。
末轮各相位张量梯度L2范数最小为5.33e-5，参数相对初始化的RMS变化最小为0.04859。
所有被检查参数均无`abs(raw)>8`元素，末轮各张量sigmoid导数均值为0.2016—0.2499，
没有观察到大面积参数饱和导致的梯度消失。此检查不表示每个像素都有非零梯度，也不证明达到全局最优。

以MoE＋OEO为例：

| 深度 | 初始化训练AUROC | 第50轮训练AUROC | 最佳验证AUROC | 所选模型测试AUROC |
|---|---:|---:|---:|---:|
| 2 | 0.5194 | 0.7756 | 0.7458 | 0.7216 |
| 4 | 0.5155 | 0.8317 | 0.7745 | 0.7013 |
| 6 | 0.5121 | 0.8882 | 0.7817 | 0.7155 |

“第50轮训练”用于比较相同训练长度；其权重不一定是验证选出的best。best对应的训练AUROC分别为
0.7449、0.7423、0.8859，不混用两种权重的结果。增加深度在此设置下确实提高了末轮训练拟合能力，
最佳验证AUROC也上升，但测试AUROC不单调。这更支持检查泛化、选模稳定性与训练目标的关系，
不支持“深层没有更新所以精度没涨”的简单解释。验证集只有22个阳性，单种子结果不足以确定哪种深度更优。

12组第一轮到第50轮训练MSE均下降。MoE＋OEO为0.4287→0.1485、0.4209→0.1251、
0.4153→0.1074；D2NN＋OEO为0.4096→0.1647、0.3600→0.1516、0.3469→0.1321。
无OEO两种架构的末轮训练MSE也随深度降低，但AUROC不相应单调。损失仍缓慢下降、部分最佳轮次在末尾，
所以50轮充分收敛尚未得到证明；当前排除了明显整层未更新与大面积sigmoid饱和，未排除优化可改进。

核验及PNG/PDF曲线位于`runs/smoke/adrenal_depth_verification_20260916`；分层诊断及初始化/best/last
验证预测位于`runs/simulation/adrenal_depth_audit_20260916`。本地保留三种深度的best权重，服务器另保留last。
训练和诊断进程已退出，GPU0无计算进程，显存回到约12 MiB。

增加相位层提供更多调节自由度，但当前不同深度并非严格嵌套的函数集合：新增层包含固定距离传播，
相位为常数也不能抵消传播；D2NN相位孔径还随深度由474改为472、470，以匹配对应MoE参数总量。
无中间OEO的D2NN始终是固定线性复场变换再平方探测，不会因增加相位层自动增加非线性阶数。
有OEO时又存在归一化、截断和相位重置；即使表达能力更强、最优训练损失更低，也不保证有限训练下的测试分类指标单调。
本次不以“更深必须更好”为调参目标，也不以训练能运行或梯度非零替代充分收敛的证据。

## 证据与命令

原两层run为`runs/simulation/adrenal_L2_s17_20260915`，此次四/六层为
`runs/simulation/adrenal_L4_s17_20260916`、`runs/simulation/adrenal_L6_s17_20260916`。
均为原划分train/val/test=1188/98/298，seed17，batch8，Adam lr0.001，weight_decay0，
StepLR每10轮乘0.7，50轮，每组7450次参数更新；原验证AUROC→MSE选模、固定0.5阈值。
只固定此前已选的Softsign与OEO关闭两种设置；没有重做激活搜索或五种子统计。
测试结果已在历史报告中出现，本次是复现而不是新盲测，不根据此次测试表现改配置。
理想九端口和局部传播拼接的物理问题依旧按此前约定暂缓。

四/六层执行commit为`8d1709946bc078ccaef86d6b3198055089e13515`；源码/环境/命令见各run metadata。
GPU0 RTX4090，PyTorch2.6.0+cu124；数据SHA256为
`22bc193a85c43be85f093bf12f3ffdc5ed79b5efda63925bd9e063a70594bf93`。
各best权重摘要见`subset_test_lock.json`，last权重摘要见训练诊断。逐样本预测与下载清单留在run中。

```bash
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/reproduction/run_adrenal.py --phase train --depth 4 --seed 17 --data /DATA/DATA1/guest3/demo_reproduction_data/adrenal/data.npz --out LightGenV2/demo_check/runs/simulation/adrenal_L4_s17_20260916
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/reproduction/run_adrenal.py --phase train --depth 6 --seed 17 --data /DATA/DATA1/guest3/demo_reproduction_data/adrenal/data.npz --out LightGenV2/demo_check/runs/simulation/adrenal_L6_s17_20260916
```

诊断入口`reproduction/audit_adrenal_depth.py`，用同一批4正常/4增生训练图检查所有相位梯度，
并在完整训练/验证集比较初始化、best、last。只读原权重，反向计算不执行optimizer.step；不读取测试图像。
诊断源码commit、完整命令、输入样本ID及每层梯度/参数变化均保留在诊断run。

诊断执行commit为`c4043093`，完整SHA见该run metadata；命令为：

```bash
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/reproduction/audit_adrenal_depth.py --data /DATA/DATA1/guest3/demo_reproduction_data/adrenal/data.npz --runs LightGenV2/demo_check/runs/simulation/adrenal_L2_s17_20260915 LightGenV2/demo_check/runs/simulation/adrenal_L4_s17_20260916 LightGenV2/demo_check/runs/simulation/adrenal_L6_s17_20260916 --out LightGenV2/demo_check/runs/simulation/adrenal_depth_audit_20260916
python LightGenV2/demo_check/reproduction/verify_adrenal_depth_runs.py --runs LightGenV2/demo_check/runs/simulation/adrenal_L2_s17_20260915 LightGenV2/demo_check/runs/simulation/adrenal_L4_s17_20260916 LightGenV2/demo_check/runs/simulation/adrenal_L6_s17_20260916 --reference LightGenV2/demo_check/adrenal_softsign_code_export_20260915_145336/results/reports/per_seed.csv --out LightGenV2/demo_check/runs/smoke/adrenal_depth_verification_20260916
```
