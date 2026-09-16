# RGB/SAR：同一冻结电子前端后的MoE与D2NN对照

电子只预训练一次，完整冻结后放在光学网络前面。两组使用完全相同的前端权重、归一化常数、
BatchNorm统计量及特征编码；不分别训练电子，不保留电子分类旁路。最终分类必须经过光学网络。

## 架构与训练合同

复用上一轮 `eurosat_frozen_cnn_20260916` 的共享CNN：训练20轮、验证选定第19轮；
移除dropout和128→10电子分类头，保留93,472个冻结参数。
56×56×3图像按训练集通道均值/标准差归一化，经三组Conv3×3–BN–ReLU–MaxPool，
通道32/64/128，空间28/14/7，全局平均池化后输出非负128维特征。
特征按固定通道顺序排成16×8网格，各元素复制到14×28像素单元，得到224×224振幅场，
逐样本归一化振幅平方和为1。该编码无可训练参数，会去除特征整体幅度尺度。

此后分别进入动态四支路MoE或无路由整孔径D2NN，光学相位初始化和传播模型沿用纯光学试验。
两种模型均使用478×478有效孔径、518×518完整传播画布、532nm波长、17μm像素和两段0.1m传播。
MoE四路均参与相干传播，路由能量份额确定入口功率比例；最终十个探测窗口能量归一化后直接分类。
无输出端电子残差、无alpha、无可训练电子适配器或分类头。
MoE/D2NN的可训练相位参数分别为479,364/456,968；共享前端相同不代表两种光学结构参数量或路由开销完全相同。

数据与前两轮一致：RGB/SAR空间配对子集，10类，每类每域训练300张、验证100张，合计6000/2000张。
两域同地点配对归属同一划分，训练/验证空间组无重叠，不读取测试集。
训练时仅在输入图像上水平翻转，再经过CNN；两组翻转、采样顺序和逐轮实际特征完全一致。
seed42，每模型20轮、batch32、Adam lr0.01余弦降至0.001、weight decay0、梯度裁剪1；
仅更新光学相位，最小化归一化探测概率的NLL，按验证准确率最高、其次NLL最低、再次较早轮次选模。

## 结果

| 同一冻结前端后的分类器 | RGB | SAR | 综合准确率 | 最佳光学轮次 | 无增强训练准确率 |
|---|---:|---:|---:|---:|---:|
| 动态四支路MoE | 79.50% | 74.90% | **77.20%** | 20 | 80.38% |
| 整孔径D2NN | 72.60% | 74.70% | **73.65%** | 15 | 78.58% |

MoE综合高3.55个百分点，即多判对71/2000张；其中RGB多69张，SAR多2张。
配对统计：122张仅MoE正确、51张仅D2NN正确，1422张均正确、405张均错误。
第20轮D2NN准确率为72.95%，MoE为77.20%。逐轮准确率与NLL曲线保存在run的 `validation_curves.png/pdf`。
不能将本次总体差距表述为两个域都有同等优势，也不能据一次验证集试验宣称稳定的测试优势。

同一数据、同一seed的历史结果供定位变化：

| 协议 | MoE | D2NN | MoE−D2NN |
|---|---:|---:|---:|
| 原图直接输入、纯光学独立分类 | 39.85% | 29.20% | +10.65个百分点 |
| 冻结电子分类概率与光学概率各0.5融合 | 75.85% | 75.90% | −0.05个百分点 |
| **共享冻结电子特征前置、光学独立分类** | **77.20%** | **73.65%** | **+3.55个百分点** |

原共享CNN加电子线性分类头的76.05%保留为参考，其分类头没有进入本次模型。
前置编码提高了两种光学模型的性能，本次差距也不再被输出端融合拉平；
相较纯光学试验，差距仍缩小，不能要求增加共同电子帮助后仍保留原有差距。
本次前端CNN本身有非线性，不应将整个系统的非线性全部归因于光学MoE。

## 运行与核验

训练run：`runs/simulation/eurosat_shared_frontend_20260916`。
源码commit：`8cb125cfaa4829e7714b5b23389f0c96e4844a72`；
独立核验及前端独立检查点加载commit：`cc4a0f17`。后者仅补充核验与加载支持，未重训或改变保存的权重。
GPU0 RTX4090，Python3.11.15、PyTorch2.6.0+cu124；完整依赖、配置及源码哈希见run metadata。
入口和完整结构说明见 [shared_frontend/README.md](../../shared_frontend/README.md)。

服务器工作树 `/DATA/DATA1/guest3/demo_reproduction_20260915`，实际训练命令：

```bash
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python -u LightGenV2/demo_check/shared_frontend/run.py --phase train --data /DATA/DATA1/guest3/demo_reproduction_data/eurosat/phase_only_v1/data.npz --electronic-run LightGenV2/demo_check/runs/simulation/eurosat_frozen_cnn_20260916 --out LightGenV2/demo_check/runs/simulation/eurosat_shared_frontend_20260916
```

核验均通过：原图输入的历史光学前向/梯度逐位不变；共享前端全程无梯度、权重与全部buffer不变；
两组20轮的实际输入特征字节哈希全部相同；相位具有非零梯度且发生更新；
本地CPU从CSV独立重算指标、选模和配对差异；服务器新进程加载本次保存的前端及光学检查点，
两组全部2000张验证图像的预测概率与原记录逐位一致。
核验run分别为 `runs/smoke/eurosat_shared_frontend_20260916`、
`runs/smoke/eurosat_shared_frontend_checkpoint_audit_20260916`，CPU记录为训练run内的 `independent_verification.json`。

SHA256：

- 数据：`f543d0c9ce330272063a1669ebacef082b26dc77994ec76b251f477634e350bd`
- 数据manifest：`f15e2a8e40257eb6fde0be060992a54a09f94fcf884b5226e64d580358996d70`
- 原空间划分：`cfe8373dd33cc0fe64f083b9ca32377e767c21b078c3f1d91f2dacecc25cb776`
- 来源CNN：`281523a2a62525ddf491b5937c1eaf4b2e13bbfbf76e8c4652e2bc79e28ad4fc`
- 本次去头前端：`61d3c189a18d814a22e5ffea2986cd0902c8e9b2cb4de6dbd90aae94e9ace15f`
- MoE最佳权重：`635ff1258ef3c7dc153f6ca28df78b6d65a2edc055c09a9108ebd5700c252e1c`
- D2NN最佳权重：`8863471b1a75199d8b4876a7653d82d331230ddb70d7c090cd54bbba4bfa98f9`

本地已保存前端、两份best权重、逐轮日志、逐样本概率及SHA256下载清单；服务器同时保留last权重。
旧版纯光学和输出融合的run、配置与权重均保留。
