# 2026-09-15 实际运行复现记录

2026-09-16新增：[RGB/SAR同一冻结电子前端对照](EUROSAT_SHARED_FRONTEND_20260916.md)。
电子前置、去掉分类旁路，两组20轮输入特征逐位相同；MoE验证77.20%、D2NN73.65%。差距主要来自RGB，SAR基本持平；完整冻结与保存权重复评通过。

2026-09-16新增：[RGB/SAR共享冻结电子、0.5融合实测](EUROSAT_FROZEN_ELECTRONIC_20260916.md)。
电子预训练20轮后冻结，两种光学模型各训练20轮；验证准确率电子76.05%、MoE融合75.85%、D2NN融合75.90%，未超过电子单独。冻结状态、相位更新与保存权重复评均已核验。

2026-09-16新增：[Adrenal正则化与选模诊断](ADRENAL_REGULARIZATION_20260916.md)。
完成12组配对正则化重训、2组六层分类损失对照，以及新旧24份第50轮权重复评。
六层过拟合减轻；四层低谷与早期权重选择相关。保留完整升降结果，不将事后固定轮次诊断替换原验证选模记录。

2026-09-16补齐：[Adrenal输入、损失与2/4/6层重训核查](ADRENAL_DEPTH_REPRODUCTION_20260916.md)。
seed17的三种深度、两种架构、OEO开/关共12组均完成50轮；原四/六层趋势接近复现，
六层两组OEO的最佳轮次有变化，不能宣称全数精确一致。150个相位张量均更新，分层梯度和sigmoid状态已核查。
下文“两层复现”保留其历史范围；最新完整深度范围、数值差异与证据以该报告为准。

RGB/SAR两模型独立交接包已生成并核验：[代码、交接说明与校验信息](../../pure_optical/README.md)。
仅包含动态四支路MoE和整孔径D2NN，原固定四支路消融及其他历史证据保留在工程中。

2026-09-16新增：[Adrenal分类退化与阈值复评](ADRENAL_THRESHOLD_AUDIT_20260916.md)。
确认四组模型无电子残差或α；固定权重、验证集选阈值后复评，未重新训练，详见报告。

2026-09-16新增：[EuroSAT移除电子分类支路的相位训练试验](PURE_OPTICAL_PILOT_20260916.md)。
三份原始数据包已下载并校验，沿原空间划分重建6000/2000张训练/验证子集；动态四路、固定四路、
整孔径D2NN已完成20轮训练。该新协议没有Qwen、电子残差、电子分类头或α，不能与旧混合模型数值直接比较。
下文“EuroSAT未复现准确率”的描述保留为原混合模型复现工作的状态，不适用于此新试验。

2026-09-16新增：[模型、张量、训练、公平对照与指标审查](STRUCTURE_REVIEW_20260916.md)。
该审查按“推理阶段相位固定”的最新定义解释双域预训练，并记录暂缓的九端口物理模型问题。

Adrenal 两层、seed17 的四组模型已从原初始化完整重训，各 50 轮；四组测试 AUROC
和最佳轮次均与原记录一致。EuroSAT 已通过模型运行检查，尚未复现数据集准确率。

## Adrenal 重训结果

| 模型 | 原记录 AUROC | 本轮 AUROC | 原／本轮最佳 epoch |
|---|---:|---:|---:|
| MoE＋Softsign OEO | 0.721599898740586 | 0.721599898740586 | 11／11 |
| MoE，无中间 OEO | 0.6719827858996266 | 0.6719827858996266 | 48／48 |
| D2NN＋Softsign OEO | 0.6563508638693754 | 0.6563508638693754 | 50／50 |
| D2NN，无中间 OEO | 0.5568634896525536 | 0.5568634896525536 | 50／50 |

对照对象为原导出包 `results/reports/per_seed.csv` 的 depth=2、seed=17、fixed_0.5 行，
不是五种子均值。探测面 MSE 最大绝对差为 5.321e-7；不声称权重或浮点输出逐位一致。
四组固定阈值准确率都为 229/298=76.8456%，阳性召回率都为零，与原记录一致。
数值可复现不等于分类阈值合理，也不增加关于非线性或专家分工的因果证据。

数据沿原数组轴平均投影、28→100 双三次抗混叠插值及截断，原划分
train/val/test=1188/98/298。batch=8、Adam lr=0.001、weight_decay=0、
StepLR 每10轮乘0.7，原探测面归一化 MSE；原验证 AUROC→MSE→较早轮次选模。
每组 7,450 次更新，所有相位参数均发生更新；共 29,800 次更新。
四组训练函数报告用时合计约 638 秒，不作为架构速度对比。

固定已报告的 Softsign，不重新筛选激活。本轮未重训4/6层或其他种子，
不能称为原始68次搜索及五种子统计的完整复现。12种架构配置都单独通过了前向/反向检查。

## 执行与独立核验

运行位置：服务器 `/DATA/DATA1/guest3/demo_reproduction_20260915`，只使用GPU0
RTX4090。PyTorch2.6.0+cu124、torchvision0.21.0+cu124；原报告为RTX5090、
PyTorch2.8.0+cu128。完整Python及依赖清单在run的 `metadata.json`。

Adrenal运行源码commit：`4934e200f4ec8cd6acdc37d6957334c4003f087d`。
独立入口 `../../reproduction/run_adrenal.py` 调用归档中的训练、选模和评估函数，
只重定向数据与输出位置；43个原始Python/YAML文件的SHA256均与本机归档相同。
不复制旧训练完成标记或旧权重。四组最佳检查点封存之后才读取本轮测试集。

数据SHA256：`22bc193a85c43be85f093bf12f3ffdc5ed79b5efda63925bd9e063a70594bf93`。

run ID：
- `runs/smoke/adrenal_20260915`：12个配置的有限损失、有效梯度、参数量、初始相位哈希与路由归一化。
- `runs/simulation/adrenal_L2_s17_20260915`：四组完整训练与测试。
- `runs/smoke/eurosat_20260915`：EuroSAT模型运行检查。

run相对本文件位置为 `../../runs/`。本机已保存Adrenal逐轮记录、逐样本预测、
四个best.pt和下载清单；服务器另保留last.pt。`subset_test_lock.json`包含最佳权重SHA256，
`independent_verification.json`记录独立核验：按逐样本分数秩和重算AUROC、重算准确率、
检查298个唯一样本ID、最佳权重哈希、50轮记录、7,450更新以及相位参数更新。
原始运行进程已退出，GPU0显存已释放。

实际运行命令（仓库根目录）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/reproduction/run_adrenal.py --phase smoke --data /DATA/DATA1/guest3/demo_reproduction_data/adrenal/data.npz --out /DATA/DATA1/guest3/demo_reproduction_20260915/LightGenV2/demo_check/runs/smoke/adrenal_20260915
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/reproduction/run_adrenal.py --phase train --depth 2 --seed 17 --data /DATA/DATA1/guest3/demo_reproduction_data/adrenal/data.npz --out /DATA/DATA1/guest3/demo_reproduction_20260915/LightGenV2/demo_check/runs/simulation/adrenal_L2_s17_20260915
```

再次执行须使用不同输出目录；入口拒绝覆盖现有run。

## EuroSAT 运行检查的范围

使用原模型实现和配置、服务器已有Qwen3-VL-Embedding-2B缓存、两张固定随机种子的
224×224合成RGB图像。MoE自动/均匀/已知域隔离路由和D2NN均完成前向、CE反向，
相关参数组有有限梯度，输出形状2×10；两种模型参数数目与原报告一致。
MoE路由功率和误差最大1.1921e-7。固定路由时没有路由器梯度属于预期行为。

冻结前端SHA256与原报告一致：
`3f494085f1c65fcb5c691d8fc5c048e4cb5d925cb220a1d54f2a04565fe3f477`。
347个被记录的Python/YAML/JSON文件SHA256与本机归档一致。
执行commit：`12c797e5`（完整SHA在run的 `result.json`）。

```bash
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/reproduction/smoke_eurosat.py --cache /DATA/DATA1/guest3/.cache/huggingface/hub --out /DATA/DATA1/guest3/demo_reproduction_20260915/LightGenV2/demo_check/runs/smoke/eurosat_20260915
```

本地和当前训练服务器尚未找到原实验的53,784张预处理图像、IMAGE_MANIFEST.json和
选定训练权重。服务器的VTAB-EuroSAT属于另一划分，不能替代该光学/SAR实验。
本轮没有计算EuroSAT准确率、没有重训其专家合并流程。
