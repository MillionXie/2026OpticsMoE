# T01｜Caltech101物品检索：baseline复现说明

整理日期：2026-09-09。本文以冻结Qwen大模型baseline为主，说明能够重建实验的主要步骤；精确权重、运行身份和历史审查见[复现入口](README.md)及[证据说明](CHECK_20260908.md)。

## 模型与任务定义

采用预训练 **Qwen3-VL-Embedding-2B**，所有参数冻结并保持评估模式，不使用LoRA或额外训练的检索头。图像与固定检索指令经原生processor处理后，完整执行视觉和语言Transformer。取最后一个有效token的隐藏向量，截取前64个Matryoshka维度，再进行L2归一化，作为图像的检索表示。

实验使用Caltech101中的十类：airplanes、Motorbikes、Faces、Leopards、accordion、grand_piano、scorpion、sunflower、watch和yin_yang。固定划分中，每类取3张gallery图像和20张query图像，其余作为训练集；总计train 2625、gallery 30、query 200。冻结Qwen不使用训练集更新参数。图像按任务数据处理流程裁剪、缩放为224×224，所有图像使用相同指令：

> Represent this image for image-to-image object-category retrieval.

## 复现流程与评价

首先固定数据清单和划分种子42，加载同一版本Qwen权重及processor。分别提取gallery和query的64维特征；每类3张gallery图像的归一化特征取平均，再归一化，得到一个类别原型。对每张query计算与十个类别原型的余弦相似度，按相似度降序排序。

报告Top-1、Top-3和MRR：Top-K表示正确类别是否出现在前K个类别中，MRR为正确类别排名倒数的均值。因此，实际评价对象是**十个类别原型的排名**，而不是三十张gallery图像的逐图排序。冻结baseline没有训练轮数、学习率或checkpoint选模过程；不要将主方法的周期test选模规则写到这一baseline上。

实现入口为任务目录的 `run.py --profile qwen --phase evaluate`，配置为 `configs/qwen_frozen_embedding.yaml`。保留同一数据清单、模型snapshot、processor配置和逐query排名，即可复核结果。更换成完整2048维表示属于另一种baseline设置，应单独标注。

## D2NN对照与测量边界

另一个光学baseline为dense D2NN：视觉和语言各包含两张224×224相位，阶段间保留CCD检测、归一化、电子处理和重新加载，不使用router或条件专家选择。两模态合计200704个相位参数，匹配的是主方法每样本激活的专家相位预算，不包括主方法额外的router和global相位。DC20版本沿用相应电子支路和检索损失，训练40epoch、seed42，并按周期EMA test Top-1选模；历史初始化和完整设置见 `configs/d2nn_active_expert_matched_dc20.yaml`。它不是无中间探测的连续两层D2NN。

如复现Qwen速度，使用 `baseline_5090d.py`，batch1、预热50次后测200条query。从第一个原生视觉Transformer block计至归一化、相似度和排名完成；gallery预计算、文件读取、图像处理及嵌入前端不计入该模型时间。应将此时间与端到端延迟分开报告。
