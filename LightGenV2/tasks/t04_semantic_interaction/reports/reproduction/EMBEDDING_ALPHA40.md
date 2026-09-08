# 新合同：无语言 Transformer，两个模态均光 Router → expert → global

日期：2026-09-08。新模型性能待训练，不能用旧98%填表。

## 数据如何经过模型

1. 指令经本地 Qwen tokenizer 得到 L 个 token ID；只取冻结 `embed_tokens.weight` 对应行，得到 `[B,L,2048]`，L最多64。
   **不调用 Qwen language_model，不用 chat/TF hidden cache**。加确定性正弦位置编码，线性2048→192并归一化。
2. language 第一级：同一192维输入分别进入因果深度卷积电子残差和光支路。
   光支路先振幅编码、光 Router 相位、10cm传播、CCD四区域读数，电子softmax/Top-2选择四专家中的两个，
   再专家相位传播及CCD读出；与电子支路同尺度融合。第二级为卷积电子残差与 global 相位传播融合。
   Router不直接作为第三条质量特征支路；softmax/Top-2仍是电子，不能称整套路由全部纯光。
3. language global之后仍是 `[B,L,192]`。补零到64，由64个可学习位置系数作共享线性汇总，得到 `[B,192]`。
   权重与输入内容无关，不是attention。它保留位置差异，但仍压缩序列，不是无损读出。
   旧版是同一条指令的token逐维平均192维、逐维最大192维，拼成384维，再投影192维；不是图文拼接。
4. RGB `[B,3,224,224]` 经冻结 Qwen patch Conv3d（为temporal kernel重复同一图像两次）和位置嵌入，
   得到 `[B,196,1024]`。这里196=14×14图像patch，不是196张图。不执行任何 Vision Transformer。
   language结果经轻量线性偏置注入视觉token；视觉线性1024→192，再进入 vision 光Router→expert→global，
   每个特征传播阶段同样只并行一条深度二维卷积电子残差。
5. vision `[B,196,192]` 还原 `[B,192,14,14]`，接受文本尺度/偏置调制与二维坐标，经过3组（lean为2组）
   指令条件卷积，输出 `[B,17,6,6]` 类别logits及 `[B,6,6]` 编辑logits。另有训练辅助4类操作头，不用于生成最终网格。
6. 最后仍使用 `where(predicted_edit, predicted_category, source_grid)` 合成。
   `source_grid` 是数据提供的真实源网格，属于额外结构化输入；论文必须披露，不能声称完全RGB端到端。

## alpha、CCD和相位

四个alpha各自独立：`0.4001 + (0.95-0.4001)*sigmoid(logit)`，初始0.60，推理和训练使用同一范围。
先按有效token的样本RMS把E、O匹配到同尺度，再 `(1-alpha)E + alpha O`，最后统一恢复尺度。
这让alpha成为可解释的混合系数，但**不等于光贡献了对应百分比的准确率**；需同权重去光消融确认。
`prompt_vision_gate` 是文本注入视觉的门，不是光电融合alpha。

物理尺寸保留478逻辑ROI、17µm网格、10cm；四专家Top-2及global，双模态共2次router和4次特征传播。
训练保留20%–30%相干未调制分量、CCD噪声、相位dropout和专家负载/能量均衡；输入/相位/CCD像素移位设0。
CCD特征读出改为 `I/mean(I)` 和线性空间平均分箱，不再执行旧clip/log或像素LayerNorm+ReLU/softplus。
分箱是CCD尺寸映射，不是文本mean/max拼接。后续光电特征匹配与电子卷积仍有归一化、GELU等电子运算。

删除两模态未参与输出的顶层 `output_adapter`/`residual_logit`，不删除真实CCD到特征的线性映射。
`metrics/phase_training_audit.json` 每轮记录物理相位（不是sigmoid前raw参数）的周期差RMS及raw参数梯度；
`metrics/training_history.csv` 每轮记录四个alpha。有梯度/更新不自动等于有任务贡献。

## 公平比较与训练

三个profile均5000train/1000test，四操作均衡；v2按完整源网格+指令去重（含组内与跨组），保持原模板分布。
原始v1文件不改动。数据在任务 `dataset/openmoji_grid_v2`，训练目录在 `runs/simulation/<profile>_s73`。
本地Qwen权重与OpenMoji素材沿用旧位置，不复制进Git。token缓存带词表文件和split SHA。
任务网络全部随机初始化；不导入旧任务头或contextual教师权重。100epoch、AdamW、余弦调度、EMA，
第1/每5/末轮test按changed-cell accuracy选best，无validation，明确这是test选择偏倚协议。
只保留best和last权重，不保留每5轮PT。

主模型、lean候选和D2NN分别使用同名config；D2NN语言/视觉各两层224×224相位，
共200704参数，匹配主模型激活的专家相位参数，**不是匹配包含global/router在内的总相位参数**。
冻结Qwen+任务头baseline不应人为削弱；之前v1数字不能直接作为v2公平对照，需在v2重新训练任务头/评估。
本次不擅自引入可训练Qwen Transformer来提升主模型。

命令见任务README；CPU形状/梯度检查：

```bash
python -m pytest LightGenV2/tasks/t04_semantic_interaction/tests -q
python -m LightGenV2.tasks.t04_semantic_interaction.smoke_embedding --profile embedding_alpha40
python -m LightGenV2.tasks.t04_semantic_interaction.smoke_embedding --profile embedding_d2nn_alpha40
```

每个正式run记录 `run_manifest.json` 中Git commit、`resolved_config.json`、`split_contract.json`、
`student_architecture.json`、初始策略、best/last、训练及router审计；原始逐样本指标留在对应run，不在报告复制一份权重。
