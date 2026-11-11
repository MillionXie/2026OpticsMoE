# T22 时间一致性公开替代模型试验

2026-11-11 用户批准独立试跑。本目录由本对话负责。参考仿真SRCC为0.8043868643，试验目标按绝对差不超过0.02记录，对应0.7843868643至0.8243868643。

## 当前架构

用户确认删除四条电子旁路、RMS融合和quality14，将router与expert换为MLP，将原Softplus位置改为ReLU，末端使用两层全连接评分。用户随后澄清，Qwen前端、merger等实际使用的模块要保持原运行结构。

2026-11-12用户已选定无偏置线性专家版本，固定入口为 `configs/selected_linear.json`。选定PT为 `runs/simulation/compact256_linear_refine_s197_20261112/LightGen.pt`，SHA256为 `acd26ecaa7a67ea7f284bec8b46a31a7a5cc6126c64f6100c350ba1bf1d00861`，大小24,140,819字节，可训练参数4,798,790。MLP消融及历史PT继续保留内部证据。

当前版本沿用空间mean/max merger和64位置的序列位置参数，视觉适配采用LN1024→Linear1024,192，并按用户要求移除早期prompt条件化。视觉分支到merger全程仅使用图像特征，文本在merger之后的明确融合点进入。router、merger和末端头的激活统一为ReLU，专家为无偏置线性层，读出采用五等级概率加权。schema6使用256维末端隐藏层和论文幅度ORM，并删除投影后LN192。历史schema3–5按其原有配置加载。

此前的prompt条件化由文本产生缩放和偏移，再作用于视觉特征。此操作已移除。当前视觉输入LayerNorm对每个token的1024维特征规范化，再投影到192维。更换prompt时，训练后模型的merger输出、帧级和视频级路由权重均逐位一致。图文在merger后的42-token拼接处开始联合处理。

逐层运行顺序如下。

1. 每视频4帧，原65%短边中心裁剪及448输入。冻结Qwen patch projection与位置编码，按原固定空间降采样得到每帧49×1024。固定Temporal prompt含38个2048维embedding。
2. 视觉适配为LN1024、Linear1024→192；投影后LN192由配对消融profile控制。归一化的均值、方差均由当前视觉token自身计算，仿射参数为视觉层自身的训练参数。
3. 帧router为LN192、Flatten9408、Linear9408→128、ReLU、Linear128→4、softmax、Top2及选中权重L1归一化。
4. 四个帧专家各为无偏置Linear192→192，逐token执行后按Top2权重相加。四专家在帧和视频槽位之间共享，视频级使用独立的一组专家参数。
5. 帧global通过Linear192→27/ReLU和token轴Linear49→27/ReLU编码。原幅值归一化后装入518整场，执行原相位调制、传播、强度探测、区域读取、强度除均值、clip8和log1p。pool49×96、LN96、ReLU和Linear96→192得到帧特征。
6. 原Frame Merger对每帧49个token作空间mean与max，拼成384维，经过LN384、Linear384→192和ReLU，再加原帧位置编码。
7. 固定prompt embedding经过独立的LN2048与Linear2048→192。四个图像token与38个prompt token在此处拼接，加原序列位置编码前42项。视频router只读四个图像token，经LN192、Flatten768、Linear768→128、ReLU、Linear128→4和Top2。
8. 四个视频专家逐token处理42个图文token并按Top2加权汇合。专家结构由同一候选的expert_mode确定，与帧级结构相同，参数独立。
9. 视频global通过Linear192→56/ReLU编码，补零至56行，执行原幅值归一化、整场传播和探测。pool42×96、LN96、ReLU与Linear96→192输出42×192。
10. 末端Flatten8064、Linear8064→256、ReLU和Linear256→5得到五档logits，经softmax后对等级分数加权求MOS。

空间mean/max属于原merger。末端时序卷积和均值、方差、差分统计已按用户要求删除。每场16视频，每视频对应一个输出。

两级global保持原17μm采样、532nm波长、10cm传播、518画布、478有效区、0.5°截止和111×111每视频相位。schema6的场调制采用alpha·D·U+(1-alpha)·U，alpha固定为2/3，相位为2π·sigmoid(raw)。旧能量混合eta=0.20对应的两项幅度比也是2∶1。零ReLU场在平方均值上先clamp再开方，保证反传有限。

478×478为中央有效区域，518×518为传播计算画布，四边各有20像素padding。16视频排布位于中央有效区域中。

## 输入到输出的完整尺寸

以下 `B` 表示光场批数，每个光场固定16个视频，每视频4帧。视频在10%、36.67%、63.33%、90%位置抽帧，帧索引为 `round(fraction * (count - 1))`。每帧取短边65%的中心正方形裁剪，转RGB并缩放到448×448。视频不足16条时循环补齐已有视频，最终只返回真实输入的评分。

四个视觉token按时间顺序对应上述四帧。每帧的49个global读出token经过原Frame Merger汇总成一个192维token，帧的数量保持4。Frame Merger沿token轴执行mean/max汇总、LN384、Linear384→192和ReLU。

### 冻结的Qwen前端与视觉适配

| 操作 | 输出尺寸 |
|---|---|
| 视频抽帧、裁剪和缩放 | `[B,16,4,3,448,448]`，uint8 |
| 合并批、视频、帧轴，像素除255、减0.5再除0.5 | `[64B,3,448,448]` |
| 每帧复制成两个相同时间切片 | `[64B,3,2,448,448]`，bf16 |
| 冻结Conv3d，3→1024，kernel与stride均为2×16×16 | `[64B,1024,1,28,28]` |
| 去单例时间轴，按原2×2分组顺序展开，加冻结位置编码 | `[64B,784,1024]` |
| 每个2×2空间组均值汇聚 | `[64B,196,1024]`，14×14 |
| 再按2×2空间组均值汇聚并恢复视频、帧轴 | `[B,16,4,49,1024]`，7×7，fp16 |
| 转fp32、LN1024、Linear1024→192、可选LN192 | `[B,16,4,49,192]` |

运行时使用提取的patch投影、位置编码与固定prompt embedding。这里的四帧各自通过patch投影，每帧的两个时间切片由该帧复制而成。

两次2×2均值汇聚均来自LG2的定制缓存流程，见 `t13_temporal_robust_training/cache_builder/cache_qwen_front.py` 的 `pool_qwen_front_tokens`。它们承担将patch特征适配到7×7、49-token输入规模的工作。Qwen原生PatchMerger采用LayerNorm和可学习的线性层完成相邻patch组映射。原Frame Merger为后续独立模块。

视觉适配的LN1024规范化输入特征，投影后的LN192规范化输出特征。两者来自原LG2，原第二次LN前还包括quality相加和prompt调制。当前以相同种子、公共参数初始化、训练配置和评价顺序，比较保留与删除第二处LN，结果及微调恢复见本轮结果。

### 帧级路由、专家与global

| 操作 | 输出尺寸 |
|---|---|
| Router LN192、展平49×192 | `[B,16,4,9408]` |
| Linear9408→128、ReLU、Linear128→4 | `[B,16,4,4]`，末轴为四专家logits |
| softmax，温度1.1，Top2掩码及选中权重和归一化 | `[B,16,4,4]`，每帧两个非零权重 |
| 每个专家无偏置Linear192→192 | 各为`[B,16,4,49,192]` |
| 按该帧Top2权重求和 | `[B,16,4,49,192]` |
| global的Linear192→27、ReLU | `[B,16,4,49,27]` |
| 转置后沿token轴Linear49→27、ReLU，再转回 | `[B,16,4,27,27]` |
| 每个27×27幅值场除以自身RMS，放入整场 | `[B,518,518]` |
| 相位调制、整场衍射传播、取复振幅模平方 | `[B,518,518]` |
| 每视频四个56×56探测区域 | `[B,16,4,56,56]` |
| 每区域除空间均值、clip上限8、log1p | `[B,16,4,56,56]` |
| AdaptiveAvgPool2d到49×96 | `[B,16,4,49,96]` |
| LN96、ReLU、Linear96→192 | `[B,16,4,49,192]` |
| 对49个读出token沿token轴分别mean与max，再拼接 | `[B,16,4,384]` |
| 原Frame Merger并加4个帧位置向量 | `[B,16,4,192]` |

四个帧专家共享于全部帧和视频槽位。代码计算四个专家输出，Top2之外的输出乘零权重。frame global将每视频的四帧排为2×2，每帧编码区27×27，统一在518×518相干光场传播。探测区到49×96的自适应均值池化是当前实际操作，宽度由56变为96。

帧级只有1个共享router和4个独立专家。一个专家的参数在所有49个token、4帧、16视频及batch之间复用。每个线性专家36,864参数，四专家147,456参数。帧router有1,205,252个参数，所有帧均复用该组参数。每帧独立计算4个路由权重，并将该帧的全部49个token送入同一Top2专家组合。

原LG2的帧router相位为 `[64,27,27]`，专家相位为 `[256,27,27]`，按视频槽位、帧位和专家编号组织。共享MLP参数是当前替代模型采用的参数组织选择。原光学expert之前先编码到27×27，expert的CCD读出回到49×192，global之前再次编码到27×27。

### 独立文本输入与视频级主干

固定prompt为 `Please evaluate the temporal quality of this video and rate it using one of the following five levels: Excellent, Good, Fair, Poor, or Bad.`。含聊天模板的固定输入对应38个2048维embedding，作为PT buffer保存。

| 操作 | 输出尺寸 |
|---|---|
| 固定prompt embedding | `[38,2048]`，fp16 |
| 转fp32、LN2048、Linear2048→192 | `[38,192]` |
| 广播到每个光场、每个视频 | `[B,16,38,192]` |
| 与4个视觉token沿token轴拼接 | `[B,16,42,192]` |
| 加序列位置参数前42项，完整参数尺寸为1×64×192 | `[B,16,42,192]` |
| 视频router读取前4个视觉token，LN192、展平 | `[B,16,768]` |
| Linear768→128、ReLU、Linear128→4 | `[B,16,4]` |
| softmax，温度1.1，Top2及选中权重和归一化 | `[B,16,4]` |
| 每个视频专家无偏置Linear192→192 | 各为`[B,16,42,192]` |
| 用每视频的Top2权重汇合42个token | `[B,16,42,192]` |
| video global的Linear192→56、ReLU | `[B,16,42,56]` |
| token轴补14行零，再除每个56×56场的RMS | `[B,16,56,56]` |
| 每视频一个56×56编码区，装入518×518整场并传播、探测 | `[B,518,518]` |
| 每视频读出115×115区域 | `[B,16,115,115]` |
| 每区域除空间均值、clip上限8、log1p | `[B,16,115,115]` |
| AdaptiveAvgPool2d到42×96 | `[B,16,42,96]` |
| LN96、ReLU、Linear96→192 | `[B,16,42,192]` |

视频级四个专家与帧级四个专家是独立的两组参数。每视频只有一组视频级路由权重，应用于该视频的全部42个token。视觉与文本在Frame Merger后的拼接点进入联合处理。两级global各有一套 `[16,111,111]` 的可训练相位参数，传播传递函数由固定物理参数计算。

### 两层电子读出与评分

| 操作 | 输出尺寸 |
|---|---|
| 展平42×192 | `[B,16,8064]` |
| Linear8064→256、ReLU | `[B,16,256]` |
| Linear256→5 | `[B,16,5]`，五等级logits |
| softmax | `[B,16,5]`，五等级概率 |
| 与5个固定等级分数点乘 | `[B,16]`，MOS |

TRAIN均值为49.7535400391、标准差为13.5796670914，用于损失尺度规范化。patch/位置buffer、prompt buffer和五个等级分数固定，其余适配层、路由、专家、global编码/读出、相位、merger、位置参数及末端两层全连接参与训练。MLP中的激活为ReLU，原Softplus位置为ReLU。

推理MOS由五档概率对等级分数的期望直接得到。训练中的回归、相关性、排序和TRAIN软标签损失使用规范化后的预测MOS与真实MOS。新增五档交叉熵监督，标签使用与Qwen baseline相同的TRAIN范围等宽分档边界。

论文Qwen baseline采用2048→5质量词logits，经softmax后对五个等级分数加权。已核验原PT的 `level_scores` 为28.62、40.79、52.96、65.13和77.30，入口为 `t06_video_quality_assessment/quality_token_batch_benchmark_200.py`。本任务复用其五档期望计算和TRAIN范围分档规则，schema6采用8064→256→5两层ReLU评分头。

选定线性PT的逐层尺寸及视觉独立性核验见 `runs/simulation/compact256_linear_refine_s197_20261112/architecture_shapes.json` 与 `independent_model_audit.json`。原始视频输入的完整558条核查及可视化见下文的开源审查记录。历史核验保留在各自run。

## 初始化和独立PT

`front.py` 定向抽取Qwen patch与position参数，保存固定prompt embedding，结果约4.9MB。独立推理加载 `LightGen.pt` 中的配置、state_dict、前端buffer、固定prompt和MOS尺度。

`--paper-initialize` 使用显式白名单导入原模型保留的视觉/文本适配层、视觉LayerNorm、merger、位置参数和两级global。每个导入张量记录原名、目标名，原PT的SHA写入run。新MLP router、expert和末端评分层单独初始化。

提取的前端参数与原缓存逐张量SHA一致。随机图像和实际视频上，独立前端与官方小组件的CPU输出均逐位相同。该视频CPU输出与历史CUDA缓存最大绝对差0.03125、平均绝对差约0.000201，原始核查记录已保留。

## 数据和评价

沿用2250 TRAIN与558 TEST，周期TEST评价用于选模，记录为开发评价。TRAIN标签用于反传。既有缓存仅向模型提供Qwen视觉特征。padding参与整场传播，从损失与指标中剔除。

原T06训练还使用了权重3.0的TRAIN教师软标签SmoothL1监督。`trial_original_merger_relu_trainsoft_s163.json` 复用此设置，严格校验教师sample_ids等于2250个TRAIN ID集合，将软标签列按名称绑定到temporal。教师文件的SHA与来源写入run。该设置仅影响训练损失。

每个run保留配置、源提交、文件SHA、环境、样本ID、逐样本预测、指标和路由统计，保存best与last。完整训练后执行16次循环换位审计并生成独立 `LightGen.pt`。

## 压缩前五等级读出与LN消融结果

四组100epoch配对试验使用相同种子、公共参数初值、训练配置与评价顺序。每对94个公共参数张量的初始化SHA完全一致，仅差投影后LN192的两个张量，共384个参数。2250 TRAIN与558 TEST开发评价身份一致。

| 种子 | 保留LN192 | 删除LN192 | 删除减保留 |
|---|---|---|---|
| 163 | 0.7642689056 | 0.7689033958 | +0.0046344902 |
| 173 | 0.7844558524 | 0.7767473117 | -0.0077085407 |
| 均值 | 0.7743623790 | 0.7728253538 | -0.0015370253 |

再从已完成的 `five_levels_ln192_keep_s173_20261111/best_checkpoint.pt` 删除投影后LN，所有其他张量逐位保留，生成明确标为未评价的初始化PT。随后分别进行50epoch恢复微调。此步骤采用已训练父权重，另行记录于配对结果之后。

| run | 最佳开发SRCC |
|---|---|
| five_levels_ln192_remove_refine_s173_20261111 | 0.7876268067 |
| five_levels_ln192_remove_refine_s181_20261111 | **0.7885892612** |

压缩前选定 `five_levels_ln192_remove_refine_s181_20261111` 的epoch20，全部六组训练已完成。该父模型包含11,888,966个可训练参数，独立PT为52,519,776字节。参考SRCC为0.8043868643，绝对差0.0157976031，进入±0.02范围。

上述SRCC对应原manifest顺序、每16视频固定分组及固定光场槽位的开发评价。该候选16次循环换位审计的平均SRCC为0.7818493842，最低为0.7727637615，完整结果见对应run的 `slot_cycle_audit.json`。公开推理涉及16视频组场、输入顺序及尾组循环补齐，使用和比较指标时须记录这些条件。

保留两层末端FC，结构为8064→1024→ReLU→5→softmax→五档期望MOS。五个等级分数与四个分档边界均由2250 TRAIN标签按baseline规则生成，已与原Qwen baseline PT逐位核对一致。训练加入权重1.0的五档CE，同时使用既有回归、相关性、排序、TRAIN软标签和路由等约束。

六组558条预测均已从CSV独立复算指标，并由导出的五档概率重算MOS，最大数值误差处于float32点乘舍入范围。选定PT通过prompt替换的视觉独立性检查、训练均值/标准差buffer替换后的MOS不变检查，以及空PYTHONPATH、从/tmp启动的原生视频推理。配对验证见 `assets/five_levels_ln_ablation_audit.json`，传输SHA清单见 `assets/five_levels_result_transfer_manifest.json`。

删除LN的权重准备入口为 `python drop_projection_norm.py --source <五档且含LN的父PT> --output <初始化PT>`。恢复微调使用 `configs/trial_five_levels_ln192_remove_refine_s181.json` 配合 `run.py --initialize <初始化PT>`，原始命令、父PT的SHA和数据身份均保存在run内。

## 2026-11-12 紧凑读出与线性专家配对试验

用户批准将末端头改为8064→256→ReLU→5，参数2,065,925。原T06实际PT的末端FC为4096→1024→1，参数4,196,353，含时序卷积等的完整读出模块为4,898,689参数。

本轮schema6的两个global直接使用论文幅度形式 `alpha*D*U + (1-alpha)*U_tilde`，取 `U_tilde=U`，训练和评价均固定alpha=2/3，对应调制与残余场幅度比2∶1。旧schema3–5仍按原能量混合定义加载，历史eta不被重解释为alpha。

非线性专家保留LN192→Linear192,384→ReLU→Linear384,192。线性对照为单层、无偏置Linear192,192。每级4个专家，两级参数独立。专家后的global输入编码、强度探测和CCD读出保留原结构。线性对照同时减少专家参数，属于替换整个专家算子的消融。

`prepare_compact_ablation.py` 从同一个已训练schema5父PT生成两份未评价初始化。按下游权重加权的神经元影响保留256个隐藏单元，仅用2250 TRAIN的前向输出拟合紧凑头的5个logit及8个线性专家。两组46个公共参数张量的初始化SHA完全一致，TEST不参与拟合。每个帧专家拟合441,000个token，每个视频专家拟合94,500个token，紧凑头拟合2250个视频。准备记录含父PT SHA、保留神经元索引、拟合误差和数据身份。

配对配置为 `configs/trial_compact256_amplitude_<mlp|linear>_s<173|181>.json`，各80epoch，两者电子/相位学习率为5e-5/6e-5，TRAIN软标签权重3。所有训练均完成。

| 专家 | 种子 | 最佳epoch | 开发SRCC | 可训练参数 |
|---|---|---|---|---|
| MLP | 173 | 0，沿用压缩初始化 | 0.7875114270 | 5,691,206 |
| MLP | 181 | 0，沿用压缩初始化 | 0.7875114270 | 5,691,206 |
| 线性 | 173 | 50 | 0.7839416688 | 4,798,790 |
| 线性 | 181 | 40 | 0.7800496005 | 4,798,790 |

两份MLP候选选择相同的epoch0权重。表中相同结果来自共同压缩初始化，后续微调均未提高该评价值。

从线性seed173的best再进行两组60epoch低率恢复。此步骤独立于上表的配对预算，使用新AdamW和cosine调度，电子/相位学习率1e-5/3e-5。

| run | TRAIN软标签权重 | 最佳epoch | 开发SRCC |
|---|---|---|---|
| compact256_linear_refine_s193_20261112 | 3 | 0，沿用父权重 | 0.7839416688 |
| compact256_linear_refine_s197_20261112 | 10 | 25 | **0.7853234272** |

MLP候选与原仿真参考的SRCC绝对差0.0168754373，线性恢复候选为0.0190634371，均在0.02范围内。评价采用同一558条原TEST开发数据、manifest顺序、每16视频分组与固定槽位。16次循环换位的平均SRCC分别为0.7816874435和0.7816959468，最低分别为0.7712375455和0.7754917768。所有选模均使用固定分组的开发评价，完整条件见metadata、history及slot_cycle_audit。线性恢复候选的最佳epoch25，最终独立PT与逐视频预测在 `runs/simulation/compact256_linear_refine_s197_20261112/`；MLP候选在 `runs/simulation/compact256_amplitude_mlp_s173_20261112/`。

六组CSV的558个ID和float32目标逐条对应manifest，SRCC、KRCC、PLCC、RMSE及MAE从CSV独立复算与保存值一致。五档概率重新计算MOS的最大误差为7.22e-6。完整配对和指标核验为 `assets/compact256_ablation_audit.json`，平台、GPU、依赖、源提交及源码SHA保存在每个run的metadata。线性恢复PT的8个专家均为无偏置192×192矩阵，读出头2,065,925参数，独立视频CLI已从/tmp、空PYTHONPATH运行成功。

专家后的global编码与CCD电子读出包含原有非线性。论文的线性光学算子作用于复振幅，强度探测取模平方，其后的LN/ReLU属于电子处理。MLP router是该替代版的既定实现，前端按现有SI的向量化描述对应。

## 历史试验

以下精度来自历史schema3的GELU版本。schema4为历史全ReLU单值回归版本，已通过前向、反向与重载合同检查。schema5为压缩前五等级期望读出版本。旧PT加载时按其历史激活恢复，已与修改前的真实0.7822584592模型逐位核验预测及两级路由一致。

早期Flatten merger原型按对应源提交封存。

| run | merger | 结果 |
|---|---|---|
| relu_s163_local_20261111 | Flatten | 80epoch完成，SRCC 0.6028183747 |
| relu_clean_s163_local_20261111 | Flatten | epoch25最佳0.6731984861，epoch28保存last时I/O异常 |
| relu_clean_refine_s163_local_20261111 | Flatten | 120epoch完成，最佳0.7354285207 |
| relu_clean_refine_s163_server_20261111 | Flatten | 120epoch完成，最佳0.7330343156 |

保存已改为临时文件写完并flush/fsync后原子替换。中断run保留，后续在新run记录父PT和新优化器身份。

schema2原merger和早期prompt条件化试验完成120epoch，最佳SRCC为0.7537028454。补回TRAIN软标签后最佳为0.7697838174，在用户要求视觉/文本分开后于epoch49停止并封存。

早期schema3的四轮试验均已完成，保持相同网络结构。

| run后缀 | 初始化 | TRAIN软标签权重 | epoch | 最佳开发SRCC |
|---|---|---|---|---|
| trainsoft_s163 | 原模型保留模块白名单 | 3.0 | 100 | 0.7722242642 |
| refine_s163 | 上一行best | 1.0 | 60 | 0.7722242642，沿用父权重 |
| distill10_s163 | 第一行best | 10.0 | 60 | 0.7785033545 |
| distill10_s173 | 第一行best | 10.0 | 60 | 0.7773062865 |

上述run均带 `separate_inputs_original_merger_relu_` 前缀和 `_server_20261111` 后缀。这四轮最佳为 `separate_inputs_original_merger_relu_distill10_s163_server_20261111`，选中epoch5，SRCC为0.7785033545。

2026-11-11用户进一步批准只通过训练方法优化精度，并要求核对完整输入输出尺寸。后续结果如下，run后缀均为 `_server_20261111`。

| run主体 | 电子/相位学习率 | TRAIN软标签权重 | epoch | 最佳开发SRCC |
|---|---|---|---|---|
| separate_inputs_lowrate_distill30_s181 | 1e-5 / 3e-5 | 30 | 60 | 0.7812283653 |
| separate_inputs_lowrate_distill10_s193 | 5e-6 / 1e-5 | 10 | 60 | 0.7798112436 |
| separate_inputs_parameter_average3 | 上两行和原0.7785模型的参数均值 | 沿用父权重 | 参数平均后评价 | 0.7797551597 |
| separate_inputs_short_distill30_s197 | 3e-6 / 1e-5 | 30 | 12 | 0.7822584592 |
| separate_inputs_short_distill60_s199 | 3e-6 / 1e-5 | 60 | 12 | 0.7815124457 |

两次lowrate试验从原0.7785模型初始化，两次short试验从lowrate的0.7812模型初始化。参数平均对96个可训练参数张量逐一核验等于三份父权重的算术平均，全部固定buffer逐张量一致，记录在该run的 `parameter_average_math_check.json`。

历史GELU版本最佳为 `separate_inputs_short_distill30_s197_server_20261111`，选中epoch1，SRCC为0.7822584592。与参考SRCC的绝对差为0.0221284051，距离0.7843868643目标下限仍差0.0021284051。所有本轮试验均已完成。

该历史run的558条CSV预测已独立复算SRCC、KRCC、PLCC、RMSE和MAE，样本ID、顺序及float32目标值与原manifest一致。完整训练后权重已执行prompt替换检查，并通过从 `/tmp`、空 `PYTHONPATH` 启动的原生视频推理。独立 `LightGen.pt` 为52,493,346字节，含11,885,250个可训练参数及所需前端buffer。对应核验记录见run内 `independent_audit.json`。

## 运行

`python -m unittest LightGenV2.tasks.t22_public_temporal_mlp.tests.test_contract -v` 检查拓扑、改变prompt时视觉特征逐位不变、原merger结构、Top2、零场梯度、两级global梯度、PT重载和原子保存。

`paths.local.json` 指定vision_cache、language_cache、manifest和提取后的front。历史schema5初始化入口为 `python run.py --paths paths.local.json --config configs/trial_five_levels_ln192_remove_s163.json --train-soft-targets <TRAIN软标签PT> --paper-initialize <原T06 PT> --out runs/simulation/<run_id> --device cuda`。

本轮准备两组schema6初始化使用 `python prepare_compact_ablation.py --paths paths.local.json --source runs/simulation/five_levels_ln192_remove_refine_s181_20261111/LightGen.pt --output-dir assets/<新初始化目录> --device cuda`。配对训练执行 `python run.py --paths paths.local.json --config configs/trial_compact256_amplitude_linear_s173.json --initialize assets/<新初始化目录>/linear.pt --train-soft-targets <TRAIN软标签PT> --out runs/simulation/<新run_id> --device cuda`，MLP对照相应选择MLP配置与mlp.pt。恢复微调使用 `configs/trial_compact256_linear_refine_s197.json`，父权重为线性seed173的LightGen.pt。实际命令及父PT SHA保存在run的metadata。

`wait_for_gpu.py` 接受任意空闲CUDA卡，要求至少6GiB空闲显存、无其他计算进程、利用率不超过5%，连续三次检查后启动。

独立视频推理使用 `python infer.py --weights <LightGen.pt> --videos <video1.mp4> <video2.mp4> --output result.json --device cuda`。

同配置权重的参数平均使用 `python average_weights.py --paths paths.local.json --checkpoints <父PT1> <父PT2> --out runs/simulation/<run_id> --device cuda`，按相同的数据、前端buffer和模型配置核验后评价，保存独立PT及逐样本结果。

本轮内部试验源代码通过私有Git bundle同步服务器，只取出本任务路径。公开GitHub远端的push留待交付确认。

## 选定线性版本的整体对应核查

本轮对照 `E:/2026LightGen2Data/投稿文件/manuscript.pdf` 与 `supplement information.pdf`，分别22页和26页。SHA和文件身份在 `assets/selected_linear_submission_source_identity.json`。该工程覆盖固定prompt条件下的LGVQ时间一致性评分。

| 项目 | 实际实现与证据 |
|---|---|
| 顺序与尺寸 | 4帧×49视觉token→帧router/线性专家/global→原merger的4个视觉token→拼接38个文本token→视频router/线性专家/global→42×192→8064→256→5 |
| 视觉与文本 | 视觉分支到merger只接收图像，文本在42-token拼接点进入。两个router均从视觉部分产生权重 |
| 动态线性加权 | 两级均满足Y=Σg_k X W_k^T，等效算子为Σg_kW_k。真实前向张量复算最大误差低于3.4e-6，支持SI S9/S10形式的对应 |
| 专家后电子处理 | global输入编码、强度探测和CCD的LN/ReLU按现有实现保留。SI线性专家算子作用于其规定的复振幅边界 |
| ORM | 与正文570–585行一致采用幅度权重alpha与1-alpha。本实现取U_tilde=U及alpha=2/3，调制后场与真实传播入口逐点相同 |
| 传播与几何 | 两次完整518×518相干传播，中央478×478有效区、四边20像素padding。16视频按4×4排列，帧级每视频4个27×27输入区，视频级每视频一个56×56输入区 |
| 探测与读出 | 帧级64个56×56探测区，视频级16个115×115探测区，区域内除均值、clip8、log1p后池化。复算CCD池化与前向值最大差低于6e-7 |
| 末端 | 两层FC为8064→256→5，中间ReLU。softmax后按Qwen baseline五档分数求MOS |
| 指标定义 | SRCC、KRCC、PLCC、RMSE、MAE与原VQA评价口径对应。相关性由原始预测计算，等级分数来自TRAIN范围 |

需要在公开说明中列明已批准的替换内容。正文83–107行及SI S3/S4描述光学探测路由，当前使用MLP路由。专家为电子无偏置192×192线性矩阵，global保持光场数值处理。SI Note 3的原系统有两轮router/expert/global、合计六次光学处理，本版本有两个global传播。公开模型的可训练参数规模为4,798,790，其评价记录与该结构绑定。论文原硬件系统的时延、能耗和规模记录对应各自原系统。

ReLU与Top2使当前路由函数呈分段形式。S9的动态线性加权可直接验证，SI S16–S21的光滑Taylor展开需在固定激活分段及固定Top2支撑的局部条件下讨论。

本轮发现并修正原生入口的尾组补齐差异。558条评价由35个光场组成，最后14条真实视频与开头2条视频共同组场。现在 `infer.py` 与缓存评价使用相同规则。1、3、15、16、17、31、32、33及558条输入的补齐身份已逐一回归。无效视频的解码器释放也有专门回归。

## 原始视频复算与光场可视化

`audit_native_inference.py` 从原始视频解码558条、共2232个真实抽帧，执行所选PT内的冻结前端与完整主干。尾组padding用有效位排除在指标之外。首次完整结果在 `runs/simulation/linear_native558_audit_20261112/`。

| 输入方式 | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---|---|---|---|---|
| 选定缓存评价 | 0.7853234272 | 0.5743142102 | 0.7900532524 | 8.7496365280 | 6.5336418767 |
| 原始视频入口 | 0.7853172110 | 0.5743013325 | 0.7900544543 | 8.7496189451 | 6.5336452607 |

原始视频和缓存方式的逐条MOS最大差0.0031852722、平均绝对差0.0002683824。冻结前端特征最大差0.03125、平均绝对差4.55e-7。缓存评价batch1与原batch4保存预测的最大差3.05e-5。原始视频SRCC与原仿真参考的绝对差为0.0190696533，在0.02范围内。

审查用统一输出目录为 `runs/simulation/linear_release_review_20261112/`。其中 `native558/` 保存最后一轮原始视频回归，`fields/` 保存从原始视频输入生成的可视化及底层数组。此前两套缓存/原生图保留在各自run，最终审查以该统一目录为入口。

`visualize_fields.py` 支持 `--videos` 原始视频入口，也支持内部的 `--paths` 缓存核查入口。原始视频方式只依赖model.py、infer.py、PyTorch、NumPy、OpenCV和Matplotlib。独立运行核查从/tmp、空PYTHONPATH开始，并禁止导入训练入口、内部data模块、原工程兼容后端和完整Qwen运行库。

每个global图展示入射幅度、SLM相位、ORM后幅度/相位、传播后强度的线性/对数显示、输出相位以及CCD预处理区域。青色方框标示478有效区。强度单位为输入RMS规范化后的任意单位，相位单位为弧度。色标范围和显示截断阈值写入manifest，NPZ保留原始复数场和未经显示截断的强度。

| 文件 | 内容 |
|---|---|
| frame_global.png / video_global.png | 两级global的完整518画布 |
| input_frames.png | 本组16视频、每视频4帧的真实裁剪输入，空间排列对应4×4光场槽位 |
| slot_detail.png | 第0槽位的4帧、4个帧级CCD区、两级路由权重、视频CCD区、五档概率及MOS |
| frame_global_arrays.npz / video_global_arrays.npz | 实际传播入口V、输出O、探测强度、相位、输入编码及CCD区域等原始数组 |
| group_predictions.csv / routing.json | 输入ID、槽位、有效位、预测和路由权重 |
| visualization_manifest.json | PT及数据SHA、源码身份、复算误差、显示参数、输出文件SHA |

可视化hook前后的MOS逐位相同。入射场按原几何从输入编码重建，并与捕获的传播入口校验。全场传播按原模型一次完成，原始O和I直接来自实际前向。第0组按原TEST顺序选取前16条，详细图固定取第0槽位。

## 工程入口与后续公开整理规划

当前源码入口为本目录。训练服务器对应 `/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t22_public_temporal_mlp/`。当前目录承载内部开发、历史消融和审查证据，选定结构由 `configs/selected_linear.json` 固定。

| 当前文件或目录 | 用途 |
|---|---|
| model.py | 模型、两级global传播、历史PT兼容加载 |
| infer.py | 原始视频抽帧、规范化、16视频组场、MOS输出 |
| visualize_fields.py | 原始视频或缓存输入的可复算光场图 |
| configs/selected_linear.json | 用户选定的线性PT、SHA、参数规模及评价条件 |
| runs/simulation/compact256_linear_refine_s197_20261112/ | 所选独立PT、训练日志、558条预测及原有审计 |
| runs/simulation/linear_release_review_20261112/ | 本轮完整原生复算及光场审查输出 |
| run.py / prepare_compact_ablation.py / configs/trial_* | 内部训练与消融复现 |
| assets/ / paths.local.json | 本机缓存、内部资产和路径配置 |

建议下一步通过固定commit的导出脚本生成 `releases/temporal_linear_reference/` 发布产物。本轮只确定规划，源码仍由当前主线任务维护。

```text
temporal_linear_reference/
  README.md
  requirements.txt
  model.py
  infer.py
  visualize_fields.py
  evaluate.py
  configs/selected_linear.json
  weights/LightGen.pt
  examples/video_list.txt
  results/reference_metrics.json
  results/reference_predictions.csv
  LICENSE
  THIRD_PARTY_NOTICES
```

公开model固定采用本次线性专家、256隐藏头和幅度ORM。发布前以所选权重的逐张量一致性、558条预测及独立像素推理进行回归。evaluate.py规划为接受用户提供的视频manifest和MOS标签，按相同分组计算五项指标。输入视频路径使用用户提供的列表，模型组件来源与Qwen版本、第三方许可说明列入发布说明。

公开PT的元数据规划为模型配置、组件来源、评价条件和权重来源身份，内部绝对路径保留在本任务原始记录。内部训练、教师记录及原项目缓存依赖保存在开发目录。当前24.1MB PT已经包含冻结patch投影、位置编码和固定prompt，推理按该PT直接运行。

公开README建议使用“temporal-quality reference implementation”，并写明MLP路由、线性专家、两个global和自身的原生视频指标，输入顺序/分组/槽位规则同时列出。历史“MLP”任务目录名继续作为内部稳定路径。
