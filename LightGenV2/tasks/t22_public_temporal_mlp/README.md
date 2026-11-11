# T22 时间一致性公开替代模型试验

2026-11-11 用户批准独立试跑。本目录由本对话负责。参考仿真SRCC为0.8043868643，试验目标按绝对差不超过0.02记录，对应0.7843868643至0.8243868643。

## 当前架构

用户确认删除四条电子旁路、RMS融合和quality14，将router与expert换为MLP，将原Softplus位置改为ReLU，末端使用两层全连接评分。用户随后澄清，Qwen前端、merger等实际使用的模块要保持原运行结构。

当前版本沿用空间mean/max merger和64位置的序列位置参数，视觉适配采用LN1024→Linear1024,192，并按用户要求移除早期prompt条件化。视觉分支到merger全程仅使用图像特征，文本在merger之后的明确融合点进入。当前MLP、merger和末端头的激活按用户最新要求统一为ReLU，当前读出采用五等级概率加权，schema为5。`visual_output_norm`控制投影后LN192的保留或删除，分别使用 `_five_levels_ln192` 与 `_five_levels_no_ln192` 架构身份。配对消融及删除后微调已完成，正式候选采用删除投影后LN192的版本。

此前的prompt条件化由文本产生缩放和偏移，再作用于视觉特征。此操作已移除。当前视觉输入LayerNorm对每个token的1024维特征规范化，再投影到192维。更换prompt时，训练后模型的merger输出、帧级和视频级路由权重均逐位一致。图文在merger后的42-token拼接处开始联合处理。

逐层运行顺序如下。

1. 每视频4帧，原65%短边中心裁剪及448输入。冻结Qwen patch projection与位置编码，按原固定空间降采样得到每帧49×1024。固定Temporal prompt含38个2048维embedding。
2. 视觉适配为LN1024、Linear1024→192；投影后LN192由配对消融profile控制。归一化的均值、方差均由当前视觉token自身计算，仿射参数为视觉层自身的训练参数。
3. 帧router为LN192、Flatten9408、Linear9408→128、ReLU、Linear128→4、softmax、Top2及选中权重L1归一化。
4. 四个帧专家分别为LN192、Linear192→384、ReLU、Linear384→192，逐token执行后按Top2权重相加。四专家在帧和视频槽位之间共享，视频级使用独立的一组专家参数。
5. 帧global通过Linear192→27/ReLU和token轴Linear49→27/ReLU编码。原幅值归一化后装入518整场，执行原相位调制、传播、强度探测、区域读取、强度除均值、clip8和log1p。pool49×96、LN96、ReLU和Linear96→192得到帧特征。
6. 原Frame Merger对每帧49个token作空间mean与max，拼成384维，经过LN384、Linear384→192和ReLU，再加原帧位置编码。
7. 固定prompt embedding经过独立的LN2048与Linear2048→192。四个图像token与38个prompt token在此处拼接，加原序列位置编码前42项。视频router只读四个图像token，经LN192、Flatten768、Linear768→128、ReLU、Linear128→4和Top2。
8. 四个视频专家逐token处理42个图文token并按Top2加权汇合，每个专家均为LN192、Linear192→384、ReLU、Linear384→192。
9. 视频global通过Linear192→56/ReLU编码，补零至56行，执行原幅值归一化、整场传播和探测。pool42×96、LN96、ReLU与Linear96→192输出42×192。
10. 末端Flatten8064、Linear8064→1024、ReLU和Linear1024→5得到五档logits，经softmax后对等级分数加权求MOS。

空间mean/max属于原merger。末端时序卷积和均值、方差、差分统计已按用户要求删除。每场16视频，每视频对应一个输出。

两级global保持原17μm采样、532nm波长、10cm传播、518画布、478有效区、0.5°截止和111×111每视频相位。评测相干漏光系数为0.20，相位为2π·sigmoid(raw)。零ReLU场在平方均值上先clamp再开方，保证反传有限。

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
| 每个专家LN192、Linear192→384、ReLU、Linear384→192 | 各为`[B,16,4,49,192]` |
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

帧级只有1个共享router和4个独立专家。一个专家的参数在所有49个token、4帧、16视频及batch之间复用，每个专家有148,416个参数，四专家共593,664个参数。帧router有1,205,252个参数，所有帧均复用该组参数。每帧独立计算4个路由权重，并将该帧的全部49个token送入同一Top2专家组合。

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
| 每个视频专家LN192、Linear192→384、ReLU、Linear384→192 | 各为`[B,16,42,192]` |
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
| Linear8064→1024、ReLU | `[B,16,1024]` |
| Linear1024→5 | `[B,16,5]`，五等级logits |
| softmax | `[B,16,5]`，五等级概率 |
| 与5个固定等级分数点乘 | `[B,16]`，MOS |

TRAIN均值为49.7535400391、标准差为13.5796670914，用于损失尺度规范化。patch/位置buffer、prompt buffer和五个等级分数固定，其余适配层、路由、专家、global编码/读出、相位、merger、位置参数及末端两层全连接参与训练。MLP中的激活为ReLU，原Softplus位置为ReLU。

推理MOS由五档概率对等级分数的期望直接得到。训练中的回归、相关性、排序和TRAIN软标签损失使用规范化后的预测MOS与真实MOS。新增五档交叉熵监督，标签使用与Qwen baseline相同的TRAIN范围等宽分档边界。

论文Qwen baseline采用2048→5质量词logits，经softmax后对五个等级分数加权。已核验原PT的 `level_scores` 为28.62、40.79、52.96、65.13和77.30，入口为 `t06_video_quality_assessment/quality_token_batch_benchmark_200.py`。本任务复用其五档期望计算和TRAIN范围分档规则，采用8064→1024→5两层ReLU评分头。

当前候选已通过完整原生像素输入的前向尺寸核验，见 `runs/simulation/five_levels_ln192_remove_refine_s181_20261111/architecture_shapes.json`。旧37模块核验记录保留在对应历史run。

## 初始化和独立PT

`front.py` 定向抽取Qwen patch与position参数，保存固定prompt embedding，结果约4.9MB。独立推理加载 `LightGen.pt` 中的配置、state_dict、前端buffer、固定prompt和MOS尺度。

`--paper-initialize` 使用显式白名单导入原模型保留的视觉/文本适配层、视觉LayerNorm、merger、位置参数和两级global。每个导入张量记录原名、目标名，原PT的SHA写入run。新MLP router、expert和末端评分层单独初始化。

提取的前端参数与原缓存逐张量SHA一致。随机图像和实际视频上，独立前端与官方小组件的CPU输出均逐位相同。该视频CPU输出与历史CUDA缓存最大绝对差0.03125、平均绝对差约0.000201，原始核查记录已保留。

## 数据和评价

沿用2250 TRAIN与558 TEST，周期TEST评价用于选模，记录为开发评价。TRAIN标签用于反传。既有缓存仅向模型提供Qwen视觉特征。padding参与整场传播，从损失与指标中剔除。

原T06训练还使用了权重3.0的TRAIN教师软标签SmoothL1监督。`trial_original_merger_relu_trainsoft_s163.json` 复用此设置，严格校验教师sample_ids等于2250个TRAIN ID集合，将软标签列按名称绑定到temporal。教师文件的SHA与来源写入run。该设置仅影响训练损失。

每个run保留配置、源提交、文件SHA、环境、样本ID、逐样本预测、指标和路由统计，保存best与last。完整训练后执行16次循环换位审计并生成独立 `LightGen.pt`。

## 本轮五等级读出与LN消融结果

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

当前选定 `five_levels_ln192_remove_refine_s181_20261111` 的epoch20，全部六组训练已完成。当前候选包含11,888,966个可训练参数，独立PT为52,519,776字节。参考SRCC为0.8043868643，绝对差0.0157976031，进入±0.02范围。

保留两层末端FC，结构为8064→1024→ReLU→5→softmax→五档期望MOS。五个等级分数与四个分档边界均由2250 TRAIN标签按baseline规则生成，已与原Qwen baseline PT逐位核对一致。训练加入权重1.0的五档CE，同时使用既有回归、相关性、排序、TRAIN软标签和路由等约束。

六组558条预测均已从CSV独立复算指标，并由导出的五档概率重算MOS，最大数值误差处于float32点乘舍入范围。选定PT通过prompt替换的视觉独立性检查、训练均值/标准差buffer替换后的MOS不变检查，以及空PYTHONPATH、从/tmp启动的原生视频推理。配对验证见 `assets/five_levels_ln_ablation_audit.json`，传输SHA清单见 `assets/five_levels_result_transfer_manifest.json`。

删除LN的权重准备入口为 `python drop_projection_norm.py --source <五档且含LN的父PT> --output <初始化PT>`。恢复微调使用 `configs/trial_five_levels_ln192_remove_refine_s181.json` 配合 `run.py --initialize <初始化PT>`，原始命令、父PT的SHA和数据身份均保存在run内。

## 历史试验

以下精度来自历史schema3的GELU版本。schema4为历史全ReLU单值回归版本，已通过前向、反向与重载合同检查。当前schema5为五等级期望读出版本。旧PT加载时按其历史激活恢复，已与修改前的真实0.7822584592模型逐位核验预测及两级路由一致。

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

`paths.local.json` 指定vision_cache、language_cache、manifest和提取后的front。执行 `python run.py --paths paths.local.json --config configs/trial_five_levels_ln192_remove_s163.json --train-soft-targets <TRAIN软标签PT> --paper-initialize <原T06 PT> --out runs/simulation/<run_id> --device cuda` 训练。

`wait_for_gpu.py` 接受任意空闲CUDA卡，要求至少6GiB空闲显存、无其他计算进程、利用率不超过5%，连续三次检查后启动。

独立视频推理使用 `python infer.py --weights <LightGen.pt> --videos <video1.mp4> <video2.mp4> --output result.json --device cuda`。

同配置权重的参数平均使用 `python average_weights.py --paths paths.local.json --checkpoints <父PT1> <父PT2> --out runs/simulation/<run_id> --device cuda`，按相同的数据、前端buffer和模型配置核验后评价，保存独立PT及逐样本结果。

本轮内部试验源代码通过私有Git bundle同步服务器，只取出本任务路径。公开GitHub远端的push留待交付确认。
