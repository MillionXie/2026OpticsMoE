# 电子参数子空间SAM：推理网络完全不变

目标仍为完整5000张public-test平均CC≥0.87，不是结果承诺。
依据[Foret等，SAM (ICLR2021)](https://arxiv.org/abs/2010.01412)的局部最坏损失优化思想，
以小幅权重扰动后的梯度更新参数，检验是否改善仅降低训练损失时的泛化。
此处是电子参数子空间上的SAM＋AdamW，不是整篇论文/其分类数据集的复现，
论文结果不证明本SALICON配置必然有效。

## 严格固定的对照

两组都从已经完成并独立复评的`moe_alpha40_viewreg_cffn_kd2_seed42/best_checkpoint.pt`
出发：CC=0.85953132，epoch65 EMA，SHA256
`c88e1a41febc878cf87d6c80d4e27d7ac0c6bddc11d73a29497490d2e841ad73`。
复制权重、重建optimizer，不称为精确恢复旧optimizer/RNG。
不重置alpha，不重置已学空间FFN；不加入global16、13×13、GRN或两标量校准。
推理仍是冻结Qwen patch前端、两级E/O同尺度凸融合、光Router Top2、alpha≥0.4、
478 ROI/224专家/17微米/10cm与85412参数解码头。新增推理参数为0。

共同训练：10000张train、无图像增强、50轮上限、41轮起精修，EMA=.995，weight decay=.01。
LR：E1e-5，相位2e-4，router/CCD读出/头2e-5，原空间FFN5e-5；保持KD=.6和原GT损失。
仍保留20%–30%随机相干未调制分量及既有光学噪声；标准eval关闭随机光学扰动。
相同batch32/test48、seed42、AMP设置；起点/epoch1/每5轮/末轮完整测试选最高CC，
从15轮起按原平台控制降LR。没有独立validation，选择和控制使用public-test，有选择偏差。
只保留best/last；不覆盖已验证的源run。

|配置|优化区别|
|---|---|
|moe_alpha40_sam_control.yaml|SAM半径0，直接走原legacy单次训练循环|
|moe_alpha40_sam005.yaml|电子子空间SAM半径0.05，两次前向/反向后一次AdamW更新|
|moe_alpha40_sam001.yaml|同一设置，半径0.01|
|moe_alpha40_sam010.yaml|同一设置，半径0.10|

追加两档半径用于检验敏感性，不修改正在运行的0.05或控制组；四组来源、
训练预算和推理架构相同。0.05组第1轮全5000测试CC=0.86110220、控制组0.85962192，
是开展半径对照的初步依据，不是最终选定/独立复评结果，不据此宣称达到0.87。
原始控制/0.05源码commit为`66566410e2f2cae6a1359c98c340b2c7ebdbb692`。

普通续训控制组现已完成50轮，选择epoch1 EMA；重载best完整5000测试
CC=0.8596219213485717，末轮CC=0.8574401378631592。
比起点仅提高约0.00009060，后续训练未继续提高。alpha=0.43100822/0.44122910，
专家选择2337/2629/2298/2736次，无未使用专家。
全部产物保留在`moe_alpha40_sam_control_seed42`，best SHA256=
`9b3bfb8ea371e93230ae1a8c32102ca2fe1bbf5acfd21bbe7806facb567a87fc`；
`selected_checkpoint_test_evaluation.json` SHA256=
`35a10fe6b631cbe227ac7f527a7026092126e88f4b63100c72c1bf99dd2955cb`。
SAM各组仍在训练，不把控制组完成误写成整个半径消融已完成。

## 实现边界

首次计算原完整损失梯度，在electronic、saliency_head、ccd_readout和已有空间FFN组上
形成共同L2范数归一化的扰动`epsilon=.05*g/||g||`。光学相位与光Router组不加此权重扰动，
但第二次反向仍对所有可训练参数求梯度并更新，包括光学相位和Router。
alpha参数始终通过原受限sigmoid映射，不能因SAM突破0.4下限。
对同一batch复用预处理/teacher目标，恢复首次前向前的CPU/CUDA/Python/NumPy RNG，
令两次前向的光噪声与dropout实现一致；结束后的RNG推进量等于一次常规前向。
这里“相同噪声”不等于无噪声，也不修改噪声幅度/物理模型。

在第二次反向后用备份精确恢复权重，再裁剪全体梯度并只调用一次optimizer.step，
EMA钩子也只执行一次。不累加首次梯度，不给部署权重保留epsilon。
第二次异常或非有限值时恢复原权重/RNG并报错，不带着临时扰动继续训练。
拒绝含BatchNorm的模型，以免双前向悄悄双计运行统计量；当前模型不使用BatchNorm。
测试指标算法/精度不变；SAM增加训练计算量，不增加部署推理次数。
日志/历史的训练CC与loss使用首次未加SAM权重扰动的前向，
`train_sam_loss_increase`记录第二次相对首次的完整损失变化，不用它代替测试CC。

## 操作

使用GitHub已发布且测试通过的commit；按实时GPU显存选择设备，不停止他人进程。

```bash
conda activate xml
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK=LightGenV2/tasks/t03_saliency
python -m pytest "$TASK/tests" -q
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_sam_control.yaml" --phase all
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_sam005.yaml" --phase all
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_sam001.yaml" --phase all
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_sam010.yaml" --phase all
```

产物分别为任务`runs/simulation/moe_alpha40_sam_control_seed42`和
`runs/simulation/moe_alpha40_sam005_seed42`（后者sam与005之间没有下划线）。
新增半径对应`moe_alpha40_sam001_seed42`/`moe_alpha40_sam010_seed42`，也不加下划线。
检查run_manifest的commit/命令、resolved_config中的rho、初始化SHA、完整测试历史、
`selected_checkpoint_test_evaluation.json`的alpha/专家占比/实际相位更新，及best可视化。
若best仍是epoch0，必须报告未超过源权重，不能记作新训练成绩。

## 训练中的候选独立核验

2026-09-10：0.05组epoch5 EMA候选已经完成独立5000张复查（训练仍继续，非最终交付）：

|指标|独立复查值|
|---|---:|
|CC（独立NumPy float64）|0.8613320359025128|
|CC（原指标累积器）|0.8613320404052734|
|KLD|0.1127731899023056|
|SIM|0.8240760560035706|
|NSS|0.9686995490074157|
|AUC-Judd|0.7701752108567858|
|MAE|0.07696553013324738|

与同batch32独立复查的来源0.8595312563按sample_id逐图配对，2947/5000张改善；
CC差均值0.00180077965，中位数0.00155842016。KLD/SIM/NSS改善，MAE较来源0.07595093退步，
不能称为所有指标均改善或跨seed显著泛化提升；仍按public-test选模，距离0.87尚有差距。
训练内batch48评估CC=0.8613320866，两种batch的差约5e-8。

- 候选checkpoint SHA256：`5aa39e30c0c04c0411138dd83464067c73ec2d240c2391b1e71a6e74bd6215f0`。
- 独立复查commit：`e97718b5ee29d32738038eef79679563d5f3508a`，训练commit仍为66566410。
- `aligned_recheck_20260910_sam005_candidate/reproduction.json` SHA256：
  `fb31f3ecc6c4aaf5d87b5adfa0581129996949f5efa3e390fdca68048ff959c6`。
- 测试ID SHA256：`625dec6bc15b2d737d39bc252cfa0c354de217fec0266dcda568913f4a3496d0`，与原候选/Qwen一致。
- `per_image_cc.csv`保留全部逐图指标，未新增周期PT；该best路径之后可能更新，须核对SHA。

同一epoch5候选随后通过`run --phase evaluate`的完整5000张结构/路由审计：
CC=0.8613320469，alpha=0.43073767/0.44109195；四专家选择2334/2640/2318/2708次，
占比23.34%/26.40%/23.18%/27.08%，有效专家数3.98033，无未使用专家。
相对本轮初始化，router圆周相位RMS约0.000814 rad，四专家约
0.01389/0.01372/0.01495/0.01465 rad，全局相位约0.01413 rad，确实训练更新。
约46%–51%的专家相位像素变化超过0.01rad；这些变化量不代表光对精度的贡献比例。
审计run为`sam005_candidate_audit_20260910`，评估前后源checkpoint SHA一致；
验证记录在`audit_source_verified.json`。其中`selected_checkpoint_test_evaluation.json` SHA256：
`d3445205280869b96aa7f1bcf9b56cf93c925eb73cf04735bc5c640569dd6b9f`。
环境/配置/源码和命令在run内，样例在`best_visualization/saliency_examples`。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --phase evaluate --config "$TASK/configs/moe_alpha40_sam005.yaml" --checkpoint "$TASK/runs/simulation/moe_alpha40_sam005_seed42/best_checkpoint.pt" --run-dir "$TASK/runs/simulation/sam005_candidate_audit_20260910"
```

使用尚不存在的审计目录。正式交付仍须对最终选定SHA复查，不能以epoch5审计代替后续权重审计。

可用`recheck_aligned`对当前best进行完整5000张、独立float64逐图CC复算。
该工具把一次读取的checkpoint字节同时用于反序列化和SHA256计算，
不会在评估结束时误将已更新best的SHA写入旧权重的结果。
不会额外复制/保存PT。训练中复查是暂时候选，不代替最终选定best的完整复评；
若最佳权重随后更新，最终交付必须再验证最终SHA。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -m LightGenV2.tasks.t03_saliency.recheck_aligned --system optical --config "$TASK/configs/moe_alpha40_sam005.yaml" --checkpoint "$TASK/runs/simulation/moe_alpha40_sam005_seed42/best_checkpoint.pt" --run-dir "$TASK/runs/simulation/aligned_recheck_20260910_sam005_candidate" --batch-size 32
```

复查目录必须尚不存在；按实际日期/目的命名，不覆盖旧证据。若训练正同时写checkpoint，
读取可能失败，应在完整写入后重试，不把损坏/不完整文件当作有效权重。

## 较早来源加入SAM的配对训练

后期SAM0.05在epoch5到0.86133，epoch10/15为0.86090/0.86077；
故补充`moe_alpha40_viewreg_sam005.yaml`，检验在较早适应阶段加入是否更有效。
它继承**已经完成的**`moe_alpha40_viewreg_cffn_kd2.yaml`，只增加rho=.05并改run目录。
不是继承后期SAM50轮，也不同时改变增强概率、损失或电子结构。
源仍是较早0.85468765的`refine_weakaug`，SHA=de477b8c…b5eea（完整值在配置继承链）；
空间FFN恒等初始化、alpha不重置，80轮，前5轮原E冻结，前60轮原同步弱增强，
KD2→.6，61轮起关闭图像增强精修，其他物理扰动仍保留。
对应无SAM配对控制的最终CC为0.85953132；后期SAM是另一个来源，不当成严格配对。

SAM同batch两次前向仅重采相同光学噪声/dropout，图像增强在外部loader执行一次，
teacher从该batch同步变换后的缓存读取一次；没有对第二次前向另取增强图。
不产生新增推理参数/分支，正式只留best/last。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_viewreg_sam005.yaml" --phase all
```

新产物在`runs/simulation/moe_alpha40_viewreg_sam005_seed42`，在空闲资源上运行，
不得覆盖或重启现有SAM对照。最终评估仍为完整5000 public-test选模，并披露选择偏差。

## 直流分量差异的训练集小样本诊断（结果）

2026-09-10，在上述epoch5/SHA=5aa39e30候选上进行只读诊断，未优化参数：
从`legacy.build_loaders(..., training=False)`的train dataset中用
`sorted(random.Random(17042).sample(range(10000),128))`取固定128张，batch16，
无图像增强/外围autocast，使用`independent_cc`逐图float64计算。
完整模型保持eval、phase dropout关闭；仅在光分支原`_apply_coherent_zero_order`
方法调用期间临时设该分支training=True，调用后立即恢复。其余CCD噪声/dropout保持关闭。
这不代表完整噪声鲁棒性评估，只隔离相干未调制项。振幅/相位eta均原样0.2–0.3，
保留原随机相对相位；每batch三次抽样的torch seed为`17042+batch_index*3+draw_index`。

|条件|对同一GT的平均CC|与无扰动输出的平均CC|
|---|---:|---:|
|无光学扰动|0.8754117339|1|
|仅未调制分量，抽样0|0.8759857500|0.9941213238|
|仅未调制分量，抽样1|0.8756770201|0.9938578242|
|仅未调制分量，抽样2|0.8754149858|0.9940261410|

该小样本没有显示单独直流项造成明显性能损失，故暂不依据“训练含直流、测试关闭”
直接新增clean/noisy双损失，也不减少20%–30%训练未调制约束。
这不能证明真实硬件无域偏移、全部噪声无影响或其他样本结论相同；不是正式5000测试分数。

## 完整测试误差分组：下一步优化依据

使用`aligned_recheck_20260910_sam005_candidate/per_image_cc.csv`与
`aligned_recheck_20260909_qwen/per_image_cc.csv`按sample_id配对全部5000张，
总体CC仍为0.86133204与0.88968469；学生在1581张上高于教师，未筛选掉其余图片。
对原测试loader的224×224 GT执行`normalize_density`，坐标线性映射到[-1,1]，
计算密度熵`-sum(q*log(q))/log(H*W)`、密度质心到原点的距离、
以及相对该质心的二阶空间矩。每种属性独立排序等分四组，每组1250张。

|GT属性|最低四分位：Qwen−学生CC|最高四分位：Qwen−学生CC|
|---|---:|---:|
|归一化熵|0.02635898|0.02788069|
|质心离中心距离|0.02528957|0.03401852|
|空间二阶矩（分散程度）|0.02373916|0.03540363|

最高分散组学生/Qwen分别0.82871286/0.86411649；最高偏心组0.85605289/0.89007142。
这是已参与选模的public-test上的描述性诊断，不是独立验证或因果结论；
“分散”指GT密度分布，不能直接等同图像的物体数量。各组只用于分析，仍保留原完整平均分，
不按测试GT修改输入、后处理、样本权重或模型推理。
其结果更支持先观察现有全局空间混合/位置变化训练对照，不支持靠单一亮度校准缩小全部差距。

## 空间增强幅度的单变量对照

`moe_alpha40_viewreg_sam_crop90.yaml`只相对早期SAM的95%最小裁剪边长改为90%，
即随机保留90%–100%边长（最低约81%面积），再缩放回224×224。
不改变推理图像，不以GT质心选择裁剪位置；随机位置仍由训练loader原算法产生。
图像/GT密度/fixation/teacher概率图仍同步裁剪缩放，密度/teacher重新归一化，
裁掉全部fixation仍按原规则回退整图。增强不引入测试图片或新教师推理。
这是近似视图一致性训练假设；裁剪会改变场景上下文，不能保证更强增强一定改善真实注视预测。

其余来源SHA、80轮、SAM=.05、前5轮原E冻结、KD2→.6、61轮关闭图像增强、
EMA、光学噪声、光Router Top2、alpha≥.4和推理结构全部不变。
严格对照为`moe_alpha40_viewreg_sam005_seed42`；不要将其与后期SAM的不同来源直接归因比较。
只保留best/last，使用释放的显卡资源，不抢占/停止他人任务。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_viewreg_sam_crop90.yaml" --phase all
```

结果在`runs/simulation/moe_alpha40_viewreg_sam_crop90_seed42`；以完整5000张公共测试选模，
不得只报偏心/分散子集改善，仍需最终权重、相位、路由、alpha审计。

## 现有电子FFN的受限加宽对照

`moe_alpha40_sam_wide576.yaml`继承后期SAM.05的完整50轮训练，从相同已完成的
0.85953132/SHA=c88e1a41…e841ad73来源开始，只把两处既有电子CFFN的隐藏维度
384改为576：`192→576→空间DW3→GELU→192`（dropout位置不变）。
输入/输出192、两级E/O、光Router Top2、alpha≥.4、所有相位/孔径/传播距离、
85412参数读出头均不变；无新分支、attention或Transformer。

相对原CFFN合计增加151296参数（两处各75648），其中扩展DW增加3456参数，
其余为现有输入/输出线性层；每图额外约29578752 MAC，不包含激活。
这是电子容量增加，不能表述为纯训练技巧或“免费提升”。
`electronic_expansion=2`保留旧构造初始化，随后任务内显式替换为576；
有效维度以`lightgen.electronic_ffn_hidden_width=576`及结构报告`ffn_hidden_width`为准。

初始化借鉴[Net2Net/Net2Wider（ICLR2016）](https://arxiv.org/abs/1511.05641)：
复制前192个隐藏通道及对应DW核，将复制通道的原输出权重按0.4/0.6分配，
原后192通道不变。两份初始特征相同、输出权重之和等于原权重，因此eval确定性函数
仅有浮点求和误差；不等分的输出权重让隐藏梯度不同，避免完全对称复制而无法增加有效容量。
不声称dropout开启时单次随机前向等价，也不声称复现原论文全部算法。
构造不消耗额外主随机流，不重置alpha；只准八个指定张量形状变化，其他权重严格加载。
再次加载已加宽权重不能重复复制/覆盖训练结果。架构标识追加`_ffn576`。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_sam_wide576.yaml" --phase all
```

配对对照为`moe_alpha40_sam005_seed42`，不是早期SAM或不同增强版本。
仅best/last；结果尚待完整5000测试和最终审计，不替换当前已核验候选。

实现commit：`b3dc44a8437b3f432161fa4cfd247e917ea57abd`，服务器78项测试通过。
真实前4张训练图的整网迁移检查：默认matmul最高精度、cuDNN TF32开启时，
logits最大差异约0.00100386；仅为诊断关闭cuDNN TF32后差异1.90735e-6。
因此功能保持是代数/完整FP32意义，不承诺不同卷积形状下TF32逐位等价。
正式训练与测试不改变原精度设置，epoch0全5000评估仍须记录实际初始化CC。
恢复原精度后同4图执行一次SAM，确认仅一次optimizer更新，相位RMS变化约0.000198869rad，
新增可训练参数实测151296。该小batch的CC不是模型性能；未另存调试checkpoint。

## 只改变教师监督：空间相关性蒸馏

`moe_alpha40_sam_spatialcc.yaml`继承原384隐藏维度的`moe_alpha40_sam005.yaml`，
同一已完成.859531来源、SHA、50轮、SAM.05、学习率、原图训练和GT损失。
只把教师项从`KL(teacher_density || student_density)`换为每图
`1 - Pearson(student_density, teacher_density)`，系数仍固定0.6。
真实标签KL1、CC1.5、SIM.25、NSS.1及路由/相位正则均保留。
不继承加宽版，不新增参数、分支、推理教师或后处理；光学合同/alpha不变。

依据：[Huang等，Knowledge Distillation from A Stronger Teacher，NeurIPS2022/DIST](https://arxiv.org/abs/2205.10536)。
借鉴其用相关性放松教师概率精确匹配的想法；本实现是显著性空间密度上的任务适配，
不是完整DIST复现，不采用跨样本的intra-class项、不把同一坐标视为跨图语义类别。
前一轮去均值特征提示比较的是192维中间特征，本轮比较最终224×224概率图，二者不同。

教师仍来自只含10000训练ID的同一缓存，显式detach；温度固定1。
每张图独立softmax、展平空间、去均值、单位范数后计算相关性，忽略正仿射幅度差异；
先减首元素再减均值避免常数密度的浮点消减误差。常数教师无空间偏好，跳过该图教师项；
全部常数时返回可反传的0，学生常数图梯度须有限。GT项不跳过。
原KL路径逐值保持原实现；新路径只在任务内SAM训练入口启用，配置拒绝不支持的组合。
训练CSV的`map_kd`在该profile表示空间相关性损失，不再是KL；
新增`map_kd_kl_reference`仅作无梯度诊断，resolved_config和teacher_cache_provenance注明loss类型。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -m pytest "$TASK/tests" -q
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_sam_spatialcc.yaml" --phase all
```

产物`runs/simulation/moe_alpha40_sam_spatialcc_seed42`，只best/last；
选模仍完整5000 public-test，有选择偏差。它是待检验训练假设，不保证达到0.87。
