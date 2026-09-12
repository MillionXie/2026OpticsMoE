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
SAM.05、.01和.10均已完成50轮；各组最终重载best与末轮指标必须区分。

### SAM半径配对已完成

|半径|选中epoch（EMA）|重载best，完整5000 CC|第50轮CC|
|---|---:|---:|---:|
|0，普通续训|1|0.8596219213|0.8574401379|
|0.01|5|0.8601504358|0.8585214598|
|0.05|5|0.8613320866|0.8602112183|
|0.10|1|0.8613582358|0.8585918246|

0.10比0.05仅高0.00002615，不据单seed微小差异声称可靠优越；不继续扩大半径。
0.01/0.10的50轮进程已退出，last PT的epoch均50，正式根目录均只含best/last，
各有17张PNG可视化；没有训练崩溃或只凭日志判定完成。
源码为`c78917867f5258b61d069bb49cc1f30632623938`，GPU6/A100。
0.01：alpha=.43084148/.44117665，专家槽数2336/2635/2312/2717；
0.10：alpha=.43098745/.44120982，专家槽数2340/2634/2322/2704；均无未使用专家。

`moe_alpha40_sam001_seed42` best SHA256：
`b392e8deaad4f139c165f9160483d77d105c77c7af9e56f37df9702557160ffd`；
其`selected_checkpoint_test_evaluation.json` SHA256：
`00fac06f9d4b7c8f9114a72e01b845bdf99cb6f917e0fa6e1a7e3184dbe6b3fe`。
`moe_alpha40_sam010_seed42` best SHA256：
`fa8fa65b00c786c0dc2f23a2dd032643de10f8f93484ca5372f7ba60abfabed4`；
其`selected_checkpoint_test_evaluation.json` SHA256：
`9f86a62fbc21f90836b55b07ade558a48841b19d824e7383dc3ed41f31ccd2de`。
表中为两组最终best完整重载结果。0.01未额外运行独立float64复算；0.10后来已完成该复算，
详见文末“SAM半径0.10的完成权重独立复核”，不替代.05和后续蒸馏候选已有的独立复评。
公开测试选模偏差仍适用，最高值仍未达到.87。

### SAM.05完成结果

`moe_alpha40_sam005_seed42`完成50轮，最终选择epoch5 EMA，完整5000张重载CC=
**0.8613320865631103**；末轮CC=0.8602112183，没有刷新epoch5。
最终best SHA仍为`5aa39e30c0c04c0411138dd83464067c73ec2d240c2391b1e71a6e74bd6215f0`，
与下文已独立float64复评及结构审计的字节完全一致，独立CC仍为0.8613320359。
最终`selected_checkpoint_test_evaluation.json` SHA256：
`ba304b1bd14c0559b98d2a8ef612dd9f2b7ec092951e21f972b905db0470277f`。
alpha=0.43073767/0.44109195，专家选择2334/2640/2318/2708，无明显坍缩；
KLD=0.11277319、SIM=0.82407608、NSS=0.96869959、AUC=0.77017522、MAE=0.07696550。
推理没有SAM运算或新增权重。相对普通续训控制.85962192提高.00171017，
但仍距离.87约.00866796；不把训练完成表述为目标达成。

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

## 候选独立核验（最终best字节已确认相同）

2026-09-10：0.05组epoch5 EMA先在训练期间完成独立5000张复查；
现已完成50轮，最终best SHA与此处相同，最终重载结果见上节：

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

源码commit `30145efdfa2371b3882cc39aa52e1eba882bfda3`，服务器83项测试通过。
正式启动前对同一来源的前4张训练图执行一次真实SAM检查：参数有限、仅一次optimizer更新，
相位RMS变化0.000198863rad；空间相关性教师项约0.112630，旧KL诊断约0.057244。
二者不是同量纲/同梯度预算，系数相同不意味着教师约束强度完全相同，最终以配对测试判断。
这不是4张图的模型性能报告，不保存该调试步骤的PT，也不用于测试集选择。

## 融合alpha的局部诊断（不改正式模型）

对已核验epoch5/SHA=5aa39e30…215f0候选，使用前述相同128张固定训练ID、
无图像增强和光学随机扰动，batch16，独立float64逐图CC。原alpha分别
0.43073767/0.44109195；保持全部其他参数固定，临时调用原`reset_fusion_logits`
比较两层共同alpha=0.4001及0.5，每个batch的原条件恢复精确原logit。

|条件|128张训练平均CC|相对原条件|改善图数|
|---|---:|---:|---:|
|原alpha|0.8754117339|0|—|
|两层均0.4001|0.8747182860|-0.0006934479|60/128|
|两层均0.5|0.8713075171|-0.0041042167|42/128|

源码为b3dc44a8所在工作树，torch seed42、标准eval精度；全部结束后恢复原logit，
未训练、未保存修改过的PT。不能凭此证明alpha全局最优或推断测试变化；
但它不支持“直接将alpha压至下限即可改善”的假设，故本轮未新增alpha重置/加速学习率试验。

## 现有空间卷积的受限通道交互对照

`moe_alpha40_sam_group64.yaml`继承原SAM.05，仍从已完成.85953132/SHA=c88e1a41…e841ad73开始，
相同50轮、KL教师项.6、原图训练和全部学习率，只改变已有CFFN的3×3卷积连接方式：
384通道逐通道卷积（groups384）→64组卷积，每组6通道。隐藏维度仍384，输入输出仍192；
前后线性层、GELU、dropout和层数均不变。原token mixer不变，不与wide576/global16组合。
这是**电子通道分组，不是64个光专家**；物理专家仍4个、光Router仍Top2，主结构仍E/O两支。

参考[ResNeXt，CVPR2017](https://arxiv.org/abs/1611.05431)中的分组卷积算子及连接稀疏性讨论，
不加载ResNeXt/VGG/Transformer、不引入其分支或额外主干。
该论文强调cardinality；本试验则从极端depthwise减少分组、增加每组通道交互，
不是论文实验方向的直接复现，也不以论文结果保证本任务提升。

每处卷积从384×1×3×3=3456参数变为384×6×3×3=20736，两处合计新增**34560参数**，
额外MAC约6773760/图（196 token），不把该估算当速度实测。
初始化将旧核放在每个输出通道对应的组内对角位置，其余连接为0，代数功能保持；
关闭TF32的完整FP32一致性须检查，默认GPU不同卷积算法不承诺逐位一致。
新增非对角连接通过训练学习；不额外消耗主初始化随机流，不重置alpha/相位。
严格迁移只允许两张指定卷积权重形状变化；加载已训练groups64权重不重复置零。
架构后缀`_cffn_g64`，配置字段`electronic_ffn_groups:64`；0表示维持原depthwise。
原`electronic_ffn_spatial`优化组现在包含41472参数，LR仍5e-5、weight decay仍0；
不新建优化组，不改变SAM扰动规则。读出头仍85412，光学尺寸/噪声/alpha≥.4不变。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -m pytest "$TASK/tests" -q
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_sam_group64.yaml" --phase all
```

产物在`runs/simulation/moe_alpha40_sam_group64_seed42`，只best/last。
配对对照为原SAM.05，不是同时更换教师损失的spatialcc组；完整5000测试选模偏差仍适用。

2026-09-10实现commit `6fea94fb0be7650277ea14d4ad77ecd7071d8b5a`，服务器86项CPU测试通过后发布GitHub。
实际前4张训练图的完整前向检查（关闭cuDNN TF32，仅此一致性诊断）logits最大绝对差0；
新增参数实测34560。恢复默认精度后单次SAM更新相位RMS=0.00019616647，全部参数有限。
这只是功能/更新冒烟，不是测试性能；记录在`runs/smoke/group64_real4_20260910/smoke.json`。
正式训练保持原精度，复用已结束且无活跃进程的`t03_sam`工作树，未修改其他训练的工作树。

## 空间相关性蒸馏：独立复评及50轮完成确认

`moe_alpha40_sam_spatialcc_seed42`的epoch5候选先完成全5000张独立复评，随后50轮训练已完成：

|指标|原SAM.05已完成候选|空间CC蒸馏epoch5候选|
|---|---:|---:|
|逐图float64 CC|0.8613320359|0.8617294774|
|KLD（越低越好）|0.11277319|0.11473469|
|SIM|0.82407608|0.82366582|

CC平均增益0.0003974414，2889/5000张改善，逐图差值中位数0.0002774605；
KLD/SIM略差，不能宣称所有指标同时改进。距离CC=.87仍差0.0082705226。
这是相同公开测试集选模的结果，不是未接触测试集的泛化证明；最终50轮best已确认同SHA，另起续训后仍须重新绑定SHA。
独立复评目录`aligned_recheck_20260910_spatialcc_candidate`，源码commit
`30145efdfa2371b3882cc39aa52e1eba882bfda3`，选中epoch5，实际载入checkpoint SHA256：
`a1fc626767ced7f36174a96b493233c2dcba600674b12504f37143af928897e9`。
`reproduction.json` SHA256：`654564cf97d37627f35fec1cfa9a28de3690361dd33bfc4ffe687a6241c6493c`。
test IDs SHA仍为`625dec6bc15b2d737d39bc252cfa0c354de217fec0266dcda568913f4a3496d0`。
独立float64与累积器差7.23e-10；完整逐图CC保留于该目录`per_image_cc.csv`。
训练仍为原384隐藏维度，不含wide576或group64改变；该增益只来自训练损失变化。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -m LightGenV2.tasks.t03_saliency.recheck_aligned --system optical --config "$TASK/configs/moe_alpha40_sam_spatialcc.yaml" --checkpoint "$TASK/runs/simulation/moe_alpha40_sam_spatialcc_seed42/best_checkpoint.pt" --run-dir "$TASK/runs/simulation/aligned_recheck_20260910_spatialcc_candidate" --batch-size 32
```

再次执行须使用尚不存在的复评目录，并确认best是否仍为上述SHA；不能覆盖已有证据。

同一SHA随后完成`spatialcc_candidate_audit_20260910`的完整5000张结构/路由/相位审计，
审计前后源文件SHA未变化，epoch5 EMA重载CC=0.8617294619：
alpha=0.43072805/0.44108027；四专家选择2329/2647/2314/2710次，
占全部10000个Top2选择槽23.29%/26.47%/23.14%/27.10%，有效专家数3.97938，无未使用专家。
相对初始化的相位圆周RMS：router 0.00077411 rad；四专家
0.01377231/0.01356511/0.01481157/0.01445823 rad；global 0.01397147 rad。
故相位确有更新，不是只更新电子部分。样例在该审计run的`best_visualization/saliency_examples`。
`selected_checkpoint_test_evaluation.json` SHA256：
`7d2a8daabc9e6a51de4e739a6e06df606c9229e9087c87b85aa5a79a4f46c5c5`。
审计命令与上文`sam005_candidate_audit_20260910`相同，但config/checkpoint/run-dir均换成spatialcc对应路径；
实际完整命令、30145ef源码commit、环境和架构报告保留在审计run。仍为阶段性候选，不是50轮完成报告。

## 教师梯度诊断与相关性蒸馏强度配对

2026-09-10，在固定已完成来源c88e1a41…e841ad73（CC=.85953132）上做只读训练集诊断：
源码30145ef，torch seed42，从原顺序10000张训练集中用
`sorted(random.Random(17042).sample(range(10000),128))`取128张，8个batch×16。
无图像增强，模型eval、光学随机扰动关闭，标准FP32前向，无optimizer.step、无新PT。
teacher仍为同一SHA绑定的train logits缓存。对同一次前向分别求真实标签总损失、
未加权KL教师项、未加权空间CC教师项对各原优化组参数的梯度；不混入router/物理正则项。
组内展平参数梯度，比较余弦与梯度范数比；下表是8个batch统计的均值，不是全训练集结论。

|参数组|KL梯度与GT余弦|空间CC梯度与GT余弦|空间CC梯度范数/GT（系数1）|
|---|---:|---:|---:|
|原电子残差|0.49456|0.58823|0.50255|
|光学router|0.74651|0.68398|0.54152|
|特征相位|0.52627|0.56008|0.43917|
|CCD读出|0.56695|0.62247|0.49165|
|显著性读出头|0.33337|0.51446|0.55941|
|CFFN空间核|0.56345|0.60302|0.46190|

空间CC在router组有1/8个batch余弦为负，其余5组均0/8；不能说完全无冲突。
当前系数.6对应范数比约.26–.34；系数2对应约.88–1.12。
该证据支持试一个同量级教师梯度对照，不证明强监督一定提升；训练噪声与SAM扰动下仍可能不同。
原始batch ID、损失、各组梯度统计在`runs/smoke/kd_gradient_train128_20260910/gradients.json`。

`moe_alpha40_sam_spatialcc_kd2.yaml`只把空间CC教师项固定系数.6改为2，
来源、50轮、SAM.05、GT损失、无图像增强、全部学习率、光路/alpha≥.4/DC完全不变。
不是从正在更新的spatialcc best继续，不与groups64/wide576/global16组合，推理参数增加0。
以原`moe_alpha40_sam_spatialcc_seed42`为配对；真实改善必须通过完整5000公开测试复评，
如KLD/SIM变差也必须同时报告。只保留best/last。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -m pytest "$TASK/tests" -q
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_sam_spatialcc_kd2.yaml" --phase all
```

上述kd2配置commit `961907d182a23fc3882a19139cd89b4563fe41f7`通过87项测试并push后在GPU2运行，
完整warmstart CC=.8595311981；比GPU3配对warmstart的微小数值差异须保留，不伪称逐位相同。
教师梯度诊断`gradients.json` SHA256：`fe9226e4a6d9cd7bfedb4b1de26b53a97e33f6f1b344ee0d96c7dd0856f0ea3e`。

同一128张、同一初始SHA、8×16 clean-eval条件进一步比较各GT损失分量的梯度，
源码961907d1，输出复用同一smoke目录`loss_components.json`，没有新增训练或PT。
电子组KLD/SIM/负NSS与CC项平均余弦=.87256/.86685/.91537；
特征相位组=.88542/.87709/.95506，6组参数的8个batch均未出现负余弦。
按当前权重计，负NSS梯度范数约CC项的.09–.108；没有证据支持直接删除NSS。
这只是固定来源的训练小样本诊断，不证明任意训练阶段都无冲突。
`loss_components.json` SHA256：`c65bfd5b0caa7c0e3338763fdc932a59485f75c709216304d424572934937c3d`。

## 保持前向不变的RMS完整反向对照

原同尺度融合前向：`En=E/rms(E), On=O/rms(O), M=(1-alpha)En+alpha On, F=rms(E)*M/rms(M)`。
现有实现将三处RMS统计量detach，因此使用近似梯度；这不破坏前向同尺度约束，
但与该前向函数的完整导数不同。新增训练选项`training.exact_fusion_backward:true`，
只恢复RMS统计量的梯度，**同一输入/权重的前向输出与旧实现逐值一致**。
不是取消归一化、去掉F或改变alpha；四专家光Router Top2、物理传播和CCD处理不变。

动机参考[Understanding and Improving Layer Normalization，NeurIPS2019](https://arxiv.org/abs/1911.07013)：
论文通过DetachNorm检验归一化统计量导数的作用。这里只借鉴“分离前向与反向影响”的实验方法，
不加入AdaNorm、LayerNorm模块或论文主干；不能由该论文推断SALICON一定改善。
当前分支RMS既有前向已经约束尺度，恢复导数不会额外引入可学习的分支增益。

训练时原函数在no_grad下提供原输出/诊断，加上`full-full.detach()`的零值项，仅替换反向。
eval/no_grad与去光消融仍直接调用原函数，不添加推理运算、状态字典项或参数。
完整导数实现先对均方值做epsilon²下限再开方，避免零输入sqrt导数导致NaN；
epsilon拐点采用该稳定分段导数，不主张不可微边界存在唯一导数。
单测检查前向/状态/RNG一致、随机方向有限差分、padding/零输入、去光和复制对象隔离。

`moe_alpha40_sam_exactfusion.yaml`继承原SAM.05：同一c88e来源、50轮、KL蒸馏.6、原图训练；
不同时使用spatial_cc、groups64、wide576或global16。只改变上述训练梯度，架构标识/推理合同不变。
参数增加0，alpha不重置，best/last策略不变，公开测试选模偏差仍适用。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -m pytest "$TASK/tests" -q
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_sam_exactfusion.yaml" --phase all
```

结果目录为`runs/simulation/moe_alpha40_sam_exactfusion_seed42`；原始SAM.05为配对控制。
此处只记录待验证方案，不能用功能测试替代完整5000张性能复评。

实现commit `486e00784a679f1dce68bb1930c88fdb8d1666c0`，92项测试通过并发布后运行。
前4张真实训练图与原SAM.05的初始eval输出逐值相同，参数增量0；单次SAM更新后
相位RMS变化0.00019610970，全部参数有限。记录在`runs/smoke/exactfusion_real4_20260910/smoke.json`，
只是功能/训练更新检查，不是性能结果。正式训练使用GPU1，复用无活跃进程且tracked-clean的`t03_balance`工作树。

## 强空间CC蒸馏epoch5的独立复评

`moe_alpha40_sam_spatialcc_kd2_seed42`仍在训练。epoch5的best在另一个GPU5/3090独立重载，
完整5000张float64 CC=**0.8620496019**；主训练GPU2/3090测试为0.8620496600。
浮点累积器CC=0.8620496045，与独立实现差2.63e-9。
相对系数.6候选的0.8617294774，2638/5000张改善，均值差+.0003201245，中位数+.0002148732。
距离目标.87仍差.0079503981，不应把该阶段性候选标记成目标完成或训练完成。

|完整测试指标|空间CC系数.6，epoch5|空间CC系数2，epoch5|
|---|---:|---:|
|CC64|0.8617294774|0.8620496019|
|KLD，低好|0.11473469|0.11424453|
|SIM，高好|0.82366582|0.82407411|
|NSS，高好|0.96811395|0.96543063|
|MAE，低好|0.07696690|0.08014790|

CC/KLD/SIM略改善，而NSS/MAE变差；这是同一公开测试集选模结果，不宣称全面提升或独立泛化。
复评目录`aligned_recheck_20260910_spatialcc_kd2_candidate`，源码961907d1，
epoch5实际加载checkpoint SHA256：`87ad4db51e3f58f9a41d6df09092439e5a09e81e93f88a3bfe2d3d008fafb29a`；
`reproduction.json` SHA256：`dbe91b3d3b45491bb15353807a1bec84667bc590584f9e2b62ebe7f1aecc7883`。
config SHA256：`78b70a850cff1a10b1e546475fb08dbbaa77dab37229cb4515294a1741b69251`；
test IDs SHA仍为625dec6b…a3496d0，逐图CC/实际命令/环境在同一复评目录。
该候选不含group64、wide576或RMS完整反向的新试验；仅教师项权重变化，推理结构不变。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -m LightGenV2.tasks.t03_saliency.recheck_aligned --system optical --config "$TASK/configs/moe_alpha40_sam_spatialcc_kd2.yaml" --checkpoint "$TASK/runs/simulation/moe_alpha40_sam_spatialcc_kd2_seed42/best_checkpoint.pt" --run-dir "$TASK/runs/simulation/aligned_recheck_20260910_spatialcc_kd2_candidate" --batch-size 32
```

源best仍可能被后续训练更新；复现前核对SHA，复评必须使用尚不存在的输出目录，保留原证据。

同一epoch5/SHA=87ad4db5…8fafb29a完成A100上的全5000张路由/相位审计：
`spatialcc_kd2_candidate_audit_20260910`，源码961907d1，审计前后源SHA相同，重载CC=.8620496356。
alpha=.43072182/.44106704；四专家选择槽2349/2625/2316/2710，
占23.49%/26.25%/23.16%/27.10%，有效专家数3.98147，无未使用专家。
相对初始化的圆周相位RMS：router .00076503 rad；四专家
.01376873/.01351170/.01444562/.01391805 rad；global .01377472 rad，光学相位确有更新。
样例在该run的`best_visualization/saliency_examples`；实际命令、环境和架构保留在同一目录。
`selected_checkpoint_test_evaluation.json` SHA256：
`8cbd3e76ee107164f7765efc75cc93438bd16f5107ecffe2edb399c07dfdef41`。
仍为训练中的候选审计，不是训练完成，也不是物理CCD实测或含噪测试性能。

## 固定训练子集诊断：泛化改善不等于训练拟合增强

2026-09-10，在源码961907d1上只读比较上述来源和KD2 epoch5：
固定`random.Random(17042)`从完整10000张训练清单抽取1024个索引，排序后评估。
没有优化器、参数更新、增强或随机光学扰动；标准前向，无外层autocast。
GPU为A100，torch2.6.0+cu124，batch32，worker2；每图Pearson用独立float64实现。
教师来自已有FP16训练logits缓存，转FP32 softmax后计算，不是重新运行教师主干。

|权重|训练子集CC|同子集学生/教师图PCC|教师更好的张数|完整5000公开测试CC64（先前独立复评）|
|---|---:|---:|---:|---:|
|来源c88e，epoch65|0.8768723655|0.8936876070|640/1024|0.8595312563|
|空间CC KD2 87ad，epoch5|0.8734264593|0.8974106955|662/1024|0.8620496019|
|缓存教师531c|0.8954796603|—|—|0.8896846851（完整教师重跑）|

子集训练CC降低.00344591，而完整测试提高.00251835，并且学生/教师图相关性提高。
支持“当前改进包含正则化作用、并非更强训练拟合”，不证明纯粹由某个机制造成。
训练子集与完整测试是不同样本，不能把二者差值当成精确泛化误差估计。
教师本身已用公开测试选模；缓存量化、单seed和子集采样边界也必须保留。
这不是新的性能候选，也不能据此保证放开前端或扩大电子网络一定改善。
因此不继续盲目加强裁剪或提高蒸馏系数；现有训练先按原定后期策略完成。

抽样ID SHA256：`84714f6ffcae089494d98efafba205dc49984d1fcf5bbb8744e3e0a227fca863`。
实际加载来源SHA：`c88e1a41febc878cf87d6c80d4e27d7ac0c6bddc11d73a29497490d2e841ad73`；
实际加载候选SHA：`87ad4db51e3f58f9a41d6df09092439e5a09e81e93f88a3bfe2d3d008fafb29a`。
教师源SHA、训练清单和标注SHA见上文缓存协议；本诊断没有创建额外checkpoint或修改原run。
结果由终端JSON输出记录于本节。重现时在该源码的干净工作树、已有数据/缓存下运行以下只读命令；
开始前确认best仍为上述SHA（活动训练可能更新best），否则不是同一候选。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u - <<'PY'
import hashlib, random
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import load_vision_backbone, build_student
from LightGenV2.tasks.t03_saliency.recheck_aligned import load_hashed_checkpoint
from LightGenV2.tasks.t03_saliency.reproduce_baseline import independent_cc
from LightGenV2.tasks.t03_saliency.training_support import TrainTeacherMaps
from LightGenV2.tasks.t03_saliency.run import _seed
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import training as legacy
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import prepare_salicon
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import density_from_logits
_seed(42)
p = Path('LightGenV2/tasks/t03_saliency')
s = load_settings(p/'configs/moe_alpha40_sam_spatialcc_kd2.yaml')
s.local_files_only = True; s.download = False; s.augmentation_enabled = False
s.inference_batch_size = 32; s.num_workers = 2
bundle = prepare_salicon(s, persist=False)
assert len(bundle.train_records) == 10000
teacher = TrainTeacherMaps(s, bundle.train_records)
indices = sorted(random.Random(17042).sample(range(10000), 1024))
records = tuple(bundle.train_records[i] for i in indices)
ids = [r.sample_id for r in records]
print('ids_sha256', hashlib.sha256('\n'.join(ids).encode()).hexdigest())
loader, _ = legacy.build_loaders(replace(bundle, train_records=records), s, training=False)
loaded = load_vision_backbone(s, torch.device('cuda'))
model = build_student(loaded, s)
model.eval(); model.core.set_phase_dropout_active(False)
sources = {
 'viewreg_cffn_kd2': 'c88e1a41febc878cf87d6c80d4e27d7ac0c6bddc11d73a29497490d2e841ad73',
 'sam_spatialcc_kd2': '87ad4db51e3f58f9a41d6df09092439e5a09e81e93f88a3bfe2d3d008fafb29a',
}
for name, expected in sources.items():
 payload, digest = load_hashed_checkpoint(p/'runs/simulation'/f'moe_alpha40_{name}_seed42/best_checkpoint.pt')
 assert digest == expected, 'Source changed; do not label this as the documented candidate'
 model.core.load_state_dict(payload['core'], strict=True)
 model.head.load_state_dict(payload['saliency_head'], strict=True)
 scores = []; teachers = []; alignments = []; seen = []
 with torch.inference_mode():
  for batch in loader:
   inp = legacy.preprocess_vision(loaded.processor, batch['images'], loaded.device)
   logits = model(inp['pixel_values'], inp['image_grid_thw'])[0]
   predicted = density_from_logits(logits).cpu().numpy()
   target = batch['density'].numpy()
   td = density_from_logits(teacher.get_raw(batch['sample_ids'])).numpy()
   scores.extend(independent_cc(predicted, target).tolist())
   teachers.extend(independent_cc(td, target).tolist())
   alignments.extend(independent_cc(predicted, td).tolist())
   seen.extend(batch['sample_ids'])
 assert seen == ids
 print(name, digest, 'CC', np.mean(scores), 'teacher_CC', np.mean(teachers),
       'student_teacher_PCC', np.mean(alignments),
       'teacher_better', np.sum(np.array(teachers) > scores))
model.restore_native()
PY
```

## SAM半径0.10的完成权重独立复核

`moe_alpha40_sam010_seed42`已完成50轮，best为epoch1 EMA，SHA为
`fa8fa65b00c786c0dc2f23a2dd032643de10f8f93484ca5372f7ba60abfabed4`。
2026-09-10使用源码961907d1、A100、batch32对全部5000张独立重载，
float64 CC=**0.8613582183**，与累积器差1.80e-10；原最终重载值0.8613582358得到核实。
KLD=.1130358380、SIM=.8240352064、NSS=.9677864185、AUC=.7700704916、MAE=.0772304476。
目录`aligned_recheck_20260910_sam010_completed`；`reproduction.json` SHA256：
`725a224a70d21bb504f8fd4eebd59815e0d5bb0d51447f37626f896c97b3244e`。
数据ID SHA仍为625dec6b…a3496d0，完整命令/环境/逐图CC在该目录。

与SAM.05已独立核验的相同5000个ID配对：2477张改善，平均差+.0000261824，
中位差−.0000386759。对ID排序后的差值，用NumPy `default_rng(17042)`有放回抽5000张、
重复2000次的均值2.5%/97.5%分位数为[−.0001614617, +.0002305905]。
这是固定权重、该数据集上的描述性重采样，不包含训练seed方差，也不校正公开测试选模偏差；
不能包装成无偏显著性检验或等效性证明。没有支持继续扩大SAM半径的明确证据，
不因此替换原候选；全任务最佳已核验训练候选仍为KD2的.86204960，目标.87未达到。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -m LightGenV2.tasks.t03_saliency.recheck_aligned --system optical --config "$TASK/configs/moe_alpha40_sam010.yaml" --checkpoint "$TASK/runs/simulation/moe_alpha40_sam010_seed42/best_checkpoint.pt" --run-dir "$TASK/runs/simulation/aligned_recheck_20260910_sam010_completed" --batch-size 32
```

复现前核对权重SHA；复评目录必须尚不存在，不能覆盖原证据。

## 强空间CC蒸馏与已有rank16空间混合的配对试验

前述早期`viewreg_global16`比同预算无global控制提高约.000385，未超过后来的SAM方案。
新增配置`moe_alpha40_sam_spatialcc_kd2_global16.yaml`只检验已有rank16算子在当前较强训练下的效果，
不是扩大rank、增加E/O主分支或解冻Qwen。对应控制为`moe_alpha40_sam_spatialcc_kd2.yaml`。
两者同一完成来源c88e、50轮、SAM.05、空间CC KD2、原图训练、原GT损失、EMA及其他组学习率。
新增全局投影组LR=5e-5、weight decay=0，与SAM电子扰动组共同参与训练；
因此参数化和该组的梯度预算不同，不能称“推理完全相同的纯训练技巧”。

沿用已审计的`SpatialTokenLinear`：在两个现有E的pointwise内部，
真实14×14空间上逐通道196→16→196，初始down为4×4 DCT基、up为0，之后均可学习。
不是attention或新增完整MLP-Mixer主干；两层合计增加12544参数、约240.8万MAC/图。
没有跨样本混合。现有CFFN/384隐藏维度、85412参数读出头、478 ROI、224专家、
光router Top2、两级同尺度融合、alpha≥.4和训练20%–30%未调制分量都不变。
来源CFFN严格加载，只允许新增四个投影张量；零up初始保持原函数，不重置alpha。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -m pytest "$TASK/tests" -q
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_sam_spatialcc_kd2_global16.yaml" --phase all
```

正式run只保留best/last，结果以完整5000公开测试为准，选模偏差照常披露。
此节是待验证方案，不代表达到.87或已经可替换实验室候选；前端仍完全冻结。

配置/合同测试源码`066ed0c46874e3611bb062f76503e6e925b7bbc8`通过93项测试并push后，
在A100/GPU6启动。复用已无活跃进程且tracked-clean的`t03_sam_radius`工作树，
不更新或重启其他仍在训练的工作树。
启动前同源码的4张真实训练图短检查：初始logits最大差0，参数增量12544；
单次SAM更新后的相位RMS变化0.00018743190 rad，两张global up各3136个元素均已非零，
所有参数有限。该临时内存检查未保存PT，不是性能测量，也不替代完整warmstart/测试复评。

## 空间CC系数.6的最终训练收尾

`moe_alpha40_sam_spatialcc_seed42`完成50/50轮，进程2523257已退出，无异常重启。
最终best仍为epoch5 EMA，SHA `a1fc626767ced7f36174a96b493233c2dcba600674b12504f37143af928897e9`，
与前述独立复评及相位/路由审计字节一致；最终全5000重载CC=.8617294619，
独立float64仍为.8617294774。第50轮测试=.8607004927，不可当成best。
正式根目录只有best/last两份PT，训练报告确认stop_reason=epoch_budget，completed_epochs=50。
`training_report.json` SHA256：`fd1a01e416db4cd75c764229f397363092c9980fe859587f0e74fcaad0b7f73e`；
`selected_checkpoint_test_evaluation.json` SHA256：`7d2a8daabc9e6a51de4e739a6e06df606c9229e9087c87b85aa5a79a4f46c5c5`。
训练源码仍为30145ef，最终报告不改变先前逐图改善及KLD/SIM取舍结论。
这是较高CC的已完成、已独立核验对照；KD2随后完成，收尾见文末，.87目标没有达到。

## 576维电子CFFN加宽对照完成：不采用

`moe_alpha40_sam_wide576_seed42`完成50轮，进程2516910已退出；源代码b3dc44a8。
best为epoch5 EMA，全5000张最终重载CC=.8611629393，第50轮CC=.8601916033。
相比原384维SAM.05的.8613320866，没有改善CC，却增加151296参数，故不采用该加宽结构。
这组是训练结束的标准完整重载，未单独跑NumPy float64复核，不宣称跨seed显著优劣。
KLD=.1128821728、SIM=.8239618841、NSS=.9685376445、AUC=.7701449018、MAE=.0770464810。
best SHA256：`debcab2b8568b9eac88927a0ba3476f7e8b128d760088f1a04901a218ef8db73`；
`training_report.json`：`0258e3ba06d2f8724bed497055caa56262c8fa1ffd55451961f618d173633a88`；
`selected_checkpoint_test_evaluation.json`：`b444b66a86c67c80e4cef4389e9c4685f35ad3885b1101e3b0a817b33b1ee167`。
保留原run及best/last，不删除中间证据、不替换原384维候选。

## 较早阶段SAM完成：有收益，但未超过后期精修

`moe_alpha40_viewreg_sam005_seed42`已完成80/80轮，源码67454db1，未重启或改变预算。
best为epoch75 EMA，完整5000张最终重载CC=.8602510413；第80轮为.8601740419。
这比同一较早来源、相同增强/蒸馏阶段的非SAM控制`.85953132`约高.000720，
但低于后期空间CC KD2精修的独立CC=.86204960；不能将不同来源的差直接解释为SAM启动时间的因果效应。
这是单seed、公开测试选模的标准最终重载，未另做NumPy float64复核，不宣称无偏泛化结论。
KLD=.1133814178、SIM=.8233874903、NSS=.9670447325、AUC=.7699143971、MAE=.0781757564。
alpha=.42788461/.43927202，四专家选择2336/2646/2297/2721（23.36%/26.46%/22.97%/27.21%），
有效专家数3.978，无未使用专家。相对初始化的圆周相位RMS：router=.0100613 rad，
四专家=.0973789/.1003972/.1603473/.1563913 rad，全局相位=.1303844 rad。
相位有实际更新、路由未明显坍缩，但该证据不能排除其他表达能力或优化瓶颈。
训练后期61轮起关闭图像增强，训练光学扰动继续保留；标准eval仍关闭随机光学扰动。

best SHA256：`ed5e1cce1412c3a2e6270a47d800ea85396c1c213fc2bfb04b3f1c452f973354`；
`training_report.json`：`480cca7d273becea3f6543350acac0632ebf79c596fb3939b33dfaf89eddbd8d`；
`selected_checkpoint_test_evaluation.json`：`2e1f198864d8bb93264dd17552c6fe3914872c62a1a6e0ce7b7d4f414b11fb96`。
路径及完整训练命令见本文件“较早来源加入SAM的配对训练”；正式权重仅best/last，保留该结果作为对照。

## 空间CC系数2最终完成：当前最佳已核验候选

`moe_alpha40_sam_spatialcc_kd2_seed42`完成50/50轮，源码961907d1，进程2540050正常退出。
最终best仍为epoch5 EMA，SHA `87ad4db51e3f58f9a41d6df09092439e5a09e81e93f88a3bfe2d3d008fafb29a`，
与前述独立float64复评、完整相位/路由审计及去光配对试验完全一致，无需把阶段性候选冒充完成结果。
最终全5000重载CC=.8620496600，独立CC=.8620496019；第50轮CC=.8611217465，不是best。
最终报告KLD=.1142445008、SIM=.8240741602、NSS=.9654307341、AUC=.7699777850、MAE=.0801479521。
alpha=.4307218194/.4410670400，专家计数2349/2625/2316/2710，与先前审计一致。
`training_report.json` SHA256：`f283f170c4a920e6419174019c1585bf29ac8c0b7cc527f170edf481da0d6caa`；
`selected_checkpoint_test_evaluation.json`：`be7bf973da31d5021db8fe584a83a32b2b54dae213bf5914c359cdee0b489678`。
stop_reason=epoch_budget，正式根目录仅best/last；独立复评SHA、命令、5000个ID及指标取舍见前文。
现在将该完成权重作为当前最高已核验CC候选；距离.87还差.00795040，仍不代表目标达成或实测CCD结果。
全局rank16组合仍是另外在训的试验，不混入这个权重或结果。

## groups64空间卷积完成：不采用

`moe_alpha40_sam_group64_seed42`完成50/50轮，源码6fea94fb，best epoch5 EMA，
最终完整5000张重载CC=.8613588051。相对原groups384/SAM.05的.8613320866仅高.00002672，
却新增34560参数，没有足够收益支持采用；不把单seed极小差当作可靠改进。
尚未另做float64独立复查。KLD=.1127691746、SIM=.8240855747、NSS=.9687701208、
AUC=.7701766212、MAE=.0769239346；alpha=.43073791/.44109234，专家计数2329/2645/2314/2712。
best SHA256：`b25a2d6d81483c9ec80f01bb163b1fd3db638fb2caeef1b8e3d29d22d11103f4`；
`training_report.json`：`d5d1a819ede8f269f20cf6ec8420c798eb7aee8e84fac7d94b85d31d6a87b1a8`；
`selected_checkpoint_test_evaluation.json`：`9cdfccd59fa75295a63410538965c3428f14ddf194211c940eb88ebc6e551d56`。
保留完整复现证据和best/last，不替换当前原384维、depthwise的空间CC KD2候选。

## rank16全局算子只读诊断：确有作用，尚无扩容依据

2026-09-10在源码e5e1ded1上检查在训global16版本的epoch5 EMA，实际加载SHA256：
`e3d2e666dc666f57d581a9afd4cfecaed4603c7df513ced21fe5af5dee05f75d`。
它的训练周期完整测试CC=.8621131796，尚未独立float64复评或完成50轮，不替换前述完成候选。
两层`spatial_up`分别为196×16；以float64做SVD，平方奇异值累计90%需要12/9个方向，
stable rank（平方Frobenius范数/最大奇异值平方）为4.0110/1.7678。
这只是权重谱描述，算子中有GELU，不能把该数值当作实际特征秩或直接判断容量已满/未满。

随后在A100、batch16、eval、无增强/随机光学扰动/外层AMP下，只读评估128张训练图。
抽样方法沿用本文件train1024诊断，但数量改128：`sorted(random.Random(17042).sample(range(10000),128))`，
对应ID SHA256为`a658dcec4a561ae342b0968d56fbe5eb9ddbe493dd4ef4e051c4361e9e623695`。
加载上述global16配置及同SHA权重，依次在同一batch运行原模型与将两张`spatial_up`临时置零的模型；
后者仅旁路新增全局混合，**不旁路光路或原电子残差**。每批先恢复原up，结束恢复原值，不保存PT。
按既有`independent_cc(density_from_logits(logits), density)`计算逐图CC：
原模型train128 CC=.8732422430，旁路新增算子=.8729466976，差+.0002955454，81/128张改善。
在正常前向的两处`token_pointwise`输入hook上，按该类相同真实14×14坐标计算
`U = spatial_up(GELU(spatial_down(X)))`，每样本统计`RMS(U)/RMS(X)`再平均：
两层均值=.04359135/.11069392，中位数=.04320111/.10995670。
均值的计算在channel和196空间维上，分母下限1e-12；不是相位、光强或alpha。

诊断证明新增算子有非零特征修正，但不能据此断言扩容一定无效或一定有效；
128张训练图不是完整测试分数，也不能证明泛化提升。现有完整测试增益很小，
因此先完成原预算，不据此扩大rank或增加其他电子结构。原run和权重未修改。

## 较早阶段空间CC蒸馏：只改训练损失的配对试验

已完成早期SAM/KL控制best=.86025104，后期空间CC/KD2候选=.86204960；
此前未检验在较早适应阶段就采用空间CC监督。本次只检验这一时点下的蒸馏损失差异，
不据跨来源分数推断效果。配置`moe_alpha40_viewreg_sam_spatialcc.yaml`继承已完成
`moe_alpha40_viewreg_sam005.yaml`，仅`distillation.loss: spatial_cc`不同。
对应控制是早期SAM/KL，不是后期固定KD2，也不是新增global16版本。

来源仍为`refine_weakaug`的de477b8c…b5eea（完整SHA在继承配置）；80轮、SAM.05、
教师权重2→.6至epoch60、原GT损失、EMA.995、前5轮旧E冻结、61轮起关闭图像增强，
以及全部优化器分组/学习率/平台控制均保持早期SAM协议。只有原CFFN/384维，
无global16、加宽、groups64或RMS反向改动；不解冻Qwen前端、不增加推理参数。
光router Top2、alpha≥.4、478 ROI/224专家、两级融合及训练DC20–30%不变。
相同训练视图对RGB、GT密度、fixation和缓存教师密度同步变换；教师密度重新归一化后
取log送入已有空间CC损失，不对logits直接裁剪插值。缓存教师对增强的等变性仍是近似假设，
不能将其描述为在线重新运行Qwen。标准完整测试不增强、不加随机光学噪声。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -m pytest "$TASK/tests" -q
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_viewreg_sam_spatialcc.yaml" --phase all
```

正式输出`runs/simulation/moe_alpha40_viewreg_sam_spatialcc_seed42`，仍只best/last。
完整5000张公开测试参与选模，必须披露偏差。本节是待验证方案，不是性能承诺；
0.87目标尚未达到，当前完成best仍为前述不新增参数的后期空间CC/KD2。

源码`ed7aa18e3b3f081770a6b9e16e0d08a92f774298`通过101项CPU回归并push后，在GPU3/4090启动，
复用已无活动进程的`t03_sam`工作树；训练PID2633439，不改变其他活动作业。
实际初始化报告确认de477b8c…b5eea、仅加入identity CFFN、未重置融合alpha，
完整5000张warmstart CC=.8546877293，与同源控制一致（GPU微小浮点差异）。
训练环境torch2.6.0+cu124、Python3.11、HF离线，命令/源码/环境保存在run；尚无完成成绩。
旧保存器残留的PCK元数据文字勘误见复现README，实际T03按CC选模；本次运行不为说明字段重启。

## 逐图教师/真值梯度诊断：不按教师分数简单关闭蒸馏

2026-09-10源码0f3c33d1，固定完成权重SHA=87ad4db5…08fafb29a，使用前述train128清单
（ID SHA=a658dcec4a561ae342b0968d56fbe5eb9ddbe493dd4ef4e051c4361e9e623695）。
A100、batch16、原图、eval关闭随机光学扰动、无外层AMP；同一已验证train-only教师缓存。
模型前向置于`no_grad`，随后仅使输出logits重新`requires_grad`，不反传或更新模型权重。
分别以`objectives.saliency_loss(z,target,fixation,s,teacher_logits=None)`和
`spatial_correlation_distillation(z,teacher_logits)`计算损失，对z调用`torch.autograd.grad`；
按样本flatten后求两梯度余弦（eps=1e-20），不是参数梯度，也不是CC分数本身。
真值损失仍KL1+CC1.5+SIM.25−NSS.1，教师项乘正系数2不改变余弦符号。

128张余弦均值=.44119645、中位=.46426605；仅3/128为负，最小−.09350635。
按独立float64 CC，教师优于学生78/128；这78张梯度全部同向，平均余弦=.55475171。
其余50张中仅3张反向，平均余弦仍=.26405025，说明“教师当前CC更低”不等于其梯度有害。
这不能证明共享网络参数上的所有更新均有益，也不能排除其他样本或训练阶段的冲突；
这里只读小样本诊断，不新增性能成绩、不做无偏泛化声明、不保存checkpoint。
因此暂不加入按教师/学生CC高低的硬门控，防止删去仍同向的监督；继续现有早期空间CC对照。

## 两组收尾：较强裁剪与RMS完整反向

2026-09-10，两个原定作业均完成且训练进程已终止，根目录均仅best/last两份PT。
以下为训练结束后重载best的完整5000张结果，未另做float64独立复评，不混称已独立核验。

|run（均前缀moe_alpha40_、后缀_seed42）|完成轮数/best EMA|CC|KLD|SIM|NSS|AUC-Judd|MAE|
|---|---|---|---|---|---|---|---|
|viewreg_sam_crop90|80/80|.8600213612|.1135510163|.8232143601|.9668219419|.7699175572|.0779729537|
|sam_exactfusion|50/5|.8615319420|.1127091609|.8241561455|.9689989939|.7702336757|.0767968273|

较强裁剪仅指边长比例下限.90（面积约.81），不改光学ROI。相同预算的.95弱裁剪控制为.86025104，
本次未改善；虽关闭增强后有明显回升，也不能把回升归因完全锁定在某一个因素或宣称跨seed显著。
该run源码3d5b520a，best epoch80 SHA：`f823f586189dcefe3035a0c31ad8ee433e4a991e2e187c5b01d5a0c75df4e1d2`；
`training_report.json`：`7bcba874d215b6079dda43e63a8dcd54a9e40d480b4fdca05be948460a8c54d3`；
`selected_checkpoint_test_evaluation.json`：`d9454439c53357a0c2870a0aab80bf6339aac3bc31145df83ac68ddea29d0269`。
alpha=.42783123/.43927994，四专家计数2324/2654/2310/2712；无未使用专家。

RMS完整反向源码486e0078，不增加推理参数。相对同源SAM.05/KL控制.86133209仅高约.000200，
低于空间CC/KD2完成候选.86204960；不能将不同蒸馏损失的两组差归因于RMS导数。
best SHA：`259784552c9e45435d438c18c8492256d2092560cb264def17ad9ba66716cc84`；
`training_report.json`：`29a9080fb776efe0ca955ed336c7e3881d163636b558170b2e3ffe56c7bc5b3d`；
`selected_checkpoint_test_evaluation.json`：`7413ce75d11af85d6754d6af1e299b1faf8f324ccb66237fadf49eefcf702d49`。
alpha=.43075329/.44109207，专家计数2324/2659/2300/2717；相位均有非零更新。
两个试验的完整命令、对照条件见前文；都不替换当前完成候选，保留证据而不删除run。

## global16候选独立复评：极小差异，不作为稳定增益

`moe_alpha40_sam_spatialcc_kd2_global16_seed42`仍在训练，epoch5 best SHA：
`e3d2e666dc666f57d581a9afd4cfecaed4603c7df513ced21fe5af5dee05f75d`。
在源码0f3c33d1、A100、batch32、ablation=none下，独立重载全部5000张：
float64 CC=**.8621131325**，与指标累积器差1.57e-9；训练记录.8621131796得到核实。
KLD=.1142900304、SIM=.8240813910、NSS=.9656057463、AUC=.7699995452、MAE=.0798730541。
报告目录`aligned_recheck_20260910_spatialcc_kd2_global16_candidate`，
`reproduction.json` SHA256：`67700d9d708b7450f67bfe4618e37bad1ca86f47eb25925a3b3be7105a54b3d2`。
测试ID SHA仍625dec6b…a3496d0；实际反序列化字节SHA被记录，活动源best后续可能变化。

和无global的`aligned_recheck_20260910_spatialcc_kd2_candidate`按相同5000个唯一ID配对：
均值差+.0000635306、中位差+.0001109618，2678/5000张改善。
按排序ID差值，用`default_rng(17042)`有放回抽5000项，重复2000次，均值95%分位区间为
[-.0000982241, +.0001913464]。这是固定权重的描述性重采样，不含训练seed方差，也未校正公开测试选模，
不是无偏显著性检验。没有充分证据支持为这一微小差异采用新增12544参数的全局算子；
暂保留无global的完成权重，等待该50轮试验收尾。两者距离.87仍有明显差距。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -m LightGenV2.tasks.t03_saliency.recheck_aligned --system optical --config "$TASK/configs/moe_alpha40_sam_spatialcc_kd2_global16.yaml" --checkpoint "$TASK/runs/simulation/moe_alpha40_sam_spatialcc_kd2_global16_seed42/best_checkpoint.pt" --run-dir "$TASK/runs/simulation/aligned_recheck_20260910_spatialcc_kd2_global16_candidate" --batch-size 32
```

复现前校验上述权重SHA，并换用尚不存在的run-dir，不能覆盖原报告；正常测试没有TTA或额外后处理。

## 用户要求暂停训练（2026-09-10）

用户因GPU/进程释放问题要求先停止。检查时PID2633439已不存在，2579012已是Z态，
全部T03正式训练命令均已不在活动进程清单中；未追加启动，也未杀其他任务或重置GPU。
随后`nvidia-smi`显示7张卡利用率全部0%、显存10–25 MiB，无本轮CUDA计算占用。
残留2579012和已完成的2547554为`<defunct>`，父进程尚未回收记录，不是继续占用显存的训练。
早期空间CC/SAM最后落盘epoch11，全局rank16组合epoch42；两者没有training_report.json，
各自best/last文件仍在。前文“仍在训练”描述为暂停前状态，不能据此自动重启。
等待用户明确恢复指示；目标.87未达到，不以暂停、停止或阶段性成绩冒充完成。
# 2026-09-10：完整干净训练集拟合诊断

这项检查不训练、不选新权重、不改推理网络；对已完成50轮KD2实验的best和last，
统一使用EMA权重、eval模式、无增强/随机光学扰动，复评全部10000张train2014。
不能把这些训练集数值填入论文测试性能列，也不能把训练日志中含SAM/噪声的CC直接与它比较。

| 权重 | 干净训练集CC（独立float64） | 已记录完整public-test CC | 说明 |
|---|---:|---:|---|
| best，epoch5 EMA | .8747706024 | .8620496019 | 既有独立测试复评 |
| last，epoch50 EMA shadow | .8768489866 | .8611217465 | 测试列是该末轮周期测试，未在本诊断重跑test |

训练拟合提高约.00208而测试下降约.00093，存在后期过拟合迹象；但训练集本身仍未达.88，
不能仅解释为“训练集已拟合很好，只需继续增强正则”，也不能仅凭这两点证明容量是唯一瓶颈。
另以既有FP16教师训练logits缓存、同10000个ID和密度图计算CC64=.8950629235，
仅作缓存拟合参考，不是新一次Qwen前向或新baseline测试成绩。缓存SHA仍为
`a45a90fe1dc029961464304373473d1271594128fc7b7f2ac80775f8638dd60e`，来源教师531c4a33。
计算逐图softmax(logits)与prepared density的Pearson再取均值，不将图像混在一起算CC。

复评入口新增`--split train`（默认仍为test），要求完整样本数、唯一ID；训练诊断不随机打乱/增强。
`--use-ema-state`显式读取last内的`ema_state.core/head`，缺失时报错，不回退live。
best本身已存EMA core/head，因此不加此开关。两个文件的SHA对应整个源checkpoint的实际加载字节，
不是另存一个EMA PT；没有新增周期权重。报告中的`aligned_readout_parameter_audit`仍是
Qwen同规格头的参考预算，不能当成光电网络总参数量。

```bash
TASK=LightGenV2/tasks/t03_saliency
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t03_saliency.recheck_aligned --system optical --config "$TASK/configs/moe_alpha40_sam_spatialcc_kd2.yaml" --checkpoint "$TASK/runs/simulation/moe_alpha40_sam_spatialcc_kd2_seed42/best_checkpoint.pt" --run-dir "$TASK/runs/simulation/fit_diagnostic_20260910_spatialcc_kd2_train" --batch-size 32 --split train
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t03_saliency.recheck_aligned --system optical --config "$TASK/configs/moe_alpha40_sam_spatialcc_kd2.yaml" --checkpoint "$TASK/runs/simulation/moe_alpha40_sam_spatialcc_kd2_seed42/last_checkpoint.pt" --run-dir "$TASK/runs/simulation/fit_diagnostic_20260910_spatialcc_kd2_last_ema_train" --batch-size 32 --split train --use-ema-state
```

命令拒绝覆盖已有run；重跑应使用新的明确命名诊断目录。每个目录含`reproduction.json`、
10000行`per_image_cc.csv`、配置与数据清单。全部ID以train/开头、无重复，CSV重算均值与报告相符。
best检查源码172b1919（160项CPU测试）；last检查源码fe327893（165项CPU测试），均先同步GitHub。
GPU1两项诊断串行，均正常退出且显存释放；未超过包括GPU0在训任务在内的两卡预算。

- best源SHA：`87ad4db51e3f58f9a41d6df09092439e5a09e81e93f88a3bfe2d3d008fafb29a`。
- last源SHA：`1cb22a0f5122ff2248c72c8f463a35e36d6396647acaa220d196aa2705033d7d`。
- best复评报告SHA：`1da42668e9252e1818ee7f9b2e97da4fe5d817f1b18c40321def8f4b9ca28f75`。
- last复评报告SHA：`882184f8d34c40263285272e459144093f9b14e02d5cf74bd064a5f235345dce`。
- best逐图CSV SHA：`e5ddd6a4bb48c6cbf9a02674d808b171d4b02280f11c0a47d8460961cec379d7`。
- last逐图CSV SHA：`6e312684e0784d27d5c413dddbb4f365adb9007738edec4047c5bacaff7b829b`。

## 拟合诊断后的受限rank64空间混合对照

上述干净训练集结果提示不能只增加正则。因此新增`moe_alpha40_sam_spatialcc_kd2_global64.yaml`，
直接继承已有global16试验，**仅把rank16改为64、另设run目录**。它不是当前正式模型的替换。
借鉴[MLP-Mixer](https://arxiv.org/abs/2105.01601)的跨位置MLP混合思路，不引入其完整骨干，
也不引入attention、Q/K/V、第三主分支或原生Qwen Transformer推理。

每个现有电子残差内、原token_pointwise之前：按Qwen的2×2块序还原14×14空间网格，
对每个通道独立做`196 -> rank -> GELU -> 196`，残差加回输入，再执行原pointwise。
矩阵跨通道共享、跨样本不混合。rank64用8×8低频DCT初始化下投影，上投影为0，
因此初始推理保持原函数。为降低高频DCT舍入误差，rank64先用float64构造再转模型dtype；
旧rank16仍保持原float32构造，未改变历史初始化。

| 项目 | 正式无global | 历史rank16 | 新rank64 |
|---|---:|---:|---:|
| 全局空间混合参数 | 0 | 12544 | 50176 |
| 该混合的额外MAC/图 | 0 | 2408448 | 9633792 |
| 显著性读出头参数 | 85412 | 85412 | 85412 |

真实模型可训练参数（core＋head，不含冻结前端）从1274511变为1324687，增加50176、约3.9%。
光学相位/ROI/传播次数、光router Top2、alpha≥.4、同尺度融合、DC20–30%及零像素偏移保持原合同。
电子计算量有增加，**不能直接套用历史版本速度/能耗**；本次不测5090D时耗/功耗。

与global16及完成的无global KD2对照同源c88e、同50轮预算、相同SAM.05、KD2、GT、EMA及学习率。
不是从当前87ad最佳权重出发，也没有加入正在另一张卡测试的COCO类别辅助标签。
global16历史run停止在42轮，比较时须明确实际完成预算，不能宣称它完成50轮。
来源SHA：`c88e1a41febc878cf87d6c80d4e27d7ac0c6bddc11d73a29497490d2e841ad73`。

源码`57febc24fa75204e421ab0ab3735608fe4d673a1`通过168项T03 CPU测试（32.03秒，13项既有警告）。
真实两张训练图验证：新旧初始eval输出最大差0，原生24层Transformer钩子调用0次；
一次SAM/优化器/EMA更新loss=.45979077，两个上投影RMS变化约4.99e-5、下投影约3.35e-5/2.36e-5，
光router相位约1.29e-5、四专家约1.86e-4–1.91e-4、global相位约1.84e-4。
短测只证明实现与更新有效，不是测试成绩，未保存checkpoint。

代码先同步GitHub，再从`.worktrees/t03_kernel13`启动新run，GPU1、启动PID917101：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_sam_spatialcc_kd2_global64.yaml --phase all
```

输出`runs/simulation/moe_alpha40_sam_spatialcc_kd2_global64_seed42/`，只保留best/last；
不改动另一张GPU0上的语义试验。当前没有新完成成绩，正式已核验最佳仍为.86204960。

## Router直接弧度参数化：不增加网络参数的训练对照

动机来自弱位置监督首轮相位检查（见UNLABELED_PRETRAINING）：router一半参数位于
sigmoid低梯度区域。这不是已证实的性能瓶颈，因此单独对照，不同时增大读出头、
加数据或更改光路。实现源码`f97c7a7d3206748f066d091622ee4230fac0fe1f`。

原来`phi=2*pi*sigmoid(raw_router_phase)`；新版本保存的同名参数**直接是弧度phi**，
传播仍为原来的`exp(i*phi)`，不改变入射振幅、478 ROI、224专家、17µm、10cm、Top2、
两级同尺度融合、alpha≥.4或20%–30%训练未调制分量。没有新增参数/分支/attention，
冻结Qwen前端和85412参数读出头不变。新方案训练总参数量与无global原学生完全相同。

采用已完成87ad（CC=.86204960）来源，配置`moe_alpha40_router_radians.yaml`继承
`moe_alpha40_extra_control.yaml`：40轮预算、SALICON-only、SAM rho=.05、KD2、EMA .995，
router学习率2e-5、feature相位2e-4，全部其它学习率/GT/测试/选模规则不变。
同数值学习率在不同坐标下不代表同物理步长；**改变优化坐标及其梯度条件正是本试验变量**。
先前相同来源控制试验提前停止，不能把它写成完成40轮；对照时比较相同轮数并披露状态。

### 权重和导出合同

- 新`architecture`末尾为`_router_radians`，旧签名的router参数仍是logit。
- 只允许显式`training.convert_router_phase_on_warmstart: true`将完全相同结构的旧权重
  转成`2*pi*sigmoid(old_raw)`；所有其它张量和alpha不变，不允许混合其它结构转换。
  初始化报告记录`router_phase_sigmoid_to_radians`。读已转换权重不会二次sigmoid。
- 不在训练中把参数硬截断/取模，EMA也保留连续坐标；光学前向的复指数本身具有2pi周期。
  相位weight decay必须0，避免对等价周期坐标施加不同收缩。
- 模型的`router.phase()`/`active_phase()`给出实际弧度；硬件沿用
  `encode_active_phase`，即`floor(mod(phi,2*pi)/(2*pi)*256)`。负相位或超过2pi均可正确编码。
- `visualize.py`与`phase_change_report`按源/目标各自architecture解释router；特征相位仍sigmoid。
  跨参数化时raw RMS不可比较，报告为null，比较实际圆周相位差。禁止旧脚本对新router
  再套一次sigmoid；如别的AI自写分析，请调用`router_phase.checkpoint_phase`或加载实际模型。
- 本次只验证既有相位编码/active_phase接口，没有制作新实验室ZIP或宣称完成硬件实测。

完整186项T03 CPU测试通过（35.54秒、13项既有警告），覆盖初始探测器场、Top2概率/选择、
active_phase和8-bit编码完全相等；梯度满足旧梯度=新梯度×sigmoid雅可比；
覆盖端点跨越、严格checkpoint迁移/重载、实际相位变化报告与单变量profile。
真实两张SALICON训练图使用源87ad检查：转换前后eval输出最大差0、active_phase的8-bit编码
完全相同。一次真实GT+KD+SAM/优化器/EMA更新loss=.38035563，router弧度RMS更新1.83137e-5，
四专家raw RMS约1.94e-4–1.99e-4、global 1.94105e-4。短测CC=.898791仅为这两张训练图，
**不是测试成绩、更不是达到.88目标**；未保存周期PT。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_router_radians.yaml --phase all
```

输出`runs/simulation/moe_alpha40_router_radians_seed42`，仅best/last。启动前先确认旧GPU1任务
及其子进程已释放，禁止占第三张GPU；公共测试选模偏差仍须披露，目标尚未达成。

### rank64停止与新对照启动记录

rank64完整定期测试第5/10/15/20轮CC分别.86223520/.86197010/.86172687/.86155056，
连续回落。因此第23轮进行中停止PID917101，last保存到22轮，**不是完成50轮**。
检查无正在写入的PT后停止，核实并清理孤儿921692、923819；
父进程和全部已记录子进程921694、921854、923967均消失，GPU1无计算进程后才启动新任务。
CPU重载旧best/last成功，未删除数据或权重：

- best第5轮、周期测试CC=.8622351954：`2618ed97ab9f1d842684b85ea9810d5a06368dcf556efb43d67066a450edc584`。
- last第22轮、非安排测试轮：`499e93ab8aa98b37953f3d068341c20fbaa3742fe0e1e4ffb8ad09a2f9b7ee80`。

此处是重载并读取已记录指标，**不是独立重新跑5000张**；微小差异不足以替换较小正式候选。
直接弧度试验在GitHub确认`dabb998373e0c07ace3c2a1500721a7289d3e129`后启动，
GPU1、启动PID1132984、cwd `.worktrees/t03_sam_early`。GPU0仅位置辅助试验PID1033718，
本助手没有增加第三卡；其他AI的ABO任务不在本任务清理范围内。
PID仅记录启动身份，终态以实际进程/日志核查为准。没有把旧训练源码中途切换成新实现。

### 直接弧度首轮/第5轮观察与端点检查

初始完整5000张CC=.8620496481，显式转换true、fusion reset false，与来源87ad一致。
同来源、相同40轮配置的旧坐标控制`moe_alpha40_extra_control_seed42`比较如下：

|周期测试轮数|旧sigmoid坐标|直接弧度|
|---|---:|---:|
|1|.8620438030|.8620568825|
|5|.8617905981|.8617967274|

差异均仅约1e-5，**没有实质性能提升证据**。直接弧度第5轮alpha=.43037564/.44090962，
约束未破坏；不替换正式已核验候选。控制组已停止于last24，并非完成40轮；新组仍运行。
第1轮best为EMA，SHA
`dfcd1263dafa5a39b07db4cf0472d2e231c39f9cbedb1a67c2fb46ed6892d393`；
它是周期测试结果而非再次独立测试，路径之后可能被更优best覆盖。

CPU读取上述best与固定来源87ad，把router两者都解释成物理相位后比较。
按来源`sigmoid(raw)*(1-sigmoid(raw))<.001`分组，每组25088像素：

|区域|实际圆周相位更新RMS/rad|硬件8-bit码改变比例|
|---|---:|---:|
|原低雅可比区域|.00015090234|.011958%|
|其它区域|.00015083341|.506218%|
|全部50176像素|.00015086787|.259009%|

相位差用`angle(exp(i*(new-old)))`，编码直接调用硬件`encode_active_phase`。
两组实际更新幅度接近，说明端点不再被sigmoid雅可比压低；但绝对步长仍小、灰度是否跨码
还受初始值到量化边界的距离影响，不能把低跨码比例解释成未学习或据此夸大精度收益。
此处使用EMA，第1轮live的SHA为
`b42e2962446bd46ae5c87fb79a5df1ade87c85d896269e66c314aa58fe146eab`，
不把live raw更新与EMA相位更新混为同一数值。未额外保存周期权重，未运行硬件。
# Router弧度坐标试验停止记录（2026-09-10补充）

`moe_alpha40_router_radians_seed42`已在第15轮保存后停止，不是完成40轮。
第5/10/15轮公开测试CC=.8617967273712158/.8614705110549927/.861286886882782，持续回落。
best仍第1轮EMA .8620568824768067，几乎等于来源.86204960，不证明新坐标带来性能提升。
best SHA256 `dfcd1263dafa5a39b07db4cf0472d2e231c39f9cbedb1a67c2fb46ed6892d393`；
last第15轮SHA256 `c45d20a02bda7bef053050804751b400491881c7b0b2578a5d96ee1746c951fe`。
停止后两个PT均CPU重载通过；父1132984及5个子进程确认退出，GPU1恢复25MiB、无本任务进程。
不删除该对照，不替换正式小参数sigmoid版本；后续特征预训练仍从正式87ad出发。

## ASAM训练备选（2026-09-13，未取得新性能结论）

依据：[ASAM, ICML2021](https://proceedings.mlr.press/v139/kwon21b.html)，采用其p=2逐元素尺度扰动思想。
对电子weight参数取 `T=abs(weight)+0.01`，bias及其他标量取1，扰动为
`epsilon=rho*T^2*gradient / norm(T*gradient)`，rho=.5。这里的rho不是普通SAM的欧氏半径，
不能把.5与旧SAM的.05直接说成“同样扰动扩大十倍”。
只在原电子子空间构造扰动，光学参数不参与临时扰动，但第二次反向仍训练所有原启用参数。
这是对光电模型的训练适配，不宣称原论文已验证SALICON、光学结构或0.87目标。

保持原AdamW、EMA和一次optimizer更新，SAM两次forward共用随机实现，并在异常时恢复权重及RNG。
不加入BatchNorm、推理网络、额外数据或测试后处理；GT与原CC-KD2不变，不叠加可靠教师或hard-CC。
`moe_alpha40_asam050.yaml`沿用完成SAM/KD2对照的c88e较早来源、50轮日程、全部组学习率。
配置中历史sam_rho=.05仅保留入口兼容；启用training.asam后实际扰动使用asam.rho=.5。

```bash
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --phase all \
  --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_asam050.yaml
```

测试/真实数据检查通过且GitHub已同步后，才可在已释放的自有GPU上运行，不得超过两卡。
所有运行结论以run记录和完整复评为准；准备备选不等于启动或达标。

### ASAM已启动及真实更新检查

源码`6cbc27c5da512f8f110dd046bf58b8fb5f737d84`，235项CPU测试通过，已推送GitHub。
真实四张train图、CPU单步检查保存在`runs/smoke/asam050_preflight_20260913/report.json`。
任务损失.44322118、第二次扰动损失增加.62421393，所有六张相位都有有限非零梯度和真实更新：
router原始参数RMS变化1.2368e-5，四专家约1.94e-4至1.98e-4，全局1.9559e-4。
这是raw参数单步差异，不是最终物理相位改善、全数据性能或硬件可用性证据。

检查直接绑定`model.core`参数对象，不能用首次forward前后的整模型参数全路径做交集：
激活光电替换时相同core对象也挂入visual.blocks，遍历前缀会变化，冻结原生块会退出遍历。
这不是相位冻结；验证实际参与ASAM的45个电子weight张量身份前后一致。
冒烟测试不保存新checkpoint，不改变正式来源权重。

确认上组PID414381及子进程退出、GPU1无任务后，启动PID/PGID476516，
GPU UUID `GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d`，50轮预算，来源c88e不变。
固定工作树`.worktrees/t03_balance`不能在运行中checkout，
正式产物位于T03 `runs/simulation/moe_alpha40_asam050_20260913_seed42`，含console和launch记录。
目前启动不代表已达到.87；如后期持续退化会停止并保留best/last，不冒称完成50轮。

半径配对备选`moe_alpha40_asam010.yaml`仅将ASAM归一化半径.5改为.1，eta仍.01，
来源、50轮日程、数据、损失、全部推理参数保持一致。不是此前普通SAM rho=.1的重复命名。
待baseline复评结束、确认空闲GPU且配置测试/GitHub同步完成后运行，总并行仍不超过两卡。
