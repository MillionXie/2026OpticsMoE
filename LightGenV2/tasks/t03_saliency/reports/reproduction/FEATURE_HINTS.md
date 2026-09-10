# 训练时空间特征监督（不增加推理网络）

## 当前候选：空间位置关系蒸馏（已启动，收益待验证）

2026-09-10启动审计：源码`aa0dc20283b789c836e9b727108eb3f41eddaced`已发布到GitHub，
独立工作树`.worktrees/t03_balance`，GPU0/PID1691668，预算40轮。
run为`runs/simulation/moe_alpha40_relational_kd_seed42`；run_manifest记录相同源码，
完整5000张初始化CC=.862049694442749，教师特征缓存SHA与本页39aa合同一致。
缓存provenance的git_commit是缓存生成时源码，不是当前训练源码；当前源码以run_manifest为准。
光学Top2/alpha≥.4/训练DC/原推理结构不变，不能把初始化成绩写成新增训练收益。

GPU0原45轮联合组停止于第19轮，best第15轮CC=.8386751629829406，保留：

- best SHA `a62f776b72b75936f71a167202abecb63f9d8d6f9c9e12f32f001d26ea331de3`。
- last SHA `2845f8fafeb3d9c03f4e3448324a1e5d4e42540a3ca479cbc1079a73a7f4440f`。

停止后两个PT均可读取；仅清理经UID/命令/cwd/进程组核实的自有父子PID
1483586/1489512/1489530/1489722/1491964/1492229。确认GPU0无计算进程、仅12MiB后再启动新组。
没有删除权重或数据。停止原因是有限两卡下的试验优先级，不是证明该组不可能再提高。
GPU1/PID1483592稳定路由组仍运行：第20/25轮CC=.8374878645896912/.8400173555374145。
其第27轮已正常完成，仍低于原正式best；不要把未完成的60轮预算写成结果。

补充只读CPU路由诊断（原训练清单前64张、batch4、eval关闭随机扰动，非全测试审计）：
旧联合组best15四专家选择次数32/32/35/29，共4种专家对，alpha=.44356108/.47995389；
稳定组当时best20（SHA `6dfdba4f7620c54548ca6ac83c0db3acf28925d69401d510d6de3e13e967c89e`）
次数29/35/34/30，共6种专家对，alpha=.48469734/.51723713。未见这64张上的明显坍缩，
不能把性能差距直接归因于路由坍缩。best20之后正常被best25覆盖，未另存阶段PT。

`configs/moe_alpha40_relational_kd.yaml`从原正式87ad的core **和原读出头**共同初始化，
不换教师头、不添加投影。参考[Liu等，CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.html)
的密集预测pair-wise蒸馏思想；不是整篇方法复现，不引入其GAN、骨干或分割标注。
这里只在训练损失内比较位置关系，不执行attention，也不把相似度矩阵乘回特征。

对学生与教师各自的`F[B,192,14,14]`：沿196个位置减去每通道空间均值，再对每位置的192维
向量做L2归一化；计算196×196余弦相似矩阵，排除对角线后求两矩阵的均方误差。
比较的是同图不同位置之间的关系；允许师生通道基底不同，对通道正交变换、正整体尺度、
每通道空间常量偏置不敏感。不使用跨图标签/测试图或逐图最优对齐。
空间去均值仅在训练损失内进行，CCD数据、推理特征、20%–30%训练DC均不变。

来源/数据/GT+空间CC map KD2/SAM.05/EMA/40轮LR预算全部继承`moe_alpha40_extra_control.yaml`；
唯一新增训练项权重1→.1，在30轮线性衰减，后10轮保持.1。缓存仍为39aa、教师为531c，
严格检查字节SHA、10k有序train ID、网格和预处理合同。没有额外图像/标注，也无新增训练参数。
函数和模型状态不变，因此起点仍应为87ad的.86205，而不是固定教师头预训练的负CC。
公开test5000张按起点/首轮/每5轮/末轮选best，仍有测试选模偏差；仅best/last。
这是待验证方法，不宣称有性能收益；与稳定路由组合计两张GPU，不开启第三个任务。

2026-09-10的CPU梯度动机检查（未训练/未保存PT）：87ad标准eval、前8张有序train、2个batch×4，
教师39aa缓存；关系MSE=.10044791/.12872997，教师非对角关系平方均值=.06267693/.07898695。
分别求`GT+2*spatial_CC_KD`与权重1的关系损失梯度，不包含路由/物理正则：

|组|任务梯度范数（两batch）|关系梯度范数（两batch）|余弦（两batch）|
|---|---|---|---|
|E|2.09865 / 2.68988|.083797 / .137495|-.00702 / .09118|
|光router|.00112562 / .00040071|.00046950 / .00036677|.90004 / .90403|
|专家/全局相位|.01108066 / .00925450|.00092997 / .00124478|.03954 / .01114|
|CCD读出|2.22861 / 2.43479|.084287 / .158401|.02113 / .01381|

关系损失到原读出头的梯度为0，因其输入是头前的融合特征；任务损失仍训练该头。
这仅证明梯度可达和该小样本上的尺度，不能推断全训练方向一致或泛化必然提高。
配置取权重1，避免一开始把关系梯度放大数倍冲击路由；不修改正在运行的固定头训练。

```bash
# 复现命令：现已有同名任务运行，不要重复执行；须先核查空闲GPU及输出目录。
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_relational_kd.yaml --phase all
```

检查run内`relational_distillation_provenance.json`、`metrics/training_history.csv`中的
`relational_weight/train_relational_loss`、原初始化SHA和最终完整测试/路由审计。
实现`d2224927`通过197项T03 CPU回归（44.46秒、13条既有依赖警告）：包含通道正交/偏置不变性、
空间错位敏感、教师无梯度、缓存SHA/身份校验、SAM双前向单次更新、零关系权重与原SAM逐参数完全一致。
测试不代表完整数据性能；GPU启动状态见本节开头，结果须以完整测试为准。
补充精度保护后的`b434b543`同样197项通过（45.63秒），显式关闭关系矩阵乘法的外层AMP，
仅该训练损失使用float32，不改实际模型推理精度。
真实train前2张、CPU、种子42、保留训练光学扰动、一次SAM更新：loss=1.05701140，
relation=.08337363，core/head键集合不变，24个原生Transformer hook调用次数0；
router raw参数最大变化约2.0e-5，四专家和全局相位均约2.0e-4，原head绝对变化总和1.70807619。
此更新只在内存中验证，没有保存新PT或改变任何正在运行的训练；这些不是泛化分数。
核验按部署的core/head进行，不比较会被激活过程替换的Qwen包装层模块键名。

## 新试验：固定教师基底预训练（待验证，不替换正式best）

配置 `configs/moe_alpha40_feature_pretrain.yaml` 与旧的弱cosine提示不同：
不增加训练投影，直接学习教师decoder输入的192通道空间特征。
学生core从已核验的87ad来源初始化，现有85412参数头改为同规格教师decoder权重；
不复制教师adapter或24层Transformer到学生推理图，参数量/光路/alpha下限均不变。
因此本试验的epoch0是**换头后的性能**，不是原87ad的.86205；原正式best完整保留。

- epoch1–15：固定这个教师decoder，训练原有光电core；AdamW普通单步，无SAM。
- epoch16–60：解冻同一个头，电子子空间SAM rho=.05，联合优化。
- 两阶段都保留GT损失与空间CC教师图蒸馏2.0，不使用额外COCO图或类别/框标签。
- 特征项为 `MSE(((S-mean_hw(S))-(T-mean_hw(T)))/sigma_T)` 加
  `.05*MSE((mean_hw(S)-mean_hw(T))/rms_mean_T)`；每通道尺度只从10000训练样本统计，
  下限.05。它只用于训练损失，不改变推理特征或CCD归一化，也不代表物理DC去除。
- 总特征权重前15轮2.0，第16轮降至.2、第40轮线性降至0，之后仅原任务损失。
- 前15轮LR：E/CCD各3e-4，feature相位5e-4，router5e-5，已有空间FFN1e-3；头冻结。
  第16轮解冻头（基础3e-4），所有基础LR乘.2，随后余弦衰减到该阶段起点的5%。
  原硬路由均衡退火保留；特征阶段的LR覆盖原joint/polish率，以history的lr列为准。
- EMA=.995在换头之后初始化；optimizer在冻结之前包含头参数，第二阶段确实能够解冻更新。
  前端仍冻结、Top2、同尺度融合alpha≥.4、478/224/17µm/10cm、训练DC20–30%保持。
  测试关闭随机扰动，不宣称硬件效果。

特征缓存沿用下文SHA39aa、教师头SHA531c（完整SHA在配置）；解析前校验实际文件字节。
缓存只用于训练、身份顺序严格一致；尺度写入`feature_pretraining_provenance.json`。
无新增模型状态字段或周期PT，仍只保存best/last；换头来源写入initialization报告。
新profile的60轮预算不是已完成结果，收益与否以完整5000张公开测试为准。

```bash
# 在已发布源码的仓库根目录执行；GPU0仅为示例，先检查空闲及两卡总预算。
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_feature_pretrain.yaml --phase all
```

同样按公开test选best，存在选模偏差；不能把多个训练变化合并后的结果当作单变量因果结论。

实现源码`89b6a5ed`：服务器xml环境CPU测试189项通过（42.18秒、13条既有依赖警告）。
真实训练图`train/000000000009`、`train/000000000089`短检查通过：完整10000张缓存身份/SHA
检查后，分别执行冻结头普通一步和解冻头SAM一步（不是完成15轮预训练）。
两步loss均有限，head参数绝对变化分别0/5.12397346；router原始参数累计RMS变化3.6511e-5，
四专家约4.999e-4至5.119e-4，全局相位5.13185e-4。光学参数确实更新；这是未映射的raw参数，
不可把它写成弧度或BMP灰度变化。注册在24个原生Transformer block上的hook调用次数为0。
这两张的CC不是测试成绩，换头后的初始训练样本CC较差是已知风险，不隐藏或用旧头成绩替代。
没有保存这次短检查的临时PT，部署参数/键名未增加。

正式启动：GitHub已发布源码`2a3b9a57796156bb66fd83daeb1c766b1d8a98df`，
独立worktree `.worktrees/t03_kernel13`（历史工作树名，不表示本模型使用13×13卷积），
GPU0、PID1294991、run `moe_alpha40_feature_pretrain_seed42`。启动前GPU0无计算进程；
旧region和router对照已分别停止并释放GPU0/1，启动该组时本助手只使用一张卡。
正式预算60轮，训练状态看run的console.log/history；不能把启动记录当作已完成结果。

### 预训练早期检查与单变量强度对照

control首轮/第5轮完整5000张周期测试CC=.7645925910949707/.8140745836257934。
每轮10000训练图，训练特征损失1.47035365→.91055371，空间MSE1.27865902→.88073992，
均值MSE3.83389253→.59627589；不仅总loss变化，固定教师头下的任务表现也在恢复。
第5轮live alpha=.43756527/.44824940（不是EMA权重的alpha报告）。仍低于旧正式best，
不能据此声称新预训练有效，需继续联合阶段并完整复评。

另做CPU只读梯度诊断：载入当时第4轮**live** last字节，SHA256
`7911a489f98914439746fa491076e8443aa6c6b9c59132d3c2f81e1c3db936f2`；不保存额外阶段PT，
last之后正常被覆盖。诊断不是可单独交付的永久epoch4权重，也不是最终性能证据。
取manifest中前16个train ID，4×4batch，RGB224无增强、eval关闭随机光学扰动，头冻结。
对同一前向分别求`GT+2*spatial_CC_KD`和`2*feature_loss`的梯度，不混入物理/路由正则，
不调用optimizer.step、不改变在GPU0训练的进程。以下为4个batch的均值（不是全数据结论）：

|参数组|任务梯度范数|特征梯度范数|两梯度余弦|
|---|---:|---:|---:|
|E|4.53742|.414823|.01406|
|光router|.000249324|.000041979|-.11652|
|专家/全局相位|.0109315|.00454925|.02648|
|CCD电子读出|2.07930|.147333|.05976|
|已有空间FFN|.245248|.0277544|.01000|

特征监督确实传入光学相位；当前权重2下，E的特征/任务范数约.0914，相位约.4162。
未观察到E/相位平均梯度强烈反向，但也并非同向；16张clean输入不足以证明全训练中的关系。
据此增加`moe_alpha40_feature_pretrain_strong.yaml`单变量对照：前15轮feature权重2→10，
其他数据、来源、头、LR、种子、60轮预算、第16轮后联合策略完全一致。它测试更强特征监督，
并不预设有益；若持续无改善则停止并释放其GPU。仍不增加电子参数或分支。

```bash
# GPU1必须先确认可用；加上control，最多两张卡，不再启动第三组。
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_feature_pretrain_strong.yaml --phase all
```

强度对照源码`3b82a8af03c057e3b204c4bc363f3a7abd3de2f5`，完整T03 CPU测试190项通过
（39.75秒、13条既有警告），新增测试逐项确认除initial_weight外的有效训练配置一致。
GitHub推送核实后，从独立`.worktrees/t03_sam_early`在GPU1启动PID1366554；启动前该卡无计算进程。
control仍为GPU0/PID1294991；只用两张卡，其他卡上的他人/其他任务进程不操作。
run目录分别为`moe_alpha40_feature_pretrain_seed42`和`moe_alpha40_feature_pretrain_strong_seed42`。
此处是启动与测试记录，不是60轮完成或超过.88的报告。

### 路由审计后的调整：固定完整输入路径，再小步联合微调

重要负面结果：control第10轮best（EMA，CC=.82740550，SHA256
`5da290707edb47e0006d7bdbc797ad05d8f9e36163de3346af14645dd069a64c`）在前256张
clean训练诊断图上选择计数为`[0,256,0,256]`，仅1种Top2组合，有效专家数2。
alpha=.44330633/.46157691，满足下限，但路由集中。相对87ad起点的圆周相位RMS：
router .02733965rad，专家0/1/2/3为.13480623/.13321689/.20036843/.17970687rad，
全局.14553317rad；这是实际相位变化，不是BMP字节变化，也不意味着性能更好。

后续检查必须区分阶段：control第15轮best CC=.8335291910171508，前64张训练图
计数`[32,32,35,29]`、4种Top2组合，已恢复分散，**不能说control始终坍缩**。
相同64张图的87ad来源为`[29,35,34,30]`、6种组合。
strong第5轮best CC=.7805594863891602，对应计数`[0,64,0,64]`、1种组合。
这些是clean训练子集的路由诊断，不是完整测试集专家分布；不包含随机硬件扰动。

control已停止并保留第15轮，strong停止并保留第9轮，不是两组完成60轮。
control best/last SHA256分别：
`b2d8e0d9d903efa1de66664e02af07fd9ff9310e53c1f4db71889b0b99508fba` /
`42ccec2b991d35037f92a3c21ddd70abc541e9052eed38227c9c91de11de1616`。
strong best第5轮/last第9轮SHA256分别：
`f8cde610ee3d70c4fd036a1f37840896cc3541bbe3507e95d4687f3b7a993d45` /
`689da6a8f245728dccb9fd76ef278e982f27417e84e38833a1bd2421f8b23ac2`。
均已停止后成功CPU重载，父1294991/1366554及各5个子进程退出；权重和数据不删除。
control值得继续联合训练；strong只是提前结束的早期负面证据，不证明完整60轮必定失败。

实现`feature_pretraining.freeze_router_path`：前15轮固定完整路由前路径的两组Linear+LN，
即`hybrid.input_adapter/input_norm`和`hybrid.optical_branch.core.input_adapter/input_norm`，
以及光router自身参数。第二组是共享的SLM振幅编码器，也用于其他光阶段，**不复制新分支**。
只冻结第一组不能保证光router输入稳定。Qwen patch/位置前端本来就冻结；所有冻结仅是训练策略。
专家/全局相位、原电子残差、CCD读出继续学习，头仍按既定策略前15轮冻结。
第16轮解冻全部，路由前输入参数单独用基础LR5e-6×联合倍率.2=1e-6；router为1e-5。
这组输入参数不加入SAM的瞬态扰动子空间，但仍用第二次反传梯度更新；其他旧profile不变。
保留光router Top2、alpha≥.4、同尺度融合、478/224/17µm/10cm与DC20–30%，新增推理参数0。

两条有区别的优化路径（不是共同起点单变量消融）：

1. `moe_alpha40_feature_pretrain_stable_router.yaml`：从87ad重做15轮稳定路由预训练＋45轮联合，预算60。
2. `moe_alpha40_feature_joint_router_low_lr.yaml`：从control第15轮best EMA进入45轮联合微调，
   不重复固定头阶段；`frozen_head_epochs: 0`，头立即可训练，输入映射立即以1e-6小步学习。
   特征权重从.2在本run第25轮降到0，LR及硬负载均衡计划与原第16–60轮对齐，
   但重建optimizer/EMA及数据顺序，**不是精确断点恢复**。来源头仍是同一个固定教师头。

```bash
# 已测试源码发布后，在仓库根目录执行，先检查GPU0/1及两卡总预算。
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_feature_joint_router_low_lr.yaml --phase all
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_feature_pretrain_stable_router.yaml --phase all
```

源码`e4c17ac8`通过193项CPU测试（41.75秒、13条既有依赖警告）；测试包含两级输入/相位
冻结期间不被AdamW权重衰减改变、解冻后恢复更新、完整参数分组无遗漏/重复、零预热联合入口。
收益待实际训练验证，不把只满足路由稳定条件当作达到CC=.88。

真实图像CPU短检查也通过：使用前两张train图，原生24个Transformer hook调用数均为0。
分离出来的两级输入映射共240864参数（从原E组移动，不是新增）；稳定方案冻结期间
这些参数及router参数绝对变化为0，更新前后clean路由概率逐值差为0。
第16轮解冻短步后参数发生变化，路由概率最大差约.000228，证明没有忘记解冻。
联合入口首步直接执行SAM，参数变化正常；来源第15轮头与教师decoder逐值相等，
初始化没有破坏已有头。所有loss有限；这不是训练或测试集性能复评，不保存短检查PT。

正式启动源码`b5b4dd26fcb942fbfbb7c95c620e22dfe1fcead1`，193项CPU测试再次通过（40.40秒）。
GitHub发布确认后，GPU0/PID1483586在`.worktrees/t03_kernel13`运行45轮joint-low-lr；
GPU1/PID1483592在`.worktrees/t03_sam_early`运行60轮stable-router-pretrain。
run目录使用各profile名加`_seed42`，完整命令由run_manifest记录。
启动前两卡无计算进程，旧父PID均不存在；此时本助手仅这两组，不操作其他卡上的任务。
不把不同来源/训练阶段的两组包装成单变量公平消融，也不声明已经达到.88。

### 稳定路由组第5轮实际权重检查（仍在训练）

源码`b5b4dd26`，`moe_alpha40_feature_pretrain_stable_router_seed42`第5轮EMA周期测试
CC=.8230674638748169（完整5000张），高于原固定头组同阶段.8140745836257934，
但低于正式best .86204960，不能作为达到目标或最终胜出的证据。
检查时best文件实际字节SHA256为
`8cd43cfa53e3d8b9777d5a377f16d27cd1a4c99c8e28334b3eacc365254c200f`；
后续best会被覆盖，未另存周期PT。

CPU、OMP/MKL各2线程，原训练manifest前64张、无增强、batch4、eval关闭随机扰动；
以`load_hashed_checkpoint`载入同一字节的core/head，router forward hook统计selected_mask。
选择次数`[29,35,34,30]`、6种Top2组合；alpha=.4437673986/.4598472714。
这是小样本训练诊断，不替代最终完整测试集路由审计。
与87ad来源逐参数比较，两级输入适配器8个参数及router相位最大绝对差均为0；
头与531c教师decoder最大绝对差也为0，实际冻结生效。
四专家raw相位参数差的RMS依次.08621831/.09695660/.14600658/.12792611，
全局相位.09849585，证明其余光学权重确实更新；这些是raw参数，不是弧度或BMP灰度。
诊断未训练、未占用第三张GPU、未修改运行中模型或保存额外checkpoint。

## 已完成对照结果

2026-09-10，`moe_alpha40_hint_control_seed42`与`moe_alpha40_hint_cosine_seed42`
均完成50轮，源码`9bc65286e03d327c962abaa0e35a52770b2a8dab`。

| 组别 | 更新后最高test CC | 第50轮test CC | 最终保留 |
|---|---:|---:|---|
| control | 0.85780809（epoch1） | 0.85570458 | epoch0，0.85812011 |
| cosine | 0.85769613（epoch1） | 0.85596133 | epoch0，0.85812011 |
| spatial-centered cosine | 0.85772776（epoch1） | 0.85589607 | epoch0，0.85812011 |

两组best的5000张完整复评均为0.85812011，alpha=0.43413550/0.44144565；
专家选择占比23.56%/26.32%/23.19%/26.93%，有效专家数3.98277/4，无未使用专家。
这不是新训练得到了同样优良的新权重，而是没有超过源权重，故保留了epoch0。
去空间均值版本`moe_alpha40_hint_centered_seed42`随后也完成50轮，源码
`b3f88069bdc6864c0d4a3338152fbed6402a3045`；重载best完整测试CC=0.8581201133728027，
同样保留epoch0，alpha和专家选择次数与上面一致。这一系列三组均无新提升，
暂不重复普通/去均值cosine提示；不能将保留源权重的分数作为新监督有效的证据。

对应run内`selected_checkpoint_test_evaluation.json` SHA256：

- control：`0eefc5a5e3d86c376757d20862a87b038c481510d4f310e2c9d6b4fe5d4520d7`
- cosine：`aee4ac592de7d1022b0a2cba9ca671784556a1123c7a3cb122909c2080b17014`
- spatial-centered cosine：`6ebe21338154cb34bcfda581ef2b858871f1eb57cdcdeaa1f4fe83295366efc8`

## 方法假设

假设：只蒸馏最终单通道显著性图，可能不足以监督两层光电融合后的192通道特征。
参考[FitNets, ICLR2015](https://arxiv.org/abs/1412.6550)的中间特征提示与训练回归器思想。
本实现不是完整FitNets复现：使用SALICON空间任务、固定现有光电模型与逐像素通道cosine损失。
论文只提供方法依据，不保证本任务提升，CC≥0.87仍是待验证目标。

## 推理与训练边界

推理完全保持`moe_alpha40_adaptive_keepkd`：冻结Qwen patch前端、光router Top2、
两层同尺度光电融合、alpha≥0.4、478有效面积、224专家、17微米、10cm、20%–30%零级分量、
pixel位移0、原85412参数解码头。无新增attention、Transformer、分支或推理次数。
本对照不使用空间FFN新增模块，从原结构历史best CC=0.85812016开始。

仅训练时：教师的`AlignedReadout.decoder`输入是经过adapter+LN的`[B,192,14,14]`真二维网格。
学生的第二融合后latent也为同尺寸，添加一个**训练专用**192→192无bias的1×1投影（36864参数），
初始化为单位阵，补偿二者通道基底不同；每个像素对192通道做L2归一化，损失为平均`1-cosine`。
不强行把教师激活幅度注入光路；teacher与投影都不参与测试或部署。
原模型`core`与`saliency_head`的state_dict/architecture标签不变。
投影保存在last_checkpoint的独立`training_only_hint`字段；不是推理权重的一部分，不装入student。
best checkpoint仅含原模型有效权重；此字段为null。EMA只作用于core/head，不包含投影。
现有trainer的“重载权重+新optimizer”不等价于精确断点续训，不能把它写成恢复optimizer/RNG。

## 缓存与数据安全

仅10000个train2014 ID，顺序必须与manifest完全一致；不缓存test特征。
教师仍为同规格头Qwen baseline，SHA256：
`531c4a330fc05e2c27b037784588716d248a29d1ab8da951d443165dcce552f8`。
float16缓存约752640000字节（约718MiB）；一个共享缓存，不复制到每个run。
记录教师SHA、数据/标注SHA、源码commit、缓存SHA、真实网格合同；拒绝错序、错误尺寸或非有限值。
本轮特征监督禁用增强以确保严格对齐，不把插值后的特征伪称为教师真实增强输出。
正常公开测试5000张、无独立validation；起点/epoch1/每5轮/末轮选最高CC，存在测试选模偏差。

## 两组受控训练

`hint_control`与`hint_cosine`的源、GT损失、map KD=.6、学习率、EMA、样本顺序相同。
唯一差异是cosine组附加hint权重.1→.02，30轮线性退火，投影LR2e-4。
共同源SHA：`36333e6ba013cac3a2801bce4babcc2fb5cd36adf02969e019437b40ceebaffd`。
最多50轮，联合训练，41起精修；电子1e-5、相位2e-4、router/CCD读出/头各2e-5。
其余采用历史原结构策略；保留best/last，自动调速，不改变正在运行的viewreg对照。
原legacy epoch继续用于control，hint epoch是任务内等价损失+显式hint；有零hint梯度更新等价测试。
所有包含训练专用投影的参数都参与梯度裁剪；投影不使用weight decay。

### 基于缓存审计追加的空间去均值对照

固定缓存SHA `39aae6b13452975186266672e51216e72391112df65b23c453de030033826304`，
10000张教师特征`F[N,192,14,14]`，float32累加：
`sum_n mean_c(mean_hw(F)^2) / sum_n mean_chw(F^2) = 0.931409527`。
即93.14%的特征平方能量属于每个样本各通道的空间常量部分。
这是电子特征的共同分量，**不是物理未调制光的比例**，不改变20%–30%光学DC。
普通cosine可能主要拟合共同分量，故增加`hint_centered`：
投影后学生特征和教师特征分别减去各自空间均值，再做同样的逐像素通道cosine。
仅训练损失变换，实际送给解码头的学生特征不变；其余初始权重、系数、数据、优化器完全相同。
有共同大偏置/不同局部结构的单测，证明新损失对局部差异敏感、对空间常量偏置不敏感。
这只是受控训练假设，是否提高完整测试CC仍需实际验证。

## 可复现命令（仓库根目录）

先Git拉取已测试发布的commit；先查GPU空闲容量，以下GPU编号是示例。

```bash
conda activate xml
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK=LightGenV2/tasks/t03_saliency
python -m pytest "$TASK/tests" -q
CUDA_VISIBLE_DEVICES=6 python -u -m LightGenV2.tasks.t03_saliency.export_teacher_features --config "$TASK/configs/moe_alpha40_hint_cosine.yaml" --checkpoint "$TASK/runs/simulation/qwen_aligned_head_staged_seed42/best_checkpoint.pt" --output "$TASK/runs/simulation/teacher_features_20260910/teacher_train_features.pt" --batch-size 16
CUDA_VISIBLE_DEVICES=1 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_hint_control.yaml" --phase all
CUDA_VISIBLE_DEVICES=5 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_hint_cosine.yaml" --phase all
CUDA_VISIBLE_DEVICES=6 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_hint_centered.yaml" --phase all
```

勿重复写同一输出目录；缓存完成且SHA验证后再训练hint。环境沿用xml/Torch2.6.0+cu124/Transformers4.57.3。
产物固定`runs/simulation/moe_alpha40_hint_<control|cosine|centered>_seed42/`。
查看feature_hint_provenance、resolved_config、初始化SHA、run_manifest的commit/命令、
metrics/training_history的hint/lr/CC及selected_checkpoint_test_evaluation。
本文件为方法与操作协议，不是已达到0.87的成绩声明。

## 仅训练集的线性读出头迁移诊断（2026-09-10，不采用）

动机：排查能否只将教师解码知识折入**现有**token_projection，而不增大电子网络。
这是一次性CPU诊断，不是新的完整训练/论文测试结果，不把小样本分数写入总表。
没有使用公开测试图片，也没有覆盖或保存新checkpoint；没有占用第三张GPU。

固定学生为87ad（配置`moe_alpha40_extra_control.yaml`），教师为531c，教师缓存39aae6b1，
完整SHA见本页及SAM_TRAINING。代码环境为xml/Torch2.6.0+cu124、当前源码dabb9983，
显式CPU、OMP/MKL各2线程。使用原训练记录按image_id排序的前256张、无增强，batch=4，
前128张拟合映射，后128张仅检查映射。两组都曾参与原学生/教师训练；后128张**不是模型的
独立验证集**，只是线性拟合未用的训练样本，不改变正式10k/5k划分。
256个有序sample_id按每行一个、末尾换行的SHA：
`93fbb898de9b00ca0a1bd03e2f520198f5980687b586b0d673d01c4c2bb1e37b`。

复算步骤与定义：

1. 用`prepare_salicon(...,persist=False)`、`legacy.build_loaders(...,training=False)`的train数据集
   `Subset(range(256))`，提取学生第二融合后的真实`[N,192,14,14]`空间特征F及原密度图。
   读取教师缓存相同ID对应的T，检查全10000个有序ID一致及上述缓存/权重SHA。
2. `X=student.head.token_norm(F.permute(0,2,3,1))`，
   `Y=teacher.decoder.token_projection(teacher.decoder.token_norm(T.permute(0,2,3,1)))`。
   每张196位置，X为192维、Y为128维；浮点64累加，不改变通道/空间顺序。
3. 仅以前128图的25088个位置拟合。中心化后
   `A=Xc.T@Xc/n; B=Xc.T@Yc/n; scale=trace(A)/192`，
   `W=solve(A+lambda*scale*I,B); bias=mean(Y)-mean(X)@W`。
4. 在内存深拷贝教师decoder，保留学生token_norm，token_projection替换为W转置及bias。
   其余decoder参数来自教师，总量仍85412；原学生core及存盘权重均不修改。
   对全部256图输出做原空间softmax，用`independent_cc`逐图float64 Pearson，再分别平均。
5. 空间R²仅检查后128图的投影目标：预测和Y分别减去各自每图的196位置均值，
   `R2=1-mean((pred_centered-Y_centered)^2)/mean(Y_centered^2)`；它不是显著性CC。

|相对ridge lambda|前128图CC|后128图CC|后128图教师投影空间R²|
|---|---:|---:|---:|
|原学生解码头|.87674842|.87137958|—|
|.0001|.62470466|.60183839|.14719877|
|.001|.62436859|.60174785|.14757751|
|.01|.62197662|.60109018|.14667721|
|.1|.61665557|.60008925|.12978821|
|1|.61614250|.60444415|.09248648|

结论只限于这个简单线性迁移：不适合直接替换现有读出头，因此未启动GPU训练或改变正式模型。
它不证明所有特征预训练无效、现有特征完全缺少语义或更大的网络必不可少。
若继续这条方向，应先设计显式的特征级预训练及完整数据验证，不能拿坏的线性移植直接交付。
诊断进程1175074已正常结束，未保留临时特征张量或额外模型权重。
