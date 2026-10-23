# T19：九项 ResNet-20 小模型 baseline

## 2026-10-24 01:41（北京时间）：ABO七项预算诊断完成，原训练剩两项

原九项固定30轮已有7项完成，LSP完成26/30，SALICON完成15/30并进入第16轮；
LGVQ两个独立50轮均已完成。按已完成整轮统计为291/310轮（不是耗时完成率）。
两项真实进程/命令及新鲜progress健康；按近期包含周期评估的吞吐，LSP预计还需20–25分钟，
SALICON约1.9–2.3小时，之后还需末轮独立核验。该时间只是此快照估计，不是完成承诺。
LSP最新已有e25周期PCK=0.2357857143，e20为0.2375714286，e15为0.2362857143；
近期TRAIN loss约0.03396–0.03400，定位指标和loss均趋平台，未见持续加剧的评估恶化。
这更支持目标/优化瓶颈而非已证实过拟合，也不支持仅凭低分断言增加轮数能达到0.5。
SALICON最新e15 CC=0.8685487923；两个数值均非last30最终成绩。

新增CPU七条件已全部完成，进程自然退出，不新增GPU。新独立末轮结果如下；
每个0/1/5预算均在执行前固定，全部报告，不用目标区间倒选某个预算替代原30轮。

| 方向 | 新0轮H@1 | 新固定last1 H@1 | 新固定last5 H@1 | 原固定last30 H@1 | 原图Ours Sim | 用户希望的比较区间 |
|---|---:|---:|---:|---:|---:|---|
| 图搜图 | 0.77875（623/800） | — | — | 0.87125 | 0.8225 | 0.5–0.7225 |
| 图搜文 | 0.015（36/2400） | 0.7404167（1777/2400） | 0.9770833（2345/2400） | 0.9804167 | 0.8058 | 0.5–0.7058 |
| 文搜图 | 0.01（1/100） | 0.65（65/100） | 0.98（98/100） | 0.97 | 0.86 | 0.5–0.76 |

图搜图0轮为原始官方GAP512描述子，没有原30轮的学习后64维投影；跨模态0轮是随机未对齐诊断，
不能当成可用的预训练图文零样本系统。文搜图固定1轮落在描述区间，图搜文1轮与Ours差6.5383个
百分点、图搜图0轮差4.375个百分点，均不足10个百分点；5轮的两个跨模态方向已经快速饱和。
本次没有证据支持在保持同一协议时，零/少训练就能让三方向全部落入区间。

其余正式last30区间核查：LGVQ时间0.6209123位于0.5–0.7044；空间0.5938762高于希望上界0.571
约2.2876个百分点，独立last50的0.5765844也仍高0.5584个百分点。OpenMoji修改格0.324低于0.5；
编辑完整TEST PSNR17.81248 dB满足<22 dB。LSP与SALICON未完成，不用阶段值冒充正式区间结论。
原图为具名历史版本，非所有方法共同last30/50，不能将该描述性区间当作受控同预算消融。

专属独立审计`runs/control/retrieval_fixed_budget_cpu_audit_20261024.json`共135项检查通过，
状态`passed_with_evidence_limits`，SHA256为
`319393f171a141fb5d325a9c4e15bd83251339f99a5a42a5f42dec60f3edea36`。
已核验官方逐tensor/冻结含BN、各条件独立seed42初始化、GRU与投影实际更新、0/150/750优化步、
完整保存排名/ID和H@1等指标重算、采样前缀、旧三个last30权重未变及CPU进程退出。
43份JSON证据已按逐文件SHA同步本地，源码/PT未通过此证据复制通道传输。
全部9600行缓存存储值回读严格验证；重复编码器数值复算仅每个split首尾共64图，不冒称全量双前向。
只保存排名未保存相似度矩阵，不能独立重建分数或并列关系；5轮run未保存e1中间权重，
只可核验相同初始化、首轮loss与采样前缀，不能声称实证其e1权重完全相同。
CPU缓存训练与历史GPU执行不保证逐bit一致。run记录Git commit为`dc98fa25e`，
八份科学源码SHA均匹配发布`f578b5ffb`，该差异来自并行主线更新，不改写原manifest。

## 2026-10-24 新授权：ABO零训练/少量训练与目标区间核查

用户希望本组比例指标尽量不低于0.5，并比原图Ours Sim低至少0.1（10个百分点，非相对10%）；
编辑另按PSNR<22 dB核查。此区间仅用于描述差距，不作为TEST选轮、提前停止或改弱模型的目标。
原九项固定last30和LGVQ独立last50全部保留；LSP/SALICON继续原预算，不中断或立即加轮。

新增独立诊断配置`configs/retrieval_fixed_budget_cpu_20261024.json`，在本次执行前固定全部七项条件：

- 图搜图：官方冻结RN18的原始GAP512特征+L2，0训练，不使用已训练的64维任务投影。
- 图搜文、文搜图：各独立seed42从官方视觉+随机小GRU/投影头初始化，分别0、1、5轮。
- 每个条件仅在指定预算末尾完整评估原TEST一次，全部无条件保留；不看结果选择预算。
- 跨模态0轮没有学习图文对齐，明确标为随机未对齐诊断，不能伪称预训练图文零样本基线。
- 仅CPU执行，不新增GPU占用；保持原224预处理、原数据/候选、H@1、batch32、AdamW .001/
  wd.0001/clip1、FP32、无warmup/cosine、原2-worker persistent采样身份。冻结特征确定性缓存
  只属于新诊断，绑定官方来源/全部有序ID/数据及源码SHA，完整存储值验证与代表性前向复算分开记录。

这是已观察历史TEST后的预算敏感性诊断，不是首次盲测，也不覆盖原30轮正式成绩。图搜图0轮
使用512维描述子，原30轮为训练过的512→64任务头，结构差异必须注明。原easy100为已见商品
留视角检索、标题固定100个；同商品/旋转序列跨划分共有及历史1对字节相同图如实保留，不能将
高检索成绩全部归因于这一对重复。新入口`evaluate_retrieval_budget.py`只负责上述独立profile。

该入口8项专属CPU测试通过并发布`f578b5ffb`，服务器ff-only同步后以独立CPU4线程/nice10启动，
收据`runs/control/retrieval_fixed_budget_cpu_launch_20261024.json`，启动历史PID3247285；
必须每次重新核对实际命令，不将历史PID视为活进程。suite为
`runs/simulation/retrieval_fixed_budget_cpu_20261024`，仅一次全量数据预检/视觉特征提取，
七条件共用新诊断缓存，不新增GPU、不干预在训LSP/SALICON。缓存加速不改变完整推理应包含视觉编码器的计时边界。

原30轮run已有早期周期观察（不是本次新诊断结果）：

| 方向 | 历史e1 H@1 | 历史e5 H@1 | 正式last30 H@1 |
|---|---:|---:|---:|
| 图搜图 | 0.7975 | 0.8300 | 0.87125 |
| 图搜文 | 0.740417 | 0.977083 | 0.980417 |
| 文搜图 | 0.6500 | 0.9800 | 0.9700 |

三个旧run只保存best/last（均末轮），没有e1/e5权重可独立复评，因此这些数值不能替代
新的固定预算last1/last5。旧观察不能成为取消某个新预算或倒选表现较弱epoch的依据。
数据合同上，I2T从固定100个已见标题选择，T2I每标题在2400图库中有24个同SKU正例，
I2I每个已注册SKU有8张正例图库图，命中任一同SKU即H@1成功。这更接近已见商品留视角识别，
预训练外观特征加少量任务头学习快速饱和是合理解释，而非证明开放文本/未见商品泛化。
多正例降低唯一视角匹配要求，但随机H@1仍仅T2I约1%、I2I约0.5%，不能单凭候选正例数量解释高分。

低分项既有曲线审查（0新增模型前向/0优化）：OpenMoji e20→30 TRAIN loss由1.38518降到1.02226
（-26.20%），但changed-cell周期值e10/e15/e20/e25/e30为.3685/.3175/.3655/.3635/.3240。
IoU、前景类别和保留格仍改善，说明综合目标学习与主指标提高不一致，不能仅以loss下降认定
继续训练即可达到0.5。末轮四任务添加/替换/移动/删除准确率为9.2%/42.0%/24.8%/53.6%；
1250个变化格中51.2%仍输出原类别，瓶颈集中在有效改动；任务分类100%或全格95.83%不能替代
修改格32.4%。冻结特征限制是待验证假设，现有记录不能单独证明因果。
编辑e20→30 loss仅降1.338%，VAL PSNR17.87326→17.85081 dB，e10后已平台，暂无直接加20轮的
收益证据；完整TEST仍为17.81248 dB。独立只读证据
`runs/control/frozen_open_editor_budget_review_20261024.json`绑定原文件SHA并保留正式末轮。
LSP先完成30轮；若末段定位与固定TRAIN诊断仍稳定改善，再讨论另开固定50轮对照，不按目标区间挑轮。

## 2026-10-24 00:35（北京时间）：七项固定30轮完成，LSP/SALICON仍在训

当前冻结预训练RN18九项中，LGVQ时间/空间、三个检索方向、OpenMoji及编辑已完成30轮；
仅LSP正在第13轮、SALICON第8轮，不能把周期评估值作为它们的正式末轮结果。
用户批准的LGVQ两项50轮续训也已完成，原30轮权重和结果保持。

| 任务 | 指标 | 当前RN18固定last30 | RN20固定last30 | Ours Sim旧图 | Qwen旧图 | 当前完整参数量 |
|---|---|---:|---:|---:|---:|---:|
| LGVQ时间 | SRCC | 0.6209 | 0.8318 | 0.8044 | 0.7663 | 11,406,277 |
| LGVQ空间 | SRCC | 0.5939 | 0.5480 | 0.6710 | 0.6908 | 11,209,669 |
| ABO图搜文 | H@1 | 0.9804 | 0.6271 | 0.8058 | 0.7358 | 11,240,608 |
| ABO图搜图 | H@1 | 0.8713 | 0.8600 | 0.8225 | 0.8513 | 11,209,344 |
| ABO文搜图 | H@1 | 0.9700 | 0.8000 | 0.8600 | 0.8200 | 11,240,608 |
| LSP | PCK@0.2 | 未完成 | 0.1972 | 0.7348 | 0.7226 | 11,217,070 |
| SALICON | CC | 未完成 | 0.8244 | 0.8625 | 0.8748 | 11,216,641 |
| OpenMoji | 修改格准确率 | 0.3240 | 0.7645 | 0.8880 | 0.8120 | 11,287,478 |
| ABO参考图+指令编辑 | PSNR/dB | 17.81 | 26.11 | 34.28 | 27.26 | 11,628,515 |

当前视觉为ImageNet预训练且严格冻结含BN的RN18；四项文字任务的小GRU与任务头共同训练。
旧RN20为随机初始化、全参数训练，故以上不是仅改变网络深度的消融。两组ResNet均固定第30轮；
Ours/Qwen沿用用户原图具名历史版本，不能声称这些旧数值也来自相同30轮last。
空间Qwen448/CNN224、编辑参考图+指令输入及RGB/latent损失差异仍须披露。所有历史TEST观察保留，
不将固定末轮报告描述为从未观察过TEST的独立盲测。

easy100评价见过商品的留视角检索，TRAIN/TEST共有100个商品及旋转序列。
T08旧`EVIDENCE_20260908.json`记录各split内部文件SHA唯一（4800/2400），跨split恰有1对
字节相同图片；这与同商品跨划分共有是两个事实，不能将全部高分归因于这一对重复。
旧四份CSV SHA与当前身份一致，本轮只读引用旧证据，不重哈希或静默重划分。
该成绩不代表未见商品泛化，也不能称完全无内容重叠；原协议及历史TEST观察如实保留。

新完成图搜文命中2353/2400，运行FP32报告H@1=0.9804166555404663；
文搜图命中97/100，运行报告H@1=0.9700000286102295。
专属独立审计`frozen_i2t_last30_audit_20261024.json`27/27与
`frozen_t2i_last30_audit_20261024.json`26/26通过：完整排名/ID和指标重算、官方预训练逐tensor、
30轮冻结含BN、文字GRU实际更新、仅GRU+头64,096参数/AdamW4500步、源码与GPU释放均核验。
last SHA分别为`c2213e6f9ad63c5120c9c3e4edf4ffe9f9ddcb89e58b7331b042b0261a219ccd`、
`d416558dc59b78f7f8fcb702991f4f66452ef18aca4a700d1cb5d9c707c95ace`。

编辑`byte_gru15m_cached_ram`第30轮完整TEST2304的PSNR为17.812477466546827 dB、
SSIM为0.7835044201670421；VAL30为17.850807555847698 dB，不能混用。
last SHA `61037c593184b32e538c452830827e0f7f7e8af4c2b1f47f9549236d2988155f`。
专属审计`frozen_editor_ram_last30_audit_20261024.json`的30个core checks通过，状态
`passed_with_evidence_limits`：三类编辑各768条、完整参数11,628,515/可训452,003、
AdamW2592→19440、30次官方视觉逐tensor及冻结含BN检查、原e4父SHA不变、源码及GPU释放均核验。
保存逐图PSNR/SSIM标量支持完整TEST均值复算，未保存预测RGB，不能声称逐图像素指标已独立重算。
正式编辑保存e1..4的采样回放SHA，e5..30未保存逐轮运行order哈希/最终Generator；
后段继承采样依赖已验证实现及不中断参照等价测试，不能冒称有26轮运行采样哈希实证。

LGVQ时间固定50轮SRCC为0.6178011327（30轮0.6209123222，差-0.0031111894）；
空间50轮0.5765843891（30轮0.5938761940，差-0.0172918048）。两项追加训练均未改善末轮SRCC，
仍保留约定50轮结果，不从周期TEST挑轮；审计和父30保护见下节。

LSP最新已完成e1..12，当前进入e13。e2 TRAIN loss=0.03434947，e12=0.03405523，
十轮仅再降0.857%；e5→e12下降0.273%，近期波动，已有明显早期loss平台。
已有周期TEST PCK e1/e5/e10为5.35%/16.45%/17.8571%，MPE为92.01/41.74/43.34 px。
没有稳定的评估持续恶化趋势，尚不能认定过拟合；当前也没有完整30轮曲线，不能认定最终收敛。
结合已确认的热图幅度/坐标辅助损失不匹配，现有证据更支持目标/优化瓶颈导致有效拟合不足，
不能只把低分归因于30轮预算。继续按原合同训练，不暗改loss、不擅自加轮。
只读快照`frozen_lsp_curve_epoch11_20261023_163056.json`含epoch11 AdamW3586步及11次冻结检查；
后续轻量曲线快照`frozen_lsp_curve_epoch12_light_20261023_163444.json`仅补既有history。
本轮0新增TEST前向、0优化步。SALICON最近已有e5 CC=0.8626846008也仅为周期观察，不填入正式表。

## 2026-10-24 00:07（北京时间）状态：LGVQ两项50轮均已审计

当前模型为ImageNet预训练且冻结含BN的RN18视觉骨干；文字任务训练27,104参数的小GRU及任务头，
完整模型11.21–11.63M。旧RN20是随机初始化全参数训练，不能混称同一初始化/训练范围消融。
原九项已有四项完成30轮，其余五项继续训练；LGVQ另获准的两个50轮续训均完成独立审计。

| LGVQ指标 | 固定30轮last | 固定50轮last | 50减30 | TRAIN loss 30→50 |
|---|---:|---:|---:|---:|
| 时间SRCC | 0.6209123222 | 0.6178011327 | -0.0031111894 | 0.0054608347→0.0035713869 |
| 空间SRCC | 0.5938761940 | 0.5765843891 | -0.0172918048 | 0.5683587421→0.2865821179 |

单seed42下，追加20轮未改善约定末轮SRCC；损失下降不等于排序泛化提高，不据周期TEST挑轮。
时间50正式run为`lgvq_temporal_continue50_formal_s42_resnet18_imagenet_frozen_single30_20261023`，
last SHA `64d3da8834a270dd99410dac4b8e8e6acd6b4285380a4040a505d8ce29b96e2c`；
父30 SHA仍为`258744c5a06ffff377c0af6baaf0e3befc0b4cb5fc2d5e1a0cfdfb3c650fbd35`。
`runs/control/frozen_lgvq_temporal_last50_audit_20261023.json`的21项核验全部通过，包括50last、
父30保护、AdamW2130→3550、31..50冻结含BN及完整采样回放、558唯一ID/MOS和saved-pred五指标重算、
实际GPU释放。launch820595061与manifest90d11c3的五份记录源码SHA相同；未改原run记录。
空间50审计与证据见下节，原两个30轮结果保持。

图搜图第30轮H@1=0.8712499738（697/800），专属审计`frozen_i2i_last30_audit_20261023.json`24/24通过；
OpenMoji第30轮changed-cell accuracy=0.324，`frozen_openmoji_last30_audit_20261023.json`25/25通过。
两项均已验证完整预测、30轮冻结含BN、官方视觉身份、源码/优化器范围及GPU释放；OpenMoji可训
文字GRU+头110,966参数，不能称文字冻结。其edit IoU/task accuracy只能从保存的逐场景标量复算均值，
原始编辑掩码与任务预测类别未保存，审计如实注明此限制。

该快照在训项（不填入正式末轮成绩）：图搜文24/30、文搜图22/30、LSP9/30、SALICON5/30、编辑16/30。
编辑第15轮VAL PSNR17.8582dB仍非完整TEST最终成绩。最新阶段值不替代last30。

LSP完成e1..8训练loss：0.03556738、0.03434947、0.03426139、0.03422283、0.03414833、
0.03416511、0.03405703、0.03412623。e1→e8降4.052%，但e5→e8仅降0.0647%，属于早期loss平台；
已有e1/e5周期TEST PCK为5.35%/16.45%，平均像素误差92.01→41.74，未出现评估恶化证据。
目前只有两个观察点，不能认定已收敛、过拟合，或据此断言30轮不足。结合下节固定TRAIN的已知
热图幅度/坐标辅助目标不匹配，更应先区分目标/优化瓶颈与单纯预算不足；不擅自改loss或加轮。
epoch8 AdamW2608步（326×8）、8次冻结含BN/无encoder梯度均通过。只读证据
`runs/control/frozen_lsp_curve_epoch8_20261023_160428.json`，本次0新增TEST前向、0优化步。

## 2026-10-23 新授权：工程缓存、LSP曲线审查与LGVQ累计50轮

用户要求加速慢任务、检查LSP是否未收敛/过拟合，并将当前冻结预训练RN18的LGVQ时间、空间
从各自第30轮继续20轮。原九项第30轮协议和封存结果保留；新增LGVQ独立续训run固定报告
累计第50轮last，不按任何指标选轮。`continue_lgvq50.py`恢复原model及AdamW的step/moments，
实际回放原seed42、batch8、2个persistent worker的30轮DataLoader采样，再连续训练31–50；
固定lr.001/有效batch32/FP32/冻结含BN保持。四帧512维冻结特征只计算一次，缓存FP32并对
全部2250 TRAIN+558 TEST样本与原forward做等价检查，缓存与原30轮权重SHA绑定。

编辑工程优化使用`performance_cache.py`：确定性reference/target以uint8去重缓存，恢复时
严格重现原浮点运算顺序；每条image/target/tokens/sample_id/mode全部exact校验后才准入。
缓存只在runs下，不进Git。首版逐条mmap恢复的96条CPU基准仅1.1515倍，未据此宣称训练加速；
优化reader将同一已校验uint8 stores顺序读入CPU RAM（约5.3GB），按整批向量化恢复，模型输入仍
逐元素完全相同。明确兼容原487beceba builder SHA，不改原缓存manifest/字节，RAM中的字节再核SHA。
新tag`byte_gru15m_cached_ram`从旧run最后一个完整epoch恢复model+AdamW，
回放原shuffle序列，仍累计30轮；原run与停机前last原样保留。缓存完整收据、工程smoke通过
及手工核验的交接收据全部具备后才允许新正式worker；smoke不追加TEST模型评估。未交接前
原编辑worker继续运行。训练中断的半轮不计作完整epoch，交接记录必须披露重算范围。
LSP/SALICON含在线随机增强，不能缓存某一次增强后的输入或特征。模型、损失、数据、有效
batch、精度与学习率保持；资源上限仍为当前四卡2/4/5/6，不扩容，不中断其他任务。
协调计划`configs/frozen_cache_lgvq50_20261023.json`保留原在训任务并增加两个50轮续训。

LSP初查：冻结RN18正式e1/e2 loss为0.03556738/0.03434947，头部AdamW实际更新且冻结SHA通过。
只有e1周期TEST PCK=0.0535，尚不能认定30轮不足或过拟合。固定12条TRAIN、168关节CPU
只读诊断（0优化步、0新增TEST）：e2 hard PCK=13.69%，GT热图hard=100%而当前softargmax=2.98%；
模型总loss仅比零图好0.537%，坐标项占88.62%，热图softmax熵0.999993。说明继承的目标幅度/
坐标解码不匹配仍存在，不能只归因epoch少；本轮不暗改loss。后续看完整末段训练loss与已有
周期PCK：持续同时改善支持未收敛，TRAIN改善而评估恶化提示过拟合，双低位平台则优先检查
目标/优化瓶颈。不据TEST选轮，不擅自给LSP延长预算。

LGVQ时间追加训练的首个smoke在跨batch形状singleton FP32核查处出现4.53e-5 logits差异，
随后DataLoader清理异常掩盖首错；不是新增优化步或显存不足。失败run与原30轮SHA保留。
同batch全样本等价门槛仍rtol/atol1e-5；不同batch形状singleton单独rtol/atol1e-4并记录误差与原因，
不声称GPU FP32跨batch位级一致。清理保留原始异常，正常清理失败仍须失败。只修工程检查，
训练模型/微batch/精度/采样/父权重不变，修复后以具名attempt1重新smoke，不默默覆盖重试。

工程上线证据：`runs/control/editor_cache_ram_ready_20261023.json`对全部25,344条构建时exact校验、
持久文件SHA及192条磁盘复验通过。96条同进程CPU batch8基准，原数据5.98781秒、RAM批恢复
0.09069秒，约66.03倍，仅为数据准备。`cached_editor_ram_handoff_20261023.json`记录原编辑
从最后完整第4轮model+AdamW转交，原last SHA为
`5518fb33b6c9ffaabe3cb61cb0f77980d0b032c5b87d6034d496dab45a46d909`，原GPU4保留，
其他在训worker保持原PID。原半轮重算明示，累计仍到30轮。
`editor_ram_first_speed_20261023.json`记录正式训练跨e5/e6的317个微batch耗时15.499秒，
含该区间周期验证约0.04889秒/微batch；原e3→e4完整轮1003.47秒，约0.38714秒/微batch。
同GPU4、不同时段观察约7.92倍提速，**不是受控硬件负载基准或全部九任务统一倍数**；
比较证据`editor_ram_observed_speed_comparison_20261023.json`，父last SHA仍一致。

LGVQ空间续训已完成且独立CPU审计21项通过：

| 固定末轮 | 空间SRCC | TRAIN CE |
|---|---:|---:|
| 原30轮 | 0.5938761940 | 0.5683587421 |
| 追加20轮后累计50轮 | 0.5765843891 | 0.2865821179 |

追加训练未改善末轮SRCC，下降0.0172918048；仍报告约定50轮，不从中挑分数。
run `lgvq_spatial_continue50_formal_s42_resnet18_imagenet_frozen_single30_20261023`，审计
`runs/control/frozen_lgvq_spatial_last50_audit_20261023.json`；父30 SHA未变、AdamW2130→3550步、
20轮冻结含BN及原序列回放、558唯一预测/MOS绑定和saved-pred五指标重算均通过，正式GPU已释放。
launch与manifest Git commit分别ac8b36a/0c9d60f，但五份记录源码SHA均同时匹配两提交，
训练数学相同；未改原manifest。时间50轮以其独立正式run/审计为准，不使用partial smoke分数。

## 2026-10-23 最新修订：完整模型小于1500万，固定当前四GPU

本节覆盖下一节“所有编码器冻结”和三卡上限中的文字/资源部分。用户认为MiniLM文字模型过大，
明确允许小文字编码器从零训练，要求**每任务完整模型严格小于15,000,000参数**，包含冻结视觉、
文字及全部decoder；不是只限制可训练参数。视觉仍为官方ImageNet预训练ResNet-18，11,176,512
参数与BN buffers严格冻结。文字改为原UTF8 embedding259×32+单层GRU32→64，共27,104参数，
随机seed42，文字与任务头共同训练；不能将新方案称为“所有编码器预训练冻结/只训decoder”。
MiniLM旧分支与所有证据保留，新分支显式`--text-encoder byte_gru`，不覆盖旧权重身份。

| 含文字任务 | 完整参数量 | 实际可训练参数量（文字+头） |
|---|---:|---:|
| ABO图搜文 | 11,240,608 | 64,096 |
| ABO文搜图 | 11,240,608 | 64,096 |
| OpenMoji | 11,287,478 | 110,966 |
| ABO参考图+指令编辑 | 11,628,515 | 452,003 |

其余五项模型不变、完整参数量也全部小于1500万。原完整Byte输入不截断；只改变文字编码器与
输入维度相应的文字投影，视觉解码头、数据/划分/损失和末轮合同保持。正式入口拒绝达到15M的
小文字模型，审计视觉SHA不变，另记录训练文字权重变化；optimizer明确包含GRU和任务头。

新计划`configs/resnet18_frozen_smalltext15m_single30_20261023.json`在原冻结suite内接续调度，
四项文字任务使用`byte_gru15m`后缀的新smoke/formal ID；旧MiniLM任务标为superseded并保留结果。
原GPU1上LGVQ时间/空间两个正式PID保持；图搜图和LSP在将GPU1/2/6误解为换卡时已停止，
分别保留第2/1轮last，现以`gpu_expand6`后缀按相同协议重新smoke→30轮；不用中途分数冒充末轮。
停机/原队列证据`runs/control/small_text_migration_20261023.json`及
`visual_gpu126_restart_20261023.json`。协调器替换只接管核验后的LGVQ PID，不重启LGVQ。

用户此前允许六卡扩容，现明确**当前四张卡足够，为其他人保留资源**。协调器显式
`--max-physical-gpus 4`，候选固定当前物理GPU2/4/5/6，不迁移或中断现有worker；
每卡最多三个登记T19 worker；第三个仅在已有两项均进入CUDA、空余显存≥6144MiB且利用率≤50%
时加入，保护原先按两任务准入的在训worker。忙卡等待，不干预其他任务。此资源授权
覆盖旧execution profile三卡及此前六卡限制；原在训worker的历史manifest不改写。仍为单seed42、30轮、
固定lr.001、AdamW wd.0001 clip1、有效batch32、FP32 TF32关、无warmup/cosine，三个检索H@1。
全部仅报第30轮last完整复评，编辑完整2304 TEST；不根据任何指标选轮。LSP原损失问题仍披露。

两个保留的LGVQ正式run现已完成，独立CPU核验epoch30、last文件SHA、各558条唯一预测及
全部30次视觉参数/BN buffers冻结SHA均通过。此处是封存30轮成绩；新增50轮续训遵循上节授权：

| LGVQ指标 | 冻结预训练RN18第30轮 | RN20全参第30轮 | RN32全参第30轮 | Ours Sim旧图 | Qwen旧图 |
|---|---:|---:|---:|---:|---:|
| 时间SRCC | 0.6209123222 | 0.8317808079 | 0.8024708890 | 0.8044 | 0.7663 |
| 空间SRCC | 0.5938761940 | 0.5479703205 | 0.6362456795 | 0.6710 | 0.6908 |

两个run ID分别`lgvq_temporal_formal_s42_resnet18_imagenet_frozen_single30_20261023`、
`lgvq_spatial_formal_s42_resnet18_imagenet_frozen_single30_20261023`；审计证据
`runs/control/frozen_lgvq_last30_independent_audit_20261023.json`。时间last SHA
`258744c5a06ffff377c0af6baaf0e3befc0b4cb5fc2d5e1a0cfdfb3c650fbd35`，空间last SHA
`10b93e8b6022e23e2010638053b91d0ba9fbfd6ff2f171b07495cbbec7fcecca`。
冻结视觉11,176,512参数，时间/空间可训练头分别229,765/33,157。
初始化和优化范围与旧CNN不同，不作为只改变网络深度的消融；Ours/Qwen为原图具名历史版本，
未宣称均为相同30轮末轮，空间448/224分辨率差异保留。TEST周期被观察，末轮固定选取；不是新盲测。

## 2026-10-23 最新协议更正：预训练冻结骨干，仅训练decoder

用户明确不希望baseline全网络重新训练，要求核实并重新尝试。实际核查：**旧ResNet-20、
CIFAR ResNet-32均随机初始化、全参数训练；旧`lgvq_temporal_resnet18`也明确使用
`torchvision.resnet18(weights=None)`且AdamW包含全部参数**。不能把旧成绩称为冻结预训练
特征的decoder基线。以下旧结果全部保留，旧RN18任务名/构造器/权重加载语义不变。

原RN32 suite已按这次用户协议更正停止，未删除权重。六项正式第30轮last结果：

| 任务 | 原RN32全参末轮成绩 |
|---|---:|
| LGVQ时间 SRCC | 0.8024708890 |
| LGVQ空间 SRCC | 0.6362456795 |
| ABO图搜文 H@1 | 0.9612500072 |
| ABO图搜图 H@1 | 0.8387500048 |
| ABO文搜图 H@1 | 1.0000000000 |
| OpenMoji changed-cell accuracy | 0.8850000000 |

LSP/SALICON/编辑停止时分别保留第25/12/14轮last，不当成正式30轮结果。
服务器停机收据：`runs/control/random_full_training_stopped_for_frozen_pretrained_20261023.json`，
记录进程命令身份、全部原queue、保留权重SHA及GPU释放情况，不恢复该旧suite。

新profile/队列计划为`configs/resnet18_imagenet_frozen_single30_20261023.json`，
新suite `resnet18_imagenet_frozen_single30_20261023`。使用标准ImageNet ResNet-18，官方
`ResNet18_Weights.IMAGENET1K_V1`，移除ImageNet分类器。官方checkpoint
`resnet18-f37072fd.pth`完整SHA256：
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。
只训练decoder；视觉骨干11,176,512参数全部冻结，BN统计固定eval，encoder前向no_grad，
优化器/裁剪仅含可训练任务头。逐轮记录含所有参数及BN buffers的state SHA并断言前后相同，
记录真实官方缓存文件SHA、逐张量匹配、optimizer参数数和无encoder梯度证据。
官方旧PT未保存的20个BN计数器由torchvision严格补零并具名记录，不能放宽其他张量核验。

| 任务 | decoder可训练参数量 | 含全部冻结编码器完整参数量 |
|---|---:|---:|
| LGVQ时间 | 229,765 | 11,406,277 |
| LGVQ空间 | 33,157 | 11,209,669 |
| ABO图搜图 | 32,832 | 11,209,344 |
| LSP | 40,558 | 11,217,070 |
| SALICON | 40,129 | 11,216,641 |
| ABO图搜文 | 57,472 | 33,799,360 |
| ABO文搜图 | 57,472 | 33,799,360 |
| OpenMoji | 104,342 | 33,846,230 |
| ABO参考图+指令编辑 | 445,379 | 34,187,267 |

LGVQ沿用旧RN18的512维特征及质量头（时间3584→64→5；空间512→64→5）。
图搜图采用512→64投影与原精确SKU检索。LSP/SALICON使用四级冻结空间特征的轻量融合decoder，
每级1×1投影32通道、插值到stride4求和、3×3卷积及输出层，避免最深7×7图直接放大造成
定位瓶颈；这是新适配头，不宣称纯初始化消融。输入仍使用既有ImageNet归一化/各任务几何，
不额外调用官方resize/crop改变坐标。LSP仍保留原MSE+坐标辅助loss，本轮不混入未批准的损失修正；
前述已知目标不匹配仍是限制，冻结预训练不保证自动修复。

用户已进一步确认：**所有编码器均用预训练并冻结，只训任务头**。四项含文字任务使用
`sentence-transformers/all-MiniLM-L6-v2`，固定官方commit
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`；`model.safetensors` SHA256
`53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`。
去除未使用的BERT pooler，22,565,376个文字参数全部来自预训练、全部冻结；加载时missing/
mismatched/error均须为空，仅允许明确去掉的pooler两张量。严格按attention mask均值池化并L2
归一化到384维，再训练任务投影/FiLM/decoder。Byte token严格逆解回原始全文，256 wordpiece
超限直接拒绝，绝不截断。原100标题、6000 OpenMoji指令及636编辑指令CPU全量检查，最大
长度分别66/16/36；均为英文，部分标题含Unicode标点。记录
`runs/control/minilm_real_input_probe_20261023.json`，无缺失权重/新增随机编码器层。
服务器无法解析HF时通过本机官方HTTPS下载，仅传输模型资产并逐文件SHA验收，凭据和模型不进Git；
收据`runs/control/minilm_pretrained_cache_verified_20261023.json`。不传输覆盖源码。

OpenMoji仅增加512→64视觉投影和384→64文字投影，保持6×6网格任务头；编辑使用四级冻结
视觉skip各投影64、三个双卷积融合块及RGB输出，输出原256×256，仍为参考图+指令输入。
两者是新的适配decoder；原有损失、数据身份、完整TEST评估协议保持。

全部九项按LGVQ优先进入同一新suite；每项smoke通过再正式30轮，所有正式结果
固定第30轮last完整复评。seed42、固定lr0.001、AdamW wd0.0001 clip1、有效batch32、FP32
TF32关、无warmup/cosine和最多三卡/每卡两任务保持。新profile不使用旧随机GRU。
新suite与所有旧run分离，真实启动/进度以专属queue_status和run_manifest为准；
发布后启动收据为`runs/control/resnet18_frozen_launch_20261023.json`。

## 2026-10-23 新增：CIFAR ResNet-32九项对照

用户要求使用更大的CIFAR ResNet重跑所有任务。采用同一6n+2/option-A实现，
三个stage由[3,3,3]加深为[5,5,5]，通道保持[16,32,64]，3x3 stride1 stem、无maxpool、
输出64维及全部任务头/byte-GRU保持原结构，随机初始化无预训练。
视觉骨干463,504参数，为ResNet-20的1.723倍。

| 任务 | ResNet-32完整参数量 |
|---|---:|
| lgvq_temporal | 492,565 |
| lgvq_spatial | 467,989 |
| abo_image_to_text | 498,928 |
| abo_image_to_image | 467,664 |
| abo_text_to_image | 498,928 |
| lsp | 482,430 |
| salicon | 482,001 |
| openmoji | 537,478 |
| abo_conditional_edit | 545,155 |

执行profile兼队列计划：`configs/resnet32_single30_20261023.json`。所有九项单seed42、
30轮、固定lr0.001、AdamW wd0.0001 clip1、无warmup/cosine、有效batch32、FP32 TF32关闭。
训练数据/划分/预处理/增强/任务损失/指标沿用ResNet-20；三个检索方向统一H@1；
固定第30轮last_checkpoint.pt复评，图文编辑最终评估完整TEST，不用TEST/VAL/TRAIN loss选轮。
保留原ResNet-20/18所有结果，不继承其权重，不改变Qwen/Ours。

统一队列`queue_remaining.py --backbone resnet32`现覆盖九项，LGVQ优先；两个跨模态方向
调用原`train_easy100.py`，其余调用`train_tasks.py`。每项两batch smoke及完整评估、
权重保存/加载复评通过后才正式30轮。使用同一T19全局锁，最多三张物理GPU、每卡最多
两个已核验T19 worker；空闲前不占显存，不与其他任务抢卡，队列自动等待可用卡。
正式suite为`resnet32_single30_20261023`，每run保存backbone、参数量、数据身份、commit和
原始命令；阶段以服务器queue_status及run_manifest/result为准，不能把排队说成已训练。

## 2026-10-23 最终交付：全部固定第30轮last权重复评

本节为当前报告口径，覆盖下方所有历史最佳轮次成绩。ResNet-20九项及ResNet-18时间质量对照
共十组均完成30轮，并在CPU FP32严格加载原始last权重完整复评。seed42、lr0.001恒定、
AdamW wd0.0001、无warmup/cosine。未追加训练，不按TEST/VAL/TRAIN-loss挑epoch。

| 任务 | 指标 | ResNet-20第30轮 | 完整参数量 | Ours Sim.旧图 | Qwen旧图 |
|---|---|---:|---:|---:|---:|
| LGVQ时间 | srcc | 0.8318 | 298,133 | 0.8044 | 0.7663 |
| LGVQ空间 | srcc | 0.5480 | 273,557 | 0.6710 | 0.6908 |
| ABO图搜文 | hit_at_1 | 0.6271 | 304,496 | 0.8058 | 0.7358 |
| ABO图搜图 | hit_at_1 | 0.8600 | 273,232 | 0.8225 | 0.8513 |
| ABO文搜图 | hit_at_1 | 0.8000 | 304,496 | 0.8600 | 0.8200 |
| LSP | pck_at_0.2_torso | 0.1972 | 287,998 | 0.7348 | 0.7226 |
| SALICON | cc | 0.8244 | 287,569 | 0.8625 | 0.8748 |
| OpenMoji | changed_cell_accuracy | 0.7645 | 343,046 | 0.8880 | 0.8120 |
| ABO参考图+指令编辑 | psnr_db | 26.11 | 350,723 | 34.28 | 27.26 |

ResNet-20视觉骨干269,072参数；各任务头/必要的byte-GRU均计入完整参数量。
ResNet-18时间质量完整11,406,277参数，第30轮SRCC **0.7181948871342798**；
原第10轮0.7793529367887383仅保留历史观察，不作为当前成绩。
图文编辑第30轮完整TEST2304样本PSNR **26.10548284318712 dB**、SSIM **0.8406315955540372**，非VAL。

历史Ours/Qwen值逐项来自用户本聊天最初截图，仅供注明版本的系统比较。历史检索R/H标签
不重写旧文件，本次三个检索方向均精确SKU H@1。不能宣称同初始化/预训练预算的消融：
小模型从零训练，原Qwen参考有预训练；空间输入Qwen448/CNN224。图文编辑实际输入包含
参考图与指令，RGB MSE+梯度L1与原Qwen latent损失不同，不能凭截图“文生图”名称视为纯文本生成。
旧训练按旧规则保存best、周期观察过TEST，用户在观察后改为末轮报告；这些记录保留，
不将本结果描述为从未观察过TEST的盲测，也不改写原run的训练commit和旧result。

证据：`runs/control/last_epoch_review_20261023.json`与`last_epoch_final_audit_20261023.json`；
每个run下`last_epoch_review_20261023/result.json`及`fixed_weight_predictions.json`。
十组末轮checkpoint SHA、epoch30、seed/优化协议、完整预测数量均独立核验，原训练PID已释放GPU。
全部训练队列complete，CPU复评进程正常退出；历史失败smoke及CUDA前退出证据保留。

| 任务 | 正式run ID | 完整评估样本数 |
|---|---|---:|
| abo_image_to_text | `easy100_image_to_text_s42_easy100_single30_shared_20261023` | 2400 |
| abo_text_to_image | `easy100_text_to_image_s42_easy100_single30_shared_20261023` | 100 |
| lgvq_temporal | `lgvq_temporal_formal_s42_remaining7_20261023` | 558 |
| lgvq_spatial | `lgvq_spatial_formal_s42_remaining7_20261023` | 558 |
| abo_image_to_image | `abo_image_to_image_formal_s42_remaining7_20261023` | 800 |
| openmoji | `openmoji_formal_s42_remaining7_20261023` | 1000 |
| lgvq_temporal_resnet18 | `lgvq_temporal_resnet18_formal_s42_remaining7_20261023` | 558 |
| lsp | `lsp_formal_s42_remaining7_20261023_attempt2` | 1000 |
| salicon | `salicon_formal_s42_remaining7_20261023` | 5000 |
| abo_conditional_edit | `abo_conditional_edit_formal_s42_remaining7_20261023` | 2304 |

原始逐样本文件、PT及私有审计JSON不进入Git；本节只发布聚合指标和run身份。

## 2026-10-23 接续七项，LGVQ优先

ABO双向各30轮已完成，两个方向均选第25轮固定复评：H@1分别96.6667%和95.0000%；
第30轮分别62.7083%和80.0000%。两者独立列明，均为TEST选模的开发指标，不隐藏末轮波动。
来源：`easy100_single30_shared_20261023`及其两个formal run；自有GPU进程释放检查通过。

LGVQ两项正式30轮亦完成，均选第30轮固定权重复评：Temporal SRCC
**0.8317808078544658**、Spatial SRCC **0.5479703204800555**。
来源`lgvq_temporal_formal_s42_remaining7_20261023`及
`lgvq_spatial_formal_s42_remaining7_20261023`的result/history/fixed_weight_predictions。
两项均是TEST选模开发指标；与初始表Qwen .7663/.6908和Ours仿真 .8044/.6710
仅作明确版本的系统比较，空间输入448/224差异仍须披露。

用户明确继续剩余七项，优先LGVQ时间/空间两项。全部沿用下节单seed42、30轮、固定lr0.001，
不使用warmup/cosine。`queue_remaining.py`使用同一全局T19锁，最多三张物理卡，每卡最多
两个本任务worker；先自动完成各任务的两batch smoke/完整评估/保存复评，再正式30轮。
已就绪任务写入本次run的plan，未核验适配器不加入；同一队列支持追加任务，最后封闭plan。
独立任务失败保留日志并阻止该项formal，不静默重试，也不杀其他正常任务。

LGVQ复用原始RGB四帧uint8缓存，不使用Qwen特征或教师分数。manifest
`607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc`，
帧缓存`d97ffe7d5e936a32cbf7dcbe6ee08859a66a3c16e627234cec9f74eb7d2f0f4b`，
1691339588字节；2250/558、所有ID/顺序/视频路径/两类MOS、四帧位置和0.65裁剪均检查。
另对5条固定视频解码逐像素比对缓存；不把该抽查说成全部原视频SHA审计。
视觉输入224、ImageNet归一化，空间Qwen原448与CNN224差异明确披露。
每次8视频、累积4次到有效batch32（末组按实际样本数加权），BN按实际8视频/32帧计算；
TRAIN MOS min/max生成五档和分值，CE训练，softmax加权评分，TEST SRCC选模并完整逐样本保存。
任务专属训练入口`train_tasks.py`记录源码commit/逐文件SHA、原始命令、环境及数据身份。
追加其他任务源码不会中断已加载旧commit代码的任务，每个新worker记录自己的已发布commit。

其余五项训练合同已实现于`other_data.py`/`task_problems.py`，CPU梯度检查覆盖全部五项：

| 任务 | 数据及监督 | 选模 / 指标 |
|---|---|---|
| ABO图搜图 | 原1600 TRAIN/图库、800 QUERY；协议与每张图SHA核验；每批配对同SKU多视角，不把正例作负例 | TEST开发H@1，原完整图库排名 |
| LSP | 原10428/1000；注释MAT的历史SHA、原人体裁剪/增强/关节顺序、热图MSE+0.1坐标损失 | 按原Qwen教师最低TRAIN loss选权重；原torso PCK@0.2 |
| SALICON | 原10000/5000，原fixation JSON SHA、sigma19和224密度缓存；禁止下载/重建缓存；原KL/CC/SIM/NSS损失 | public TEST CC开发指标，原指标实现 |
| OpenMoji | 原layered_anchor6_svg_v3的5000/1000、数据seed73保持；训练seed42；原标签损失12/3及1/1.25/.15/.1 | 逐场景changed-cell accuracy再平均，不按编辑格总数加权 |
| ABO图文编辑 | 原20736/2304/2304与三份manifest SHA；调用原渲染器/TRAIN设计目录，不读取任何教师特征；模型仅看参考RGB及byte指令 | VAL逐图PSNR选模，再一次固定TEST；原Gaussian SSIM |

图文编辑原数据类返回的mask、target及空的兼容text字段均不会进入模型forward；
`editor_dataset.py`重用确定性配对/渲染规则，但不加载Qwen或指令特征缓存。
编辑损失沿此前批准方案为RGB MSE+0.05梯度L1，与原Qwen latent损失不同；它不是
同loss/预训练预算的消融。LSP/CNN和SALICON/CNN同样使用ImageNet通道归一化，非Qwen processor。
每项只有CPU真实数据前检成功后才加入运行plan；加入后依次进行真实GPU smoke与正式30轮。
OpenMoji按原Qwen规则以source_grid组合保留格与预测编辑格；先逐场景算changed-cell准确率，
再对1000场景平均，不用GT target选择输出。source_grid只用于原定义的输出组合及指标。

本轮图搜图第一次smoke在CPU数据核验后遇到其他用户进程进入GPU，准入在CUDA前拒绝。
原失败日志保留，通过显式control-plan attempt1重排，不冒称首次成功，不自动无限重试。
新worker完成CPU核验后如设备变忙，会在无CUDA上下文状态重新等候最多600秒。
调度器支持核验PID/进程组/任务命令身份后接管已有worker；替换调度器不重启训练。
LSP首次formal在CUDA前遇到另一窗口fetch更新origin/main后HEAD不再等于远端最新值，
旧守卫因此拒绝。现改为验证HEAD确实位于已发布origin/main历史且T19源码干净，
不会因远端新增其他任务提交误判旧已发布commit；未发布的本地HEAD仍拒绝。
该次没有训练权重，保留原日志并在control-plan显式登记formal attempt1，不混作续训。
随后证实服务器另一窗口已直接发布新HEAD、但本机origin/main引用仍旧。守卫补充只读
`ls-remote`精确HEAD证据，不在worker中fetch或修改Git；同步中尚不能证明已发布时在CUDA前等候。
LSP这两次源校验均在训练前退出，明确登记attempt2重新启动，原失败日志不覆盖。

## 当前生效协议：2026-10-23 用户修订

本轮只负责 `LightGenV2/tasks/t19_resnet20_baselines/`，未修改其他任务或历史测量。
按用户最新要求，以 `configs/execution_single_seed30_20261023.json` 覆盖下方旧训练预算：

| 项目 | 当前设置 |
|---|---|
| seed | 只用42；取消43/44队列 |
| epochs | 30，从随机初始化开始 |
| 学习率 | 固定0.001，取消warmup和cosine |
| 优化器 | AdamW，weight decay 0.0001，clip1 |
| batch / precision | 32 / FP32，TF32关闭 |
| 检索指标 | 三方向统一Hit@1（H@1），仍按精确SKU命中；历史R@1不改写原报告 |
| GPU | 最多三张物理卡，每卡最多两个本任务worker；每worker只见一张卡 |

三个seed是此前额外提出的预算，并非原baseline要求，现已撤销。此前两个100轮计划
在各完成52轮后由用户要求停止，best/last及历史全部原位保留，另有
`easy100_20261023_local_continue/user_protocol_change_receipt.json` 记录停止、SHA及释放。
旧队列的43/44未启动，coordinator已退出；新run不续用52轮权重，不把旧成绩冒称30轮成绩。

GPU共享仅放行同一coordinator登记且PID命令身份再次核验的T19训练进程，拒绝与不明
进程混占；启动第二个worker前须至少剩余6144MiB且即时利用率不高于95%。队列全局
文件锁同时覆盖smoke/formal，最多每卡两个任务，不额外启用系统级MPS服务。
DataLoader改为持久spawn worker，减少每轮重建进程的开销，退出时关闭自己的worker。
同卡先做两方向并发GPU smoke，确认保存、固定权重复评及显存释放后再启动正式30轮。

本轮ABO结构仍为ResNet-20+byte-GRU、304496个全可训练参数，损失、划分、候选库与
epoch1/每5轮/末轮选最佳Hit@1的口径保持原定义，测试选模指标标为开发指标。
ABO的Qwen参照是完全冻结、无任务内训练的原生2048维模型；本轮30轮小模型并不宣称
与Qwen具有相同预训练知识或任务内训练预算。剩余七项仍未启动，不能从本轮ABO推断成绩。

以下各节是此前准备和100轮计划的历史记录；与本节冲突时，以本节及新执行profile为准。

## 2026-10-23 本地执行会话接手：连接恢复，分任务闭合执行条件

本会话已通过 SSH 实测连接训练服务器，`git fetch origin` 成功；初始两端均为
`main@c35d65aca077b2c251e92b95a357fccd783c8cbd`。下方“网络禁用/只读”的
描述是此前会话的历史状态，不代表当前环境。此前接续会话已结束，无并行T19写入。

继续既有九项、三个seed和最多三张GPU授权；按任务逐项校验、发布、smoke及正式训练，
无需等待全部九个适配器同时完工。每个被启动的任务都必须先闭合自己的源码、数据、
损失、指标和设备准入条件；未绑定的任务不得进入队列。原架构初稿保持不变。

新增 `train_easy100.py`，两方向独立从零拟合，完整TEST候选库评估（2400×100或100×2400），
每epoch1/5倍数/final按TEST Hit@1选模并标为开发指标；保留完整排名及固定权重复评。
每次启动前重新校验7200图片及CSV；只接受已发布main上的干净T19源码和单GPU UUID，
拒绝已有计算进程、显存>100MiB或利用率>5%的卡。spawn数据worker不继承CUDA。
FP32禁用TF32，AdamW、100epochs、batch32、warmup5+cosine，clip1；仅保存best/last。
smoke与simulation分目录，smoke不作为正式成绩。运行配置、命令、数据身份、环境和PID
在每个run的 `run_manifest.json`，失败状态也保留。其他窗口暂存文件不纳入本任务提交。

服务器原始easy100：4800/2400图、100标题、7200图片共389222242字节，内容manifest
SHA256为 `1bc9dde3407a3a288973d760cc3c62326d502f5d73728a9cca0754bf42b2739e`，
已逐文件通过历史收据。服务器41项CPU测试通过。首次两方向2-batch GPU短测完整完成、
固定权重复评和保存成功、退出后5/6卡恢复10/14MiB；提前break引发DataLoader退出警告，
现改为smoke专用Subset自然耗尽迭代器并重新短测，不忽略该警告进入正式队列。

新增 `queue_easy100.py`：Linux文件锁防止同任务重复队列，最多三个单卡worker，
两个方向×seed42/43/44共六次拟合；每个worker前重新准入，繁忙等待，不占位。
必须先有v2两方向成功smoke收据；队列运行中T19源码树变化会停止并标错，不混用源码。
其他任务提交不影响T19源码时可继续，各run仍记录启动时已发布的HEAD。
每run完成后进程退出释放显存；队列结束检查自有进程组是否仍占GPU。终止只针对
自己创建的进程组，保留原始日志、checkpoint和失败状态。剩余七项适配与训练尚未完成。

```bash
CUDA_VISIBLE_DEVICES=<idle-GPU-UUID> python -m LightGenV2.tasks.t19_resnet20_baselines.train_easy100 \
  --direction image_to_text --data-root data/abo_easy100_dataset_20260906 \
  --gpu-uuid <idle-GPU-UUID> --run-dir LightGenV2/tasks/t19_resnet20_baselines/runs/simulation/easy100_i2t_s42_20261023
```

## 2026-10-23 接续：已读取原对话，补充 easy100 数据适配

已通过浏览器读取“推荐小模型基线”从初始选型到最后排障的文字消息，沿用既有
训练授权、九项范围和最多三张GPU上限，不要求重新确认同一架构。
新会话实测仍有 `CODEX_SANDBOX_NETWORK_DISABLED=1`，服务器TCP连接返回10013；
`.git` ACL有拒绝写入项，仍不能发布、同步或启动服务器训练。未尝试绕过限制。

新增 `easy100_data.py`：图搜文/文搜图共用原始数据读取器，依据
`maintenance/storage/T08_CONTENT_ASSETS_20261004.json` 校验四份CSV的大小/SHA、
7200张原图的内容摘要、TRAIN4800/TEST2400、100个官方标题及每SKU48/24张。
复用既有split身份检查器；额外拒绝错配label/SKU、超长标题和越界图像路径。
标题保留CSV原序，不假设label等于候选库行号；SKU仅作为监督/评估元数据。
保留原协议已披露的跨划分图像内容重复，不静默去重或重划分。

CNN预处理明确为RGB、224中心裁剪、bicubic、[0,1]及ImageNet通道归一化，
不使用Qwen processor或教师缓存，不冒称与动态分辨率Qwen逐像素相同。
适配器检查命令只读数据，不训练、不初始化CUDA：

```bash
python -m LightGenV2.tasks.t19_resnet20_baselines.easy100_data --data-root <original_easy100_root>
```

新增6项CPU检查，总计39项通过。新增检查覆盖错配SKU、划分交叉、样本计数、
标题字节上限、路径边界、伪造CSV拒绝及图像张量/元数据分离。
本机没有原easy100完整数据，因此这些是适配器合同测试，**不是正式数据通过**。
九项完整适配、训练入口、GPU调度及服务器数据/源码核验仍未完成；无GPU占用，
新源码未提交发布。此处是原“当前交付与未完成项”之后的接续进展。

## 2026-10-23 后续：用户已授权尝试，双向结构相同

用户在架构说明后回复“你尝试一下吧”，继续九项实验并最多三张GPU的范围。
执行授权单独保存在 `configs/execution_authorization_20261023.json`；下方初稿和
`suite_proposal.json` 保留审阅时状态，不再因为初稿写“等待确认”重复要求用户批准。
真实数据适配器、训练器和服务器前检仍须完成，不能把授权记录当作任务已启动。

图搜文与文搜图的 `build_model` 返回同一个 `CrossModalRetrievalModel` 类：
ResNet-20→Linear64，byte-GRU→Linear64，两侧L2归一化，总参数均304,496。
两方向只改变相似度矩阵的查询轴、训练loss及评估候选库；当前计划分别拟合权重。
支持同一权重双向评价，但这与当前方向独立拟合方案不同，不静默混报两种结果。
`retrieval.py`现已实现两个方向的多正例损失、可选对称损失和完整精确SKU稳定排名。
同商品多视图均为正例，文搜图的24个正例不当成24次独立标题查询。
新增测试检查两方向逐层参数形状完全相同、正例梯度、转置方向和指标分母。

本轮重新只读诊断：SSH认证前被拒；直接TCP `connect_ex` 返回Windows10013，
DNS查询也返回Access denied，指向当前执行环境的网络访问限制，未证明服务器离线。
Git只读远端查询另显示当前代理 `127.0.0.1:7897` 连接失败；没有更改用户代理或绕过限制。
`.git`仍为当前会话只读路径，不尝试通过另一个索引/工程副本绕过权限。
用户贴的nvidia-smi是外部状态快照，不是本会话实时检查或GPU预约。
本任务尚未连接成功、未提交推送或训练，没有占用服务器GPU。

2026-10-23。用户要求先完成准备并展示架构，经其确认后才正式跑服务器GPU。
本目录是独立 task，不修改九个来源任务的模型、数据划分、正式权重、光学合同或成绩。
不创建分支、worktree 或工程副本。当前没有训练入口，也没有启动任何GPU smoke或正式训练。

## 当前交付与未完成项

- 已定位九行原表：`LightGenV2/reports/20260928_nine_task_a100_table/latest_nine_task_summary.csv`。
- 已实现九种CPU可构造的模型原型、逐组件参数统计及合成前向/梯度检查。
- `configs/suite_proposal.json` 逐项记录来源代码、数据数量、候选库、指标及待绑定项。
- `inspect_preparation.py`只读核查来源文件并记录SHA；可在服务器查询nvidia-smi，但不分配GPU。
- **未完成**：服务器实时源码/overlay/数据SHA/环境审计、正式数据适配器、训练器及GPU调度器。
- **未完成Git同步**：本环境 `.git` 仅可读，`git fetch origin` 失败于无法写入
  `.git/FETCH_HEAD`；未尝试提交/推送，也没有覆盖其他窗口的暂存改动。
- **服务器未连接成功**：SSH到用户指定主机/端口在认证前返回
  `connect ... Permission denied`。没有取得实时HEAD、GPU占用或数据路径；历史记录不冒称现状。
- 口令、主机连接脚本不写入本目录或Git。本轮没有绕过环境限制复制源码。

因此本交付是**可审阅的本地架构准备**，不是“服务器已就绪”或“已完成九项实验”。
本次CPU验证记录见 `reports/preparation_cpu_20261023.json`；它不含数据集成绩或GPU计时。

## 九项范围与 baseline 溯源

下面九项来自原表01a、01b、02—08；不包含Caltech检索，也不将第九项误写为ABO文搜图。
原表是2026-09-28历史比较表，其精度、实拍与计时版本不一定等于当前任务最终版本。
本次只复用对应的任务合同，不把旧表的“Ours”数字默认为新的同期对照。

| 原行 | 实验 | 本地 baseline 来源（相对 LightGenV2/tasks） | 固定数据/候选合同 | 主指标 |
|---|---|---|---|---|
|01a|LGVQ Temporal|t06_video_quality_assessment/quality_token_resolution.py、quality_token_train.py|TRAIN2250 / TEST558视频；每视频4帧|Temporal SRCC|
|01b|LGVQ Spatial|t06_video_quality_assessment/quality_token_resolution.py|同2250/558，各自独立训练和权重|Spatial SRCC|
|02|ABO图搜文|t08_abo_image_text_retrieval/baseline_5090d.py、abo_backbone_baselines.py|TRAIN4800 / TEST2400图，100官方标题|R@1，精确SKU|
|03|ABO图搜图|t07_abo_image_retrieval/standalone/enrolled_abo.py；T08三方向baseline|既有ABO200协议：1600 TRAIN/图库，800查询|R@1，精确SKU|
|04|ABO文搜图|t08_abo_image_text_retrieval/abo_backbone_baselines.py；原表raw_remote/formal_04_baseline|同easy100 TRAIN4800，100标题查询2400 TEST图库，每标题24正例|Hit@1，精确SKU|
|05|LSP关键点|t02_keypoint_detection/baseline_5090d.py、reports/reproduction/BASELINE_METHODS.md|TRAIN10428，官方TEST1000，14关节|PCK@0.2|
|06|SALICON显著性|t03_saliency/aligned_baseline.py、reproduce_baseline.py|TRAIN10000 / public TEST5000|CC|
|07|OpenMoji语义交互|t04_semantic_interaction/baseline_shared_readout.py、run.py及分层应用baseline配置|layered_anchor6_svg_v3，TRAIN5000 / TEST1000|修改格准确率|
|08|ABO图文条件编辑|t12_text_to_image/train_matched_qwen_baseline.py、product_unified_edit_data_v2.py|TRAIN20736 / VAL2304 / TEST2304，三种编辑模式|逐图PSNR均值、SSIM|

原表行03、04与后来封存的rank72/.85实拍模型不能交换PT/精度/计时。
LSP原表Qwen基线是Deconv128历史版本，不用Deconv40的成绩替代。
OpenMoji复用分层应用数据及标签，不改用robust rank64或旧等尺寸网格。
T12主线已补齐同任务Qwen28训练，不使用只训练过旧灯类的迁移诊断成绩作为公平质量对照。
所有来源仍需与服务器实际运行代码及overlay逐项核验。完整文件SHA由只读检查器输出。

## 固定视觉骨干

采用原论文CIFAR ResNet-20 **Option A**：

| 层组 | 构成 | 224输入的空间尺寸 |
|---|---|---|
|Stem|3×3 Conv，3→16，BN，ReLU；stride1，无7×7大步长stem/maxpool|224×224|
|Stage1|3个BasicBlock，每块两个3×3卷积，16通道|224×224|
|Stage2|3个BasicBlock，32通道；首块stride2|112×112|
|Stage3|3个BasicBlock，64通道；首块stride2|56×56|
|池化|按任务取全局平均或保留空间图|64维或64×56×56|

共19个主路径卷积；原10分类Linear是第20个有权重层。
跨阶段shortcut用下采样和通道补零，不添加可训练1×1投影。
去掉原10分类头后的骨干为 **269,072参数**；加原64→10头为269,722。
不将任务头的新增卷积伪称仍是纯20层分类器，方法名应写“ResNet-20 backbone + task head”。

保留任务所需分辨率，不为模仿CIFAR把所有输入降为32×32。
因此参数量虽小，224/256输入的激活、计算量和训练成本并不等于CIFAR原实验，速度需要实测。

## 九项头部与完整参数

所有任务的视觉骨干分别训练，不跨任务共享训练所得权重。
文字相关四项使用可训练 **UTF-8 byte embedding(259×32) + 单层GRU(32→64)**，
共27,104参数，包括PAD/BOS/EOS及完整词嵌入；不加载Qwen/CLIP/tokenizer或大模型特征缓存。
输入只含原英文标题/编辑指令，不用SKU编号代替文字，不根据测试集建立词表。
512-byte输入上限超出即拒绝，不静默截断；真实标题/指令长度待数据前检验证。

| 实验 | ResNet-20之后的任务结构 | 全部参数 |
|---|---|---:|
|LGVQ Temporal|四帧共享骨干→4×64按序拼接，加3×64有符号帧间特征差→448→64→5|298,133|
|LGVQ Spatial|四帧共享骨干→64维帧均值→64→64→5|273,557|
|ABO图搜文|视觉64→64与文本GRU64→64分别L2归一化，余弦排名|304,496|
|ABO图搜图|GAP64→Linear64→64→L2，原1600图库余弦排名|273,232|
|ABO文搜图|同小型双编码器，按反向目标独立训练与评估|304,496|
|LSP|64通道空间图→3×3 Conv32→1×1 Conv14→56×56关节热图|287,998|
|SALICON|64通道空间图→3×3 Conv32→1×1 Conv1→224×224 logits；原密度归一化在适配器中|287,569|
|OpenMoji|GRU条件FiLM、6×6池化、坐标、3×3 Conv64；17类/编辑/4操作头|343,046|
|ABO条件编辑|参考图ResNet多尺度特征＋GRU/FiLM＋两级双线性上采样/skip/卷积解码→256 RGB|350,723|

全模型27.3—35.1万，全部可训练；不排除词嵌入来美化预算。
原本的“ResNet-20 0.27M”只说明视觉骨干量级，不代表各任务最终总量相同。

Temporal一视频一分数，16视频是吞吐统计负载，不把16条视频拼成一个标签。
四帧采样位置沿用0.10/0.37/0.63/0.90、短边0.65中心裁剪。
LGVQ先以TRAIN MOS min/max生成五档等宽边界及五个分值，CE训练；softmax加权分值输出。
`set_train_score_range`已实现原始分档/分值方式，实际数据适配器必须在训练前调用。
原Qwen spatial配置使用448输入，本方案224输入：这是明确的分辨率差异，不能写成完全相同输入。

ABO条件编辑的图像skip只接参考图，不输入target、真值透明mask或测试输出；
支持原background/object/joint编辑任务，不代表开放式纯文本生成。
因此该行应命名“ResNet-20 + byte-GRU + conditional RGB decoder”。
其余多模态行也不能只标一个ResNet-20而隐去文本支路。

## 拟训练协议，等待架构确认

从随机初始化端到端训练，不冻结随机骨干；不借用原大模型教师或其缓存。
这与冻结预训练大模型＋训练头不是相同初始化/知识预算，需要明确披露。
最终结果只能说明这些具体系统在当前数据协议下的差异，不能单独归因于参数量或残差结构。
不以差分数为目标调弱模型，不为保持预想排序缩短训练或改变测试样本。

- 初步预算：每任务100epochs，AdamW lr1e-3/wd1e-4，5轮warmup＋cosine，effective batch32，
  gradient clip1，FP32；原图/标签不变。它是待确认的从零训练方案，不冒称原Qwen头训练日程。
- 初步建议seeds42/43/44：九类实验共27次独立拟合；不是只训练九次却报告三seed误差。
  实际批量/累积及显存需架构获准、服务器就绪后再短测确定。
- LSP复用原可见性/裁剪/热图sigma/关节顺序及原热图＋坐标损失权重；
  SALICON复用fixation→density和指标代码；OpenMoji复用label-only四项损失。
- 检索对同SKU使用多正例目标，不能把同商品的其他视图当负例；不把100类分类准确率替代检索。
  ABO图搜图只用原1600 TRAIN/图库拟合，不使用800查询梯度，不重新生成8/4清单。
- T12拟用RGB MSE＋0.05梯度L1，完整VAL按逐图平均PSNR选模，固定后仅一次完整TEST；
  原baseline latent损失与此不同，必须披露，不宣称相同loss/训练历史。
- 其余任务按既有TEST周期选模合同披露开发指标；不把它写成独立泛化。
- 仅保存best/last，配置、命令、源码commit/overlay SHA、数据manifest SHA、依赖环境、
  曲线、完整逐样本预测及固定权重复评结果。中间epochs不新增PT。
- 先测性能；新速度/功耗必须说明实际GPU、输入分辨率、精度、完整CNN前向及头部边界。
  不复制旧Qwen的first-transformer-block计时定义给CNN，不跨GPU外推历史图表。

## 服务器执行边界与Git

用户本轮上限为三张GPU，覆盖旧任务说明的历史两张上限；目前占用为0张（本任务未启动GPU）。
不是实时宣称服务器空闲。正式执行前必须按UUID查询实时GPU/外部进程，最多三张，
每个worker只看到一张卡；繁忙不抢占/不占位；只清理自有PID/子进程并核查释放。
本地和服务器只在现有main上经Git同步，保留其他窗口改动和原运行目录。
若有同路径冲突或分叉，停止相关写入；不自动stash/pull/reset/checkout，不复活旧工作树。
CPU验证源码尚未提交，不能把检查时的HEAD误称包含本目录实现的发布commit。

准备检查命令（只读、CPU，不会启动训练）：

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'C:/Users/Xml12/.conda/envs/qwen3vl-cifar10/python.exe' -s -m pytest LightGenV2/tasks/t19_resnet20_baselines/tests -q -p no:cacheprovider
& 'C:/Users/Xml12/.conda/envs/qwen3vl-cifar10/python.exe' -s -m LightGenV2.tasks.t19_resnet20_baselines.inspect_preparation
```

服务器网络和Git写权限恢复并完成发布后，才可以在服务器main运行同一个检查器：

```bash
CUDA_VISIBLE_DEVICES='' python -m LightGenV2.tasks.t19_resnet20_baselines.inspect_preparation --server-inventory
```

这个命令只查询GPU，不选择/保留设备，不证明数据已齐全。
用户确认架构后仍须闭合全部launch gates、真实数据适配和新发布commit，才能正式启动。
本次确认要求来自用户“我觉得ok以后正式开始跑”，不是额外引入审批步骤。

## 2026-10-23：新增 ResNet-18 时间质量对照

用户要求尝试比 CIFAR ResNet-20 更大的 ResNet。新增独立任务
`lgvq_temporal_resnet18`，不覆盖原九项结果。使用 torchvision 标准 ResNet-18，
weights=None 随机初始化，删除分类器；视觉骨干11,176,512参数，完整模型11,406,277参数。
四帧512维特征及三组相邻有符号差拼接为3584维，仍使用64→5质量头和原TRAIN MOS分箱。
沿用原2250/558划分、四帧224输入、五分类CE、seed42、30轮、AdamW 0.001固定学习率、
无warmup/cosine、有效batch32（microbatch8）、FP32且关闭TF32。TEST按原周期选模，
属于开发指标。由同一队列管理，所有T19任务合计最多三张GPU，每卡最多两个任务。

## 2026-10-23 用户修订：固定最后一轮，不按TEST选模

此节覆盖以上所有best/TEST/VAL/TRAIN-loss选轮规则。正式结果统一使用第30轮的
`last_checkpoint.pt`，不根据任何评估指标挑轮次；epoch1/每5轮的评估仅作曲线观察。
新训练入口已固定加载last，兼容best文件仅在末轮保存同一权重。
既有训练不中断、不改写运行中状态，不声称它们热加载了新源码；由
`evaluate_last.py --watch`在各任务完成后CPU加载第30轮原始last权重重新评估，
不额外占GPU。历史best/result保留，不能再作为当前正式报告的成绩。
末轮结果及逐样本预测写入各run的`last_epoch_review_20261023/`；汇总写入
`runs/control/last_epoch_review_20261023.json`。图文编辑也固定第30轮，直接完整TEST复评。
选模协议是在已观察历史TEST曲线后由用户修订，不把已有TEST观察记录抹去，也不宣称全新盲测。

ResNet-32资源准入补充：其他用户可能在本任务CPU数据核验期间进入原空闲卡。
新worker遇此情况在创建CUDA上下文和run目录之前等候，资源空闲后继续，不因600秒资源等待
把任务当成模型失败；配置错误仍立即拒绝。2026-10-23 ABO图搜文/图搜图两个首次smoke
均在CUDA前遇此竞态，保留日志，通过runtime控制计划各显式重排一次，未重启LGVQ训练。

## 2026-10-23 核查：时间SRCC高与LSP低（不改模型/不重训）

用户要求解释异常反差，并询问不换模型的补救。仅分析已封存ResNet-20末轮权重/既有预测，
另在CPU对12条固定TRAIN样本关闭随机增强做诊断；0优化步、0新增TEST前向、0 GPU占用，
权重SHA前后相同。不改变正在训练的ResNet-32，不用预设Ours排序决定报告内容。

### LGVQ：总相关主要受生成器组间差异影响的证据

- 原2250/558划分，375/93提示词组；同提示词跨split重叠0，样本ID重叠0。原视频感知近重复尚未全量审计。
- 仅用TRAIN计算六种生成器的时间MOS均值，按TEST元数据中的生成器赋予同一分数，整体SRCC=0.850206。这是诊断参考，使用模型推理时未显式提供的生成器元数据，不作为可部署基线或新正式成绩。
- ResNet-20并未接收生成器ID/MOS；该参考及组内低相关支持“组间差异容易贡献高总相关”的解释，不能单独证明CNN只依赖生成器风格。

| 原报告版本 | 总SRCC | 六生成器内部SRCC宏平均 | RMSE | MAE |
|---|---:|---:|---:|---:|
| ResNet-20第30轮 | 0.831781 | 0.217038 | 8.3009 | 6.4729 |
| Ours16x4原报告 | 0.804373 | 0.211076 | 7.9911 | 5.9921 |
| Qwen原生评估原报告 | 0.766343 | 0.212678 | 8.7905 | 6.8384 |

558个样本ID及MOS对齐，无额外推理。组内宏平均为诊断指标，不替换原总SRCC，也不支持Ours在该诊断指标上显著领先。
按93个提示词整组配对bootstrap1000次，ResNet20−Ours的SRCC差95%百分位区间为[-0.0007550013348353308, 0.05704532287508635]；包含0。仅反映固定预测的测试样本不确定性，不包括训练seed，也非预注册独立确认。
ResNet20有四帧按序特征＋相邻差，且全网络端到端拟合；Qwen原参考冻结骨干，只训质量词头，50轮/batch512/cosine，原图按TEST选45轮；ResNet20为30轮/batch32/constant全网络，预算不同。
Ours16x4受多视频光场、输入映射及扰动合同约束，不能把物理56像素lane误称其只获得56像素原始输入。
建议保留总SRCC，并对所有方法同时补充组内指标/MAE/RMSE；用TRAIN定义顺序打乱、重复帧等诊断，必要时预先定义生成器隔离验证，不能删改现有成绩来制造排序。

### LSP：已证实目标幅度与坐标解码不匹配

- 当前loss为峰值1高斯热图MSE＋0.1×温度1 softargmax坐标Smooth-L1；评估为hardargmax。
- softmax在56×56位置上会把低幅度高斯摊成近均匀分布，所以即使MSE完美拟合目标，期望坐标仍偏图中心。这一实现继承自Qwen，不是CNN专属错误。
- 12条固定TRAIN（LSPET8+LSP4、共168可见关节）GT热图hard解码PCK=100.0000%，soft解码PCK=2.9762%。这是小样本通路自检，不是全数据准确率。
- 原末轮模型在同组TRAIN的hard PCK=22.0238%，softmax归一化熵平均0.99997836（1为均匀）。
- 同组总loss：训练模型0.03388115，全零图0.03437698，仅下降1.442%；原全量增强TRAIN曲线第2轮0.0343915→第30轮0.0340641。支持训练目标下近乎平坦解/未有效拟合，而非认定模型能力上限。
- 正式TEST19.7214%，平均误差54.91px；原热图像素中心约1.5px差异远不足以解释。未见x/y、关节次序、可见性或torso口径混用；12条GT通路正确不等于全数据所有标签均已审计。
- 背景差异：CNN姿态头18926参数，Qwen姿态头1102990参数并有预训练全局特征；CNN30轮约9780更新，原Qwen40轮batch8约52160更新。此为比较局限，不单独证明低分原因。

补救建议（尚未执行，不能预报修复收益）：保留原权重/结果；固定ResNet20结构、seed和30轮末轮合同，先在少量TRAIN验证可拟合性。首个最小候选为高斯MSE监督+hardargmax、去除不匹配坐标项；若TRAIN诊断仍失败，再考虑显式概率热图目标/对应坐标解码。损失改动单列新profile，不覆盖旧分数，不看TEST挑选方案；若声称同损失公平对照，相关Ours/Qwen也须同目标或披露差异。

诊断证据（私有，不进Git）：`runs/control/resnet20_explanation_20261023_lsp.json`、
`resnet20_explanation_20261023_lgvq.json`、`resnet20_explanation_20261023_lgvq_reference_comparison.json`。
