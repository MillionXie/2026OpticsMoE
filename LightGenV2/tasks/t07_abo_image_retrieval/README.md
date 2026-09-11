# T07 ABO 商品图搜图：独立光电工程

> **找当前最佳看这里**：独立固定权重复评77.50%（372/480），去光59.375%；目标≥389/480，尚未达成。
> 已固定保存权重：`runs/simulation/verify_strong_ep4_20260912_gpu1/best.pt`，不随训练覆盖。
> 该目录的`phase_masks.png`和`final_report.json`对应同一权重；训练曲线在下方来源run。
> 复现/指标口径看[复现入口](reports/reproduction/README.md)，执行命令看[COMMAND](COMMAND.md)。
> 下文75.21%等较旧数值是训练起点或历史对照，不是当前最佳；新排队方案没有成绩前不替代此指针。

> 来源：`runs/simulation/domain_distillation_20260912/domain_distill_strong/artifacts`，0.3蒸馏第4轮live。
> 训练仍在进行，但该候选已用独立进程在RTX4090、batch4上复核正常/去光及480-query/120-gallery。
> mAP@10=0.7266903；去光下降18.125个百分点；干净原训练99.8611%，仍有明显泛化差距。
> 训练源码`6ab405fe`，复评源码`5f3c704a`；权重SHA=`e5c0eaab4c84766b1ee231dd14271e97737604c0dcd675d9e2f4957c6932658d`。
> 该权重相位打乱/噪声复评尚待训练结束；此前76.25%完整结果作为历史对照保留。

## 持续目标与不可变约束（2026-09-12）

用户明确目标：原协议测试Hit@1≥0.81。480个query至少要389个命中，即81.0417%；
不能通过四舍五入、改变测试集/图库、移除难类别或混入训练商品来达标。
保持532nm、10cm、17μm逻辑像素、478²有效场、224²四专家/间隔30、Router Top2和V/L共六次捕获。
光相位允许正常梯度训练，但不修改光路/ROI/排布/传播模型来换成绩；alpha始终严格>0.4。
推理不允许添加Transformer/attention大网络。允许训练期冻结教师或蒸馏，教师绝不进入学生部署。
原数据与baseline保留；每轮记录改动、train/test曲线、相位变化、专家选择、同权重去光结果。
选模仍按用户指定的周期test口径，属于test-selected；不能称作未被查看的独立测试。

下一训练手段：`domain_distill_light/strong`。独立子进程只给1440原训练图+4116外部训练图
生成冻结Qwen 2048D缓存（不前向val/test，5556张）；学生随后仅读取缓存，不加载完整Qwen。
比较教师和学生对“其他训练商品中心”的相似度分布，用KL蒸馏，不强行匹配2048D/64D坐标。
剔除自身商品，教师Top1类别错误的训练query关闭蒸馏；正确query按正例概率高于随机水平的幅度加权，
减少错误教师与GT损失冲突。GT检索/分类监督仍保留，训练图像/光学噪声不变。
权重0.1/0.3两个对照，前三轮热身；不改变输入头、电子残差、读出头或optics.py。
缓存和首轮反向验证有效后，补充`domain_distill_stronger`（权重0.6），其余配置逐项相同。
它仍从同一75.2083%权重开始，用已释放的第三张卡测试更强的训练约束；不加载教师网络推理。
不能把0.6解释为光融合alpha：这是训练KL损失的权重，alpha边界依然是[0.4001,0.8]。
缓存校验原manifest、外部pool、全部训练sample ID顺序及图像字节SHA，防止旧缓存串用。
蒸馏模块、teacher权重/缓存都不是推理依赖；最终报告显式记录`teacher_at_inference=false`。
入口与等待/释放GPU的方法见COMMAND第14节；不同时超出三张GPU，也不挤占他人工作。

蒸馏代码已在`6ab405fe`提交并同步GitHub，本地/服务器59项CPU测试通过；正式GPU训练尚待前序完成。
实际排队记录（GPU1 RTX4090，等待时不占CUDA）：

- `runs/smoke/domain_distillation_20260912`，监督PID2208844：等待旧池续训完成，生成训练教师缓存，再跑1轮蒸馏检查。
- `runs/simulation/domain_distillation_20260912`，监督PID2208989：等待上述检查成功，依次运行light/strong各24轮。

两个正式候选都从同一个已完成75.2083%best独立开始；不是strong接着light继续训。
源码锁定独立worktree；任一依赖失败就停止，不自动抢占或终止别人的GPU任务。
扩大池组已完成24轮及同权重复评：第16轮EMA为76.25%（366/480，RTX3090），
去光60.625%（下降15.625个百分点），相位打乱47.9167%，轻微CCD噪声76.0417%。
干净原训练Hit@1=100%，仍有明显新商品泛化差距；尚未达到389/480目标。
当前完整复评最佳：`runs/simulation/domain_refine_wide_20260912_gpu2/domain_refine_wide/artifacts/best.pt`，
SHA256=`6f23466a1570a024e5bcf8408ae70a01065399e28e818af5a394dff15bfa850a`。
GPU2训练子进程2185591已退出，队列确认CUDA释放。暂不覆盖旧实验交付包。
旧池续训对照已完成24轮并释放子进程2184980；最终best第16轮live为75.8333%，
去光60.8333%（下降15个百分点），相位打乱54.1667%，干净原训练99.9306%。
权重`runs/simulation/domain_refine_control_20260912/domain_refine_control/artifacts/best.pt`，
SHA256=`86c020ab057447bf1825c226f64117b24909f7a2c5957a61ab508327972f35d8`。
教师缓存已完成并释放进程2226915，5556张训练图/2178商品、22.26MiB、0可训练教师参数；
缓存SHA256=`aa5a5a952c0a96836b4b035d7905ca600f9082dc0bd26b5396cdab4cea36f2db`。
CPU只读审计：排除自身商品后的扩充训练图库Top1类别正确92.0626%，非零蒸馏门控91.8287%；
原训练/外部训练的教师正确率分别94.3750%/91.2536%，这是训练监督可靠度，不是新的测试分数。
蒸馏1轮检查子进程2231131已完成125步及最终复评并释放：KL=2.17415、批内教师正确率92.8%、
平均置信权重0.2891；best仍是75.2083%起点，因此检查只证明流程正常，不算性能提升。
正式light已启动：GPU1、子进程2236343、源码6ab405fe，随后同卡独立跑0.3 strong。
0.6 stronger已启动：`runs/simulation/domain_distillation_stronger_20260912_gpu2`、GPU2、
监督PID2239228/子PID2239231、源码8928c863（本地/服务器59测试通过）。初次shell分页器吞字导致的
启动尝试没有产生训练子进程，已确认退出；上面记录的是重新启动后核实存活的正确进程。
三个蒸馏强度除KL权重外配置相同，均从75.2083%起点独立训练；不继承彼此last。
下方75.2083%仍是所有后续实验共同起点，正式蒸馏结果尚未完成。

续训初始化对照`domain_distill_resumeaux`：代码核查发现，以前虽然checkpoint保存了训练辅助头，
每次续训仍新建随机分类proxy与光学辅助分类头。新增对照只恢复同一75.2083% live起点保存的辅助头；
与0.3 strong的其余配置完全一致，仍重新初始化优化器。不声称重置就是泛化差距的已证实原因。
仅接受已审计的同标签起点SHA，拒绝EMA起点（其辅助头未做EMA）及缺键/尺寸错误/非有限权重。
学生state_dict/光路/前端/推理head不变，辅助分类头依然不进入推理；execution记录初始化方式。
先在GPU4现有跨视角训练完成后做1轮检查，再独立从原起点做24轮对照，命令见COMMAND第15节。

教师温度校准对照`domain_distill_sharpteacher`（COMMAND第16节）：原训练1440图、seed42抽取20000对，
排除同商品后，同类/异类平均余弦：学生64D为0.79326/-0.01120，教师2048D为0.73182/0.49562。
相同温度0.1并不对应相同概率锐度；教师类间余弦间隔约为学生的0.294倍。
因此本对照仍用学生温度0.1，教师改为0.03（训练间隔比例估算后取整），KL系数0.3；
不是根据test试出温度，不改物理直流分量，也不改推理Router的softmax温度。
这是监督尺度不匹配的可检验假设，不宣称已解释全部差距；光路、推理和其他训练设置与0.3 strong相同。
默认教师温度仍等于学生温度，单元测试验证旧配置逐值一致；新参数只用于这个明确命名的对照。

本轮跨视角组也已完成24轮：best第8轮live为75.8333%，去光61.0417%，相位打乱53.75%，
干净训练99.8611%；末轮live只有72.5%，未超过扩大池组，不作为当前最终候选。
best SHA256=`0407822b5a5a7120c7ee3ef6fc8af82cf51a85584d3cf18a5c554080da472393e`，
位于`runs/simulation/domain_refine_views_20260912/domain_refine_views/artifacts`；进程2192031已释放CUDA。
GPU4恢复辅助头检查已接续启动，监督2246410/子进程2261324；正式队列2246561等待其成功。
GPU2教师温度检查队列2261218等待0.6组结束，正式队列2261413等待该检查成功；等待不占显存。
温度对照源码0c8998cb、本地/服务器61项测试通过；此处PID为启动审计，实时状态以各run/status.json和实际进程为准。

额外数据路线：`runs/simulation/domain_pool500_20260912`已在CPU生成，3390外部商品/6780图，
含旧2058商品全部、新增1332商品；与原目标商品/图片ID交集为零，排除20个精确或近重复候选商品。
manifest SHA256=`6811b561dda2df27f274c9551bab83ce9149cccf2de6102ad429bd6078113e51`。
原图不复制，原train/val/test和120商品图库不改；合并训练为3510商品/8220图。去重仍是保守启发式，
不宣称排除了所有跨商品语义近似。纯冻结Qwen的评估输入/图库未改变，原95.2083%基线仍适用。
`domain_refine_pool500_mix13`从已完整复评76.25%起点继续，原/外部商品采样改为每类1:3，
每batch仍40图、类别均衡；每轮167主batch可覆盖全部商品，24轮，无教师/无跨视角额外前向。
这是更大池+采样比例的组合优化，预算与128步对照不等，不做单因素贡献宣称。光路、alpha>0.4、推理不变。
命令见COMMAND第17节；仅在GPU4恢复辅助头实验成功结束后接续，先检查，再正式训练。

cap500数据/采样源码`f9c09069`已同步GitHub，本地/服务器62项测试通过；真实数据CPU预检确认
167个batch覆盖3510商品，逐类1原+3外部比例成立。等待队列为检查PID2274165、正式PID2274380，
源代码锁定worktree，等待不占GPU。恢复辅助头检查已完成75.625%、去光59.1667%，进程2261324释放；
正式组子进程2266611已启动。以上只记录启动身份，尚不是新正式最佳。

续训辅助头口径更正：代码复查发现`domain_distill_resumeaux`虽先恢复全部辅助参数，epoch0之后
又把640参数的类别proxy覆盖为原训练类别均值；实际保留恢复的是光学辅助分类头，而非整个辅助头。
先前“全部恢复”的表述不准确；在跑的1e19af4a组按此旧语义解释，不改其运行代码或历史结果。
新增`domain_distill_resumeaux_full`明确跳过这次覆盖，恢复同一已审核live起点的全部训练辅助参数。
旧profile保留原数值行为；新增`execution.category_proxy_initialization`区分初始化方式。
这不更改学生推理state_dict/电子残差/光学参数化；是否改善必须由新run实测，不能冒用旧组成绩。

### 小型电子感受野对照（已排队，尚无成绩）

`domain_refine_context7`：已有两个Vision电子残差的depthwise卷积由3×3改为7×7，
通道仍192、层数不变；仅增加15,360个参数。Language卷积仍5、最终64维读出头不变。
两个Vision卷积的局部路径感受野由5×5扩到13×13 token（图像网格14×14）；这不是新光学孔径。
不新增分支、attention、Transformer或教师；**optics.py、前端、六次捕获、相位尺寸和光Router Top2均不改**。
用已验证76.25%权重初始化：原3×3权重放到7×7中心，其余全零；光学及其余张量逐值原样继承。
因此起点函数在数值容差内保持，随后新增外围卷积权重可接收梯度；不能用改变光仿真换取分数。
训练沿用cap250池、1:1采样、24×128步、相同损失/学习率/噪声；不与cap500同时改变数据。
当前76.25%逐类chair58.33%、wall art45.83%、light fixture66.67%、mirror56.25%，
这为完整形状上下文提供了待验证动机，不是已经证明的失败原因。类别/样本不移除。
metadata和audit记录`electronic_context_kernels={vision:7,language:5}`；独立部署据此构造正确层形状。
最终必须复评正常、去光、相位打乱；不把更强电支路的收益称作纯光的收益。命令见COMMAND第18节。

源码`2ca45836`已同步GitHub，本地/服务器66测试通过。使用真实4张训练图，旧源码模型与新模型
CPU完整前向最大绝对差为0，光学张量逐值一致，新增参数精确15,360；这不是GPU全量精度复评。
GPU1检查队列PID2282318、正式队列2282606分别等待既有0.1/0.3组及新检查，等待不占CUDA。
完整辅助头恢复修正代码`a68a76cd`另有实际checkpoint的9,876参数保持检查；full profile尚未训练，
不能引用正在跑的旧partial恢复组结果作为full结果。

轻蒸馏0.1已完成24轮：`runs/simulation/domain_distillation_20260912/domain_distill_light/artifacts`，
best第4轮live，Hit@1=75.8333%、mAP@10=0.7108571、去光59.375%、相位打乱52.5%、轻噪声75.8333%。
干净原训练99.7917%；末轮EMA71.875%/live71.4583%，不使用last作为最佳。不超过76.25%现有候选。
best SHA256=`4fef8e0f4b839016a1701a6f14fb73c1b691930189d7070011f5f91a2666d8e1`；
子进程2236343退出并确认释放，0.3组子进程2284756接续，同样从75.2083%独立开始。

### 训练图库类别数量校正（只改loss）

`domain_refine_balanced`从76.25%best续训，仍用cap250池与1:1采样，不使用teacher或7×7电子变体。
CPU清单审计：训练图库chair/rug/sofa/wall art/light fixture/stool/pillow各262商品，bed198、mirror101、vase45；
固定测试图库各类仍12商品。旧gallery NLL累加同类商品的exp(similarity/T)，会包含训练图库数量先验。
例如所有相似度相等且剔除自身商品时，chair NLL=2.1212而vase=3.9015，不能把这差异全部当作视觉难度。
新loss先按每个query剔除自身商品，再将各类exp-score和除以该类有效商品数，
因此上述均匀相似度例子各类都是log(10)。原始余弦排名、margin损失、Top1记录和测试推理完全不改。
这只是可检验的目标尺度修正，不证明类别不均就是全部泛化差距；不删除商品、不缩小测试图库。
旧profile默认关闭、逐值兼容；单测检查候选复制不变性、剔除自身后的计数和有限梯度。
精确开关`execution.common_config.gallery_class_balance=true`，命令见COMMAND第19节。

loss校正源码`18ec64dd`已同步GitHub，本地/服务器68测试通过；GPU2检查/正式队列PID2290430/2290665
等待教师温度组及检查成功，不占额外显卡。尚无性能结论，不替代既有权重。
0.6组已完成：best第4轮EMA 75.2083%，去光59.375%、相位打乱53.3333%、轻噪声74.7917%，
干净训练99.9306%；best SHA256=`753eefaeb40515169178b0bdb0e29f9aa2cf96249e3d8713e5733ca1a35fff60`。
该组子进程2239231已释放；教师温度检查2288089也完成并释放（75.2083%、去光59.5833%），
温度正式组子进程2293304已接续。轻/强蒸馏不能按系数越大越好作结论。

0.3组第4轮live中间候选77.50%，mAP@10=0.72669031，原训练排除自身商品99.8611%；
source6ab405fe、RTX4090、原480-query/120-gallery。当前best.pt读取前后SHA一致：
`e5c0eaab4c84766b1ee231dd14271e97737604c0dcd675d9e2f4957c6932658d`。
这是运行中快照身份，后续若更高best覆盖应以最终报告为准，不额外复制周期PT。
四个alpha=0.4369612/0.4394684/0.4290614/0.4281045；专家/全局相位相对75.21%起点圆周RMS
变化0.02662～0.03188rad，V/L Router为0.004789/0.001808rad，12组均更新。
这是既有3×3电子架构，不含context7；只增加训练蒸馏。去光及完整固定权重复评未完成，尚未达到81%。

独立评估入口已增加checkpoint+SHA双参数（COMMAND第20节），防止默认读到旧assets/best。
原assets及其清单不覆盖；新命令核对权重读取前后SHA，并在报告写入真正评估的权重身份。
本轮拟对77.50%候选独立重新从原图评估正常/去光，尚不能用训练history代替这一步。

## 共同起点75.2083%及已完成的续训对照（历史记录）

三组目标相关扩充已完成40轮，CUDA子进程均退出。混合训练best为第15轮live，
**Hit@1=75.2083%（361/480），mAP@10=0.71290385，去光60.2083%（下降15个百分点）**。
光Router Top2、四个alpha为0.43703/0.43953/0.42904/0.42813，没有靠降低光融合系数取得此成绩。
专家/全局相位相对69.79%起点的圆周RMS变化约0.073～0.113 rad；Router约0.0034～0.0069 rad。
干净原训练商品检索99.7222%，过拟合仍在；40轮live测试73.9583%，用best而不是last。
同预算原数据对照70.4167%，先外部适配再混合72.5%。均为单seed、test-selected，不是独立无偏估计。

最佳权重：`runs/simulation/domain_mixed_20260911/domain_mixed/artifacts/best.pt`，SHA256
`0e523f3a08631e0d248032c58534761758841e23702e155df8f2d3f4cd472f85`。
代码`09835251`，该目录包含逐类/逐样本指标、学习曲线和相位图；本次不覆盖旧交付包。

下一轮从上述75.2083%固定起点比较三组，配置为`standalone/domain_refinement.json`，命令见COMMAND第13节：

- `domain_refine_control`：922外部商品池继续混合训练，作为重新启动优化器的对照。
- `domain_refine_wide`：每类最多250商品的新池，实际2058商品/4116图，类别不足250的不强行补齐。
- `domain_refine_views`：与wide相同池/预算，再对同一商品的两张不同图计算归一化检索向量的cosine一致性loss；
  权重前三轮从0.05升至0.15，两个视角都回传，不使用教师，也不把同类不同商品冒充同一商品。

每组24轮、每轮128个主batch（40图，原/外部1:1），每4轮同口径测试live/EMA，保留best/last。
views额外做一次训练前向/反向，训练FLOPs不是严格等预算；**推理仍单张图片、同一模型、六次光传播**，
无额外视角、TTA、网络分支、attention或完整Qwen。光学几何、DC/噪声、alpha>0.4保持不变。
新池`runs/simulation/domain_pool250_20260912`，SHA256
`5348ffec1ba51a547ee1be81341e1c4bbbf1b970b9aeec47e557ddd7888f8488`；所有原200商品仍被排除。
不修改原120/40/40划分、480测试query、120商品评估图库、类别或标签；只是扩大梯度训练数据。
冻结Qwen已在`frozen_qwen_20260912`从1920张train/test图重新前向核验（原尺寸与square分别处理），
主baseline仍95.2083%，0训练参数、无微调，source09835251，GPU0 RTX4090且已释放。
新池包含全部旧922商品；新优化分数以各run实际final_report为准。

本轮训练代码`ac737d41`，GitHub分支`experiment/t07-domain-refinement-20260912`，本地/服务器55项测试通过。
正式队列位置均在`runs/simulation/`（下表是启动身份，不是完成声明）：

| 方案 | Run ID | GPU | 监督PID |
| --- | --- | --- | --- |
| 旧池续训对照 | `domain_refine_control_20260912` | 1 / RTX4090 | 2184972 |
| 扩大商品池 | `domain_refine_wide_20260912_gpu2` | 2 / RTX3090 | 2185586 |
| 扩大池+跨视角约束 | `domain_refine_views_20260912` | 4 / RTX4090 | 2186098 |

views等待`runs/smoke/domain_refinement_20260912_gpu4`成功后自动启动，等待不占CUDA。
原不带gpu后缀的wide队列及原smoke因0/3号卡被他人占用，被准入检查拒绝，**没有启动训练子进程**；
这些小型status仅留作资源审计，找成绩用上表。不同GPU不比较训练耗时，所有结果记录实际硬件。
每组最终报告正常/去光/打乱相位/轻微CCD噪声；结束或失败释放自己的GPU进程，不终止他人任务。

## 2026-09-11 目标相关数据扩充试验

新增 `domain_target_control / domain_mixed / domain_curriculum`，源码和命令见 COMMAND 第12节。
三组从同一高alpha 69.7917% best续训，40epoch、每轮至少64个40图batch；不是从头预训练。
新增池只按原十类精确product_type映射，排除原200商品、共享图片ID、文件SHA及近重复；
候选需本地图片可用。此为元数据筛查，不冒充人工验证全部标签；原标签/图片不修改。
实际新增922商品、1844图：前八类各100商品，home mirror 89、vase 33，每商品2图。
池位于`runs/simulation/domain_pool_20260911/`；manifest SHA256为
`b27487006630e95ef375e702a6f828ccd897b6b2dbea1f8250244f5da88efac8c`。
与目标商品/image ID交集均0；6个候选商品因目标图像精确/近重复被排除。
原120训练商品、40未用val商品、40测试商品划分不变，**评估图库固定原120商品**。
混合每类2原商品+2新增商品；curriculum前10epoch每类4新增商品，后30epoch混合；
target_control只用原商品。每轮循环覆盖所有活跃域的商品，记录实际商品/图片覆盖。
图库排序训练使用该阶段活跃训练商品中心，推理始终只用原图库；不把外部商品放进test gallery。
三组前端、网络、损失、噪声/增强、光Router Top2、alpha>0.4保持一致，只改变训练数据和顺序。
无新教师、无TF/attention、无新电子分支；弱亮度/对比度增强，不裁掉物体。
保留20%–30% DC等原训练噪声（四分之一batch），本轮不增加独立相位dropout。
新run只留best/last，初始权重可作为best保底，报告epoch=-1不算新训练提升。
用户本次明确允许最多三张GPU：可各组一张，结束/失败退出子进程，记录CUDA进程释放检查。
原69.79%及已交付包不覆盖，新结果需完整正常/去光复评后再决定采用。

服务器已提交三组40epoch任务（源码`09835251`），run均位于本任务`runs/simulation/`：

| Run ID | 训练数据安排 | GPU索引 | 启动监督进程PID |
| --- | --- | --- | --- |
| `domain_target_control_20260911` | 原120训练商品，40轮 | 1 | 1535413 |
| `domain_mixed_20260911` | 原120+新增922商品，混合40轮 | 3 | 1535414 |
| `domain_curriculum_20260911` | 新增922商品10轮，混合30轮 | 0 | 1535415 |

以上是启动记录，不是完成声明。以各run的`status.json`、`<profile>/artifacts/history.json`和
`final_report.json`为准；curriculum队列须等`runs/smoke/domain_20260911/status.json`完成后启动，
等待时不占GPU。每个队列结束会记录`gpu_context_released`；不要用PID历史记录判断当前占卡。
学习曲线、相位图、best/last与最终去光评估都留在对应artifacts，不新增散落的结果目录。

当前入口为 `python run.py`，只使用本文件夹的 `standalone/`，**不导入T01或旧experiments，也不加载完整Qwen模型**。
日常步骤看 [COMMAND.md](COMMAND.md)；维护、导出与历史数值证据看 [复现入口](reports/reproduction/README.md)（源码仓库内）。

## 给老师检查的结构

以下结构说明原70.2083%独立交付版本。当前数据扩充试验使用同一计算图，但输入已改为
保全物体的`contain_white`，融合alpha硬范围为[0.4001,0.8]，起点实际约0.429～0.440；
不能把下方旧包的低alpha与当前试验混为一谈。试验还关闭教师loss，辅助类别代理只用于训练。

- 固定图片224×224 + 固定英文检索指令（见standalone/data.py）；无类别/商品标题输入。
- 冻结Qwen patch Conv3d、视觉位置表、主merger，以及固定prompt用到的token embedding行。
  原始tokenizer和processor保留；未知prompt/token会报错，不能冒充支持任意语言输入。
- V：196×1024 → Linear/LN到192 → 两级光电融合 → 投影回1024并加输入跳连 → merger得到49×2048。
- 图像特征填入文本模板的image token位置；L：S×2048 → 192 → 两级光电融合。
- 每个模态：光Router一次CCD选Top-2；四个专家中的两个一起传播/CCD；融合后电子重编码、再加载全局相位/CCD。
  V/L共六次捕获，不是三层专家再额外加Router。Router能量标准化、softmax、Top-2仍为电子运算。
- 电子残差：V两组3×3 depthwise二维卷积，L两组核长5因果一维卷积；每组含192→384→192通道MLP。
  无attention、Transformer、VGG或额外图像分支。当前未采用失败试验中的扩核/增强读出头。
- 同尺度融合：En=E/rms(E)，On=O/rms(O)，M=(1-alpha)En+alpha On，F=rms(E)M/rms(M)。
  尺度统计停止梯度，alpha范围[0.01,0.95]，best约0.087～0.104；不是性能贡献比例。
- 最后有效token mean/max拼成384维 → LN → Linear64 → L2。头仅25,408个参数。
- CCD明确保留 `frame mean → clip12 → log1p → pool224 → row LayerNorm → ReLU → Linear192`；
  **不是纯线性CCD读出**，不得在部署时擅自去掉并继续引用当前指标。
- 专家224×224，2×2排列、间隔30，有效场478×478，FFT画布518×518；532nm、10cm、逻辑17um。
  8um相位SLM需物理映射，不可直接把224逻辑像素当硬件像素。相位为2πsigmoid(raw)，当前从best续训，不是全零初始化。
- 独立模型保留原相位、融合及电子权重。只删除无输出作用的L隐藏状态重构投影、原始TF模块、LM头、
  未用的词表行、旧优化器和历史候选。输入前端约2915万冻结参数，可训练约278万。

## 当前结果及边界

固定best源SHA256：`3674981c3499555077c7eadc3072a11e07583675000ec4c012616a96185867f5`（epoch10）。
旧版A100为70.0000%；旧版4090为70.2083%。独立版最终隔离目录4090完整复评为70.2083%、去光67.2917%，
测试embedding对旧版平均余弦0.99999756；存在微小混合精度差，不宣称逐位相同。最终实跑结果见包内reference和[验收记录](ACCEPTANCE.md)。
独立assets约85.6MB，包含processor、模型、仅训练样本的教师目标；无完整2B模型。

ABO similarity10：10类200商品、每商品12视角；train/val/test按商品120/40/40，
即1440/480/480图片，val不使用。480测试图查120训练商品中心，**同类别算相关**，不是同商品实例检索。
每个query有12个正候选，报告Hit@1（旧称R@1）、mAP等，不混用positive Recall。
先逐视角L2归一化，再按商品均值/L2；不按真实类别预筛选候选。

“新商品泛化”指目标数据train/test商品ID互斥、类别相同（不是未见类别，也不保证Qwen预训练从未见过该商品）。
测试商品不参与梯度训练，但test定期参与选checkpoint，所以不能称为从未查看的独立测试。
当前是**类别相关的图搜图**：同类别不同商品算相关，不是必须找回同款SKU。标题不进入输入、候选向量或排序；
仅用于人工核查manifest类别。例如B075Z8THYY的dresser标题与bed标签需确认类别定义，未改标签/删除样本。

训练包括监督对比、训练教师向量/类别中心、Router均衡、相位DC与工作点约束；
训练中20%～30%相干未调制分量、截断偏置高斯CCD噪声、相位旁路；位移=0、k空间约束关闭。
评估是确定性理想仿真，**不是实际光路或固定20%零级光测试**。
EMA、每5epoch test选best，仅best/last；成绩为test-selected，不能声称独立无偏测试。
冻结完整Qwen baseline历史95.2083%，当前仍明显落后。完整baseline需另行加载大模型，
不混入此精简学生运行入口。

## 文件怎么找

| 文件 | 内容 |
| --- | --- |
| run.py | 独立命令入口 |
| standalone/frontend.py | 冻结Qwen必要前端，无模型/Transformer类 |
| standalone/model.py | 电子残差、融合、图文组装、检索头 |
| standalone/optics.py | Router、相位、传播、CCD及实测注入接口 |
| standalone/data.py | 固定数据协议、图库和指标 |
| standalone/cli.py | 评估、去光、独立微调、报告和相位图 |
| standalone/export.py | 一次性CPU选取原权重；交付接收人不需要执行 |
| assets/ | 运行资源（在交付包内；不进入Git） |
| reference/ | 交付版本实跑报告（在交付包内） |

源码中的 `legacy_run.py`、旧configs/refinement/retrieval_contract只保留历史审计，不进入独立ZIP。
旧后端被其他任务共用，不做全仓删除。历史证据仍在Git和runs；本次不删除数据/旧权重。
`build_lab_package.py`白名单打包当前代码、assets、ABO数据和reference；名字沿用公共规则，
**此次是仿真/微调包，不是包含SLM/CCD厂商SDK的实验室控制包**。实测注入接口不能冒充已验收的自动采集流程。

GPU默认一张，同一助手最多两张；退出后确认自己的PID已释放，不杀他人进程。

## 当前训练优化试验

`--profile teacher_curriculum` 保持上述推理图不变：训练集教师特征预热→光电联合→无蒸馏收尾，
batch40、跨商品正样本、关系蒸馏递减、EMA、四个alpha固定为当前best数值。
这是任务内预热，不是额外外部数据集预训练；不宣称已获得提升。详情及命令见COMMAND第5节。
正式独立包与70.2083%证据不覆盖；新候选通过全量正常/去光评估后再考虑替换。

上述小集curriculum已完成30epoch，未提升（新epoch最高70.00%、末轮67.9167%），保留原best。
下一方案是`standalone.broad_transfer`：从较大原始ABO中限量选商品类型，在与目标商品/图片做去重的
预训练池上联合训练相位与电子，再迁移当前10类。训练辅助类别头不进入推理，alpha固定、无完整Qwen/教师。
代码/参数及完整命令见COMMAND第6节；目标75%，尚未获得新性能结论。
预训练池已审核并选出128类型、6144商品、12288图；正式run为`runs/simulation/broad_transfer_20260910/`，
阶段结果分别在`artifacts/pretrain`和`artifacts/adapt`。详细数据排除/训练身份见复现入口；原正式包不覆盖。

该轮已完成：预训练后目标Hit@1为62.0833%，微调新epoch最高67.50%（10/15轮），末轮67.2917%；
最终选择epoch -1，即原70.2083%保底，不代表预训练带来了提升。

### 严格高alpha候选（尚非正式70.21%权重）

`standalone.broad_transfer --mode adapt --profile high_alpha` 使用原最好相位/电子参数热启动，
四个融合系数改为`0.4001+0.3999*sigmoid(raw)`，初始0.45；浮点饱和也不会低于或等于0.4。
只在高alpha候选中选优，绝不拿旧低alpha成绩当本版本结果。计划60epoch×64steps，训练batch40。
前5epoch冻结主要电子残差与输入/输出adapter，训练相位、Router、光学编解码、融合系数及读出；
随后联合训练，分组学习率+余弦衰减+EMA。相位峰值LR0.004，Router0.0005，无教师loss。
轻量裁剪、翻转、小角度旋转、亮度/对比度、低概率轻模糊；25% batch加入轻量CCD噪声及20%～30%DC。
V/L融合前光特征另加训练用分类监督，其辅助头不进入推理、不替代最终检索。
每5epoch全量test选EMA best（明确test-selected），仅best.pt/last.pt。
最终输出同权重去光、相位像素打乱、单种轻量CCD噪声测试、相位更新量和Router分布。
推理仍为原六次光捕获、Top2、无TF/attention；Vision外层输入跳连仍在，alpha不是全网能量/性能贡献比例。
固定ROI、正常精度相位、k滤波/像素位移关闭，CCD解码仍含已有log1p等非线性（未新增）。
命令见COMMAND第7节；超过75%是优化目标，不是已测成绩。旧ZIP不会自动被替换。

该60epoch试验已完成（source `3eb20ae2`）：best为55轮，Hit@1 **67.9167%**，同权重去光64.1667%
（下降3.75个百分点），相位像素打乱52.7083%，单种轻量CCD噪声67.0833%。alpha约0.432～0.441。
证据：`runs/simulation/high_alpha_aug_20260910/artifacts/final_report.json`，不是低alpha的70.21%版本。

### 高alpha检索对齐续训

`--profile high_alpha_retrieval` 仅从严格高alpha的best续训，推理图/参数量/几何完全不变。
训练图库由1440张**训练图**生成120个商品中心，每epoch刷新；每个query排除其自身商品的全部视图，
同类别其他商品作为正候选，其他类别为负候选。损失为多正例检索NLL+最难负商品margin，
降低辅助分类CE/光特征分类权重；保留轻量跨商品监督对比、Router均衡及原高alpha噪声。
训练图库是无梯度的缓存，不加入推理、不使用test/val图，不按真实测试标签筛候选。
先25epoch光电联合调整，最后5epoch仅训练已有读出头（光学与电子残差冻结）；减轻裁剪/旋转等增强。
每5epoch分别评估live和EMA，按同一test指标选best；保留高alpha起点，明确test-selected。
仅best.pt/last.pt；训练日志另记排除自身商品后的训练图库Hit@1，以区别辅助分类正确率。
参数覆盖集中在`standalone/retrieval_training.json`，基础合同继承`high_alpha.json`。见COMMAND第8节。
这是针对训练目标与实际检索不一致的尝试；不声称已解决所有语义表征差距，未达到75%前不作为正式提升。

该30epoch续训已经完成（source `e59fc45f`），选择epoch15 EMA：Hit@1 **68.9583%**，同权重去光
64.5833%（下降4.375个百分点），相位打乱55.2083%，轻微CCD噪声67.7083%。四个alpha约0.430～0.440。
训练PID2135641已退出，所用GPU已释放。仍未达到75%，不将低alpha70.21%混入此候选。

组会数据分析及老师审阅用独立代码位于仓库根 `LightGenPublic/tasks/t07_abo_image_retrieval/`。
该目录是独立仿真/续训审阅快照，不覆盖本目录的优化代码或旧实验室包。清理没有删除历史run/数据；
数据审计脚本为本任务 `analysis/audit_for_meeting.py`，只读CPU分析，保持原始测试口径。

### 完整输入 / SAM / 全场语言读出（2026-09-11）

在本目录继续优化，**不覆盖已交付LightGenPublic审阅包**。三组都从同一68.9583%高alpha权重开始：
`preserve_adam`：等比缩放、白色补边到224²，关闭裁剪和旋转增强，AdamW适配；
`preserve_sam`：同上加SAM，rho=0.03、前三轮线性热身；
`preserve_fullfield_sam`：在SAM基础上把L端expert/global CCD从“pool224后截前77行”改为整场pool到77×224。
V端读出不改，输入图像/光学ROI/专家大小/17um/10cm均不改，无新增网络参数、TF、attention或教师。
相位/Router/电子一起更新，alpha仍严格>0.4，原噪声、20%～30%未调制分量和专家均衡保留。

SAM是Sharpness-Aware Minimization，不是Segment Anything：两次反传共用同批图与相同随机噪声/dropout，
先暂时扰动活动参数，再精确恢复原值，用第二次梯度做AdamW更新；异常时也恢复，冻结参数不扰动。
参考[原论文](https://arxiv.org/abs/2010.01412)。这增加训练计算，不增加推理网络；不保证涨分。

三组使用相同的30epoch×64steps、batch40与适配学习率，末5轮仅读出，EMA/live每5轮按test选优。
第一组相较旧68.96%同时包含输入修改和新适配训练，不能把差值全归给裁切；SAM两组对照的配置更严格匹配。
新输入/读出模式写入checkpoint metadata，图库构建、训练、评估统一读取；旧权重默认仍为原中心裁切/前行读出。
选择只在各自新合同内进行，不能拿旧68.96%不同预处理的结果充作保底。仅best/last。
`generalization_queue` 单卡串行、每组一个独立进程，失败停止；状态/指标汇总到queue的status.json，完整命令见COMMAND第9节。
本轮已启动，源码 `c1f4b471`；结果统一在 `runs/simulation/generalization_20260911/`，
实时状态看该目录的 `status.json`。顺序为SAM、AdamW对照、SAM+全场读出；启动时队列PID3329183，
第一组PID3329218，只使用GPU1（UUID见COMMAND）。不能仅凭历史PID判断仍在运行，应检查状态/进程。
本地和服务器32项测试通过，`runs/smoke/generalization_20260911` 的两步CUDA训练及完整复评已结束，
12片相位均有非零更新；这不是正式性能结果。未完成正式测试前不声称已超过0.7。
新权重必须用本版T07加载，不直接放入旧审阅ZIP。

### V/L同时完整CCD读出对照

`preserve_fullfield_both_sam` 在上一组仅L全场的配置上，**只把V也改为全场读出**。
V：整幅478²强度经原强度处理后pool到196×224 → 原rowLN/ReLU/Linear → 196×192；
L：整幅478²同样pool到77×224 → 原读出 → 77×192。不再丢弃pool后的底部行，
但平均池化/归一化本身仍有信息损失，不宣称保留全部光能或所有细节。
expert/global共四处CCD读出统一生效，光Router四探测区读出不变；参数量、相位几何、alpha下限不变。
与仅L全场组同源权重、SAM、完整商品输入、学习率、采样与选模，便于隔离V端修正的作用。
命令见COMMAND第10节，结果为`runs/simulation/generalization_fullfield_both_20260911`。
此补充组等待原generalization队列成功结束后在同一GPU运行；等待不创建CUDA上下文，不改正在训练的源码。

### 当前完成成绩与防过拟合续训

2026-09-11：完整输入AdamW组 `generalization_20260911/preserve_adam/artifacts` 已完成30轮，
source c1f4b471，epoch25 live，Hit@1=69.7917%（335/480）、mAP@10=0.68241708；去光57.2917%，
相位打乱47.2917%，轻微CCD噪声69.3750%。alpha=0.4286～0.4399；较旧约束版只多4张命中，不声称显著提升。
best SHA256=`c8509b44fbc0f7bcd6e1f0507376b6964bf483fca8205308407b790a295a100f`。
同轮SAM和仅L全场SAM均为68.9583%，未改善Hit@1。以上均为test-selected、同数据标签协议；旧版本不覆盖。

新对照 `regularized_control` / `regularized_phase05` 都从此69.7917%完整输入权重出发：
等比例缩放到原224画布的85%～100%后随机放置，保留整个物体、不裁切、不旋转；水平翻转25%、
亮度/对比度0.85～1.15、色彩0.9～1.1、15%概率轻微模糊(radius0.4)。仅训练增强，测试仍确定性完整输入。
电子二维以上weight使用AdamW weight_decay=0.01；相位raw、Router相位、alpha、bias和归一化参数不做衰减。
第一组不加新相位dropout，第二组每个训练batch增加expert/global 5%、Router 2%的8×8分块相位旁路，
逐样本独立。被选区域调制改为exp(i*0)=1，保持单位幅度，无1/(1-p)增益；推理/图库/干净评估关闭。
这不是移除专家，也不改Top2。已有25% batch的20%～30%DC与旧旁路/CCD噪声另行保留，二者明确分开。
此处phase dropout是物理相位旁路正则化，不等同于标准神经元dropout，也不是已经验证有效的结论。
参考[Dropout](https://jmlr.org/papers/v15/srivastava14a.html)、[AdamW](https://arxiv.org/abs/1711.05101)。

**准确率记录口径**：每epoch记录带增强/噪声的batch检索Hit@1和辅助分类accuracy；每5epoch以及最终best，
对同一live/EMA权重记录关闭增强/dropout/训练噪声的完整1440训练query与480测试query。
训练query排除自身整个商品（119候选、11正例），测试仍120候选、12正例；均按同类为相关，不用训练分类accuracy替代检索。
`history.json`内`test`/`test_live`各自包含`train_clean_leave_product_out`；`learning_curves.csv/png/pdf`
明确分别呈现干净检索、同权重train-test差距、随机训练批次指标。旧epoch没有对应权重时不虚构干净准确率，
仅使用已保存最佳模型的检索特征补算该best，结果放`runs/simulation/overfitting_audit_20260911`。
新训练只保留best/last，30epoch×64steps，batch40，单GPU串行，操作见COMMAND第11节。

补算已完成（不是训练batch准确率），固定各自best、eval模式、排除自身商品：

| 版本 | 干净训练Hit@1 | 测试Hit@1 | 差距（百分点） |
| --- | ---: | ---: | ---: |
| 原高alpha | 99.8611% | 68.9583% | 30.9028 |
| 完整输入AdamW | 99.9306% | 69.7917% | 30.1389 |
| 完整输入SAM | 99.7222% | 68.9583% | 30.7639 |
| 仅L全场SAM | 99.5139% | 68.9583% | 30.5556 |

证据在`runs/simulation/overfitting_audit_20260911/summary.json`，源13c4b2bc；所有缓存重算测试值与原报告一致。
summary SHA256=`1985dde0b501fbfdff00814d6a1502516fe54736f56de963ecc89d22b68268d7`。
本地已下载相同目录的CSV/PNG/PDF/JSON并核对摘要及最佳候选图SHA。没有新增历史epoch权重。
防过拟合新队列源码13c4b2bc、本地/服务器43项测试通过；
`runs/smoke/antioverfit_20260911`（启动PID3838308）等待V/L全场组结束，
正式`runs/simulation/antioverfit_20260911`（启动PID3840530）等待该真实训练检查成功后依次运行phase05/control。
等待进程不占GPU；当前进度以status.json为准。正式新分数未产生，不能把排队说成已训练成功。
