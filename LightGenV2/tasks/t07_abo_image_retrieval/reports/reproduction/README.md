# T07 复现说明入口

**83.125% 同权重光学噪声消融（2026-09-21）：** [完整定义、结果与复现命令](OPTICAL_NOISE_ABLATION_20260921.md)。
仅最后 L2 换噪声为 41.375%，V2+L2 换噪声为 32.875%，四个特征光学层全换噪声为 17.250%；同权重去光为 76.375%。

**83.125%独立交付入口（2026-09-17）：** [交付说明](../../delivery_83125/README.md)与[硬件适配合同](../../delivery_83125/AI_HANDOFF.md)。
由任务 `build_lab_package.py --profile enrolled83125 --protocol <固定protocol.json>` 生成；
仅打包固定best、离线processor、对应2400照片与原图复评证据，不混入旧类别检索权重。
交付根目录 `delivery.py verify/evaluate/finetune/export-phase`；具体命令见交付说明。
本包为独立仿真与实验适配参考，未包含厂商SDK或声称现场硬件已验证。

**老师审查冻结Qwen baseline：见[独立baseline说明与代码](../../baseline_reproduction/README.md)。**
该入口同时复现旧未见商品类别检索94.375%和新已登记SKU检索85.125%，不需要光学或其他任务代码。
两者正确答案/图库/查询组成不同；新预测只按同类别判断时为98.75%，仅是诊断，不替换同SKU指标。

## 当前协议：ABO已登记200商品，多视角图搜图（2026-09-15）

**先看本节；下方480查询/类别检索及替代数据集均是历史协议，不能把分数混到本表。**
**最新独立原图复评：正常83.125%=665/800、同权重去光76.375%=611/800，下降6.75个百分点。**
冻结Qwen64为85.125%，差距2.000个百分点。比上一83%净多1张，不宣称显著提升。
干净TRAIN图库自图排除94.625%=1514/1600，仍有11.5pp差距，不宣称过拟合消失。
800张TEST单独路由合格：V份额[26.750%,26.3125%,22.500%,24.4375%]、6组合、最大组合36.125%；
L份额[13.875%,30.5625%,26.875%,28.6875%]、5组合、最大组合32.125%。份额分母为每图两次选择。
alpha V=[.4477030933,.4471307099]、L=[.4449214339,.4451110065]，不等于性能或光功率贡献比例。
完整最新状态以[任务README](../../README.md)及下列run日志为准，不以文件夹名称推测完成。

### 最新83.125%：TRAIN同SKU内协方差收缩，原推理图不变

Run `runs/simulation/abo200_readout_within_sku_verified_20260915/`；
best SHA `c9926cbaaa1ef066d9657a8028dffc130a33915aa9f392551573dc79894192d0`；
`verification/final_report.json` SHA
`63f8ce8d545c230838c96b183d84b3ff85380ea9b164697b3d95c7b72a4fe432`。
源码`67566b888d56a1b1913e956012ac3567d7dd54f7`，463项本地/服务器回归；
Python3.11.15、torch2.6.0+cu124、GPU4 RTX4090、CUDA autocast BF16、batch4。
全2400原图正常/去光重新编码，67.60秒是加载与I/O在内的复评耗时，不是硬件benchmark。

起点为下节538477e0原83%。先读取它经原图核验的TRAIN1600个64维描述子并L2归一化，
逐SKU减去TRAIN均值，仅用类内残差估计C，再将C归一化到平均特征值1。
固定`A=(.98I+.02C)^(-1/2)`，写回原`W_new=A W_old, b_new=A b_old`。
**推理不计算协方差、不需要SKU标签、不减商品均值、不加载教师，也不新增层。**
实际度量特征方向增益范围[.9685503,1.0098274]，W/b合计Frobenius变化.08728988。
原六次10cm传播、Top2、所有相位与电子主干/alpha/metadata保持不变；这不是新的mask训练。
拟合只用TRAIN，QUERY仅比较收缩强度.05/.1/.02：82.875%/82.625%/83.125%，选择.02。
不使用teacher ridge的失败权重、不集成预测、不改变检索图库或相关性定义。

最初e5803db5源的同参数试验`abo200_readout_within_sku_whiten_s02_20260915`也经原图复评665/800，
但其报告把0 SGD步的统计拟合误记为未训练；已修复出处记录并重新生成本节PT、重跑原图。
两个PT的所有模型state和metadata逐值相同，只有源代码/校准履历记录不同。
本节PT正确记录`fitted_on_this_dataset=true,test_selected=true`；不覆盖旧报告。
正式说明只引用本节新报告。命令见[COMMAND110](../../COMMAND.md)。
同环境`source83_control/`重新从原图复评旧权重为664/800；新权重665/800，恰好1张错变对、0张对变错。
`verification/weight_train_audit.json`核验二者上游读出输入和router掩码逐值相同，所有相位/电子主干/alpha冻结。
早期83缓存有1张QUERY中间特征不逐位一致，原因未定；TRAIN统计全部相同。
不忽略这一发现：另跑的同环境旧权重控制已排除它解释本轮单张净提升的可能性。
原83权重保留，全部自有GPU进程已释放；本地同名run镜像含best/last、预测、复评与逐文件SHA清单。

### 上一83.00%：相位+原读出Top1短程训练

统一run：`runs/simulation/abo200_phase_head_top1_verified_20260915`。
best SHA `538477e0168c90cb7e0952d7c32ef5f3c4391568839a2865f86a21febf0b27b2`；
独立原图报告`verification/final_report.json` SHA
`8fd14ece4cb8cd25bafa259ba54ff193503769ebb2b3f6e28e135732676b7c8ec`。
训练与复评源码均`d111dd91d36700faeaee1bba41f3b51cbc35d639`，本地/服务器410测试；RTX4090、batch4。
原始run为`runs/smoke/abo_phase_head_top1_20260915`，`promotion_manifest.json`提供完整复制校验。
从下节b14a34ea起点，按COMMAND92节命令仅训练1轮2步，EMA.99获选；不是8轮训练。
只有12份相位和原Linear weight/bias改变，scope_audit逐值检查其余全部冻结、metadata相同。
双向TRAIN bank Top1 softplus、温度.1、margin .02，10%读出dropout、轻微亮度/对比度，
保留原10% batch的噪声/DC与路由均衡；不加推理模块、不改物理布局。QUERY不参与梯度。
best相位改变量约2e-6 rad、router约6e-8～9e-8 rad，非常小；不能把两张收益归于单一因素。
最终训练器评估和随后独立原图评估都为83.00%，mAP=.7297231523、NDCG=.7873187667。
同best去光重新编码全图库/查询，非单独重训；分split路由合格。所有自有GPU进程已释放。
周期TEST、EMA及跨run择优存在选择偏差，当前分数不是独立未触碰测试估计，也不是硬件结果。
完整训练/独立复评命令见[COMMAND92节](../../COMMAND.md)；原计划8轮未启动。

### 上一82.75%：固定种子读出复核

Run `runs/simulation/abo200_readout_top1_seed73_20260915`；只将下节训练seed42改为73，
起点仍为b1e205c7、不是从10925d29续训。原图mAP=.7296366567、NDCG=.7872969034。
best第150步SHA `b14a34ea12aad01a305d27596e4e47edade5610fae2e24190f452c7034b68cbe`，
原图报告SHA `b632cd9e4b32ca2af1994082ffea20085b7ce82f452761233a7604b01deb34b2c`，
源码737f9303、原图RTX4090/batch4，67.48秒不是硬件基准。参数/路由/TRAIN审计在verification/weight_train_audit.json。
预先固定seed42/17/73三组原图为82.50%/82.625%/82.75%，均值82.625%；共享已择优训练起点，
不是独立从头三次实验。不集成、不隐去较低结果，仍有TEST周期与跨run择优偏差。
seed17 best第100步SHA `9acfd172cd013b2fbae18705385c3e01d20560d0dd2b28b55f42e823944d1424`，
对应原图82.625%/去光76.625%、TRAIN94.6875%，所有证据在其同名run的verification目录。
复现使用COMMAND90节的固定种子命令；raw复评替换源run和由报告自动读取的SHA即可。

### 上一82.50%：TRAIN Top1排序读出seed42

Run `runs/simulation/abo200_readout_top1_20260915`；源码`737f9303da192413ec92d45415d4e5cc8224ebaf`，
本地/服务器406测试。原图mAP@10=.7304591394、NDCG@10=.7873025540，超过下节NLL版本，Hit@1持平。
仅原head weight/bias从下节权重继续训练，其余张量/metadata完全相同，alpha与上面的路由份额不变。
目标为TRAIN内最近正确/错误SKU的softplus排序（margin .02、温度.1），dropout .1、锚定.1、
lr .00005、800步、batch128、seed42，第450步按周期TEST Hit/mAP选中；不是训练epoch450。
完整拟合/复评命令见[COMMAND90节](../../COMMAND.md)。
best SHA `10925d290c732b600fb7976a9202339b59d63fd9cc1fed9fd50256f58bb01b26`；
原图报告`verification/final_report.json` SHA `d1635300d5cd5ae5708855767501621cc31999e3d798ee23b53eab79b8cbc5db`。
`verification/weight_train_audit.json`含逐张量比较、TRAIN自图排除及gallery/query路由门槛。
缓存82.625%不能代替原图82.50%；复评使用RTX4090/batch4，68.76秒仅任务运行时间，不是硬件benchmark。
CPU/复评进程已退出，原GPU0相位训练继续；83%仍是目标而非已有成绩。没有新增推理层或教师。

### 上一82.50%：训练期读出dropout与弱锚定

- Run：`runs/simulation/abo200_direct_readout_dropout_weak_20260914/`。
  起点为下节81.875%完整权重，只拟合原384→64读出的weight/bias，其他张量及metadata逐项不变。
  TRAIN/gallery1600张进入loss，800 QUERY只做周期选模；输入为完整光电主干输出经过原readout.norm的384维。
  原六次传播、相位、Top2、同尺度融合、alpha、电子残差、Qwen冻结前端不变，没有新增推理层。
- TRAIN-only query/gallery独立feature dropout=.1，NLL温度.1，相对起点平方偏离约束=.1；
  CPU4线程、batch128、lr.0001余弦退火、800步、每50步按TEST Hit@1/mAP择优，第450步最佳。
  参数直接写回原Linear。推理时不执行dropout、不需要教师；PT epoch表示CPU优化step。
- 源码`3c0ff5455014d7f51c70abf40cbdd806f015829f`，本地/服务器各400测试；[COMMAND第86节](../../COMMAND.md)
  给出缓存、拟合和参数对照。best SHA：`b1e205c70505de9df77ea52bed9c962bebacd3171e9f7b55b0569ec25f9fafd6`。
  原图4090/batch4完整normal/remove复评81.36秒；报告`verification/final_report.json` SHA
  `eb0901a854f6ac9bddefcfeae00ea4f15a52d94a1f7046c645173d191ee0cafc`。
- 正常mAP@10=.7247445437、NDCG@10=.7830821068；这两项略低于旧81.875%起点，不能声称全指标提升。
  `verification/weight_train_audit.json`记录TRAIN排除自身、参数不变性、数据/权重/缓存SHA、alpha与分split路由。
  标准余弦逐图库图排序，无SKU聚合、rerank、类别预过滤。缓存82.375%与原图82.50%相差一张，以原图为准。
- 400步的anchor1对照原图82.25%/去光76.25%，为`abo200_direct_readout_dropout_20260914`；
  dropout=.2对照未提升。原图复评PID1688116/1678450与CPU拟合均退出；旧光学再加热也已完成。
  周期TEST、跨run选择有偏差，单seed，不作统计显著性声明。83%目标仍未达。

### 上一81.875%读出校准起点：完整原图证据

- Run：`runs/simulation/abo200_metric_weak_anchor_20260914/`，在下述81.25%光优先起点上
  仅CPU拟合64x64线性度量，使用TRAIN1600的同SKU多正例、自图排除NLL和.1单位矩阵约束。
  共800优化步、每50步TEST择优，第350步最佳；这里PT epoch=350是优化步数，不是350轮原图训练。
- 将度量合并回原head：Wnew=A@W、bnew=A@b，仍单个384→64层，不新增推理参数/分支。
  逐张量核验除`readout.projection.weight/bias`外完全不变，metadata相同；光相位/alpha/路由不变。
  这不是重新训练光学相位；光学学习历史由下面光优先训练链提供。
- best SHA：`a579413726de92c22000f3379c185df83e604860990e57010a5bb39259aeb482`。
  拟合/原图复评源码`a4ff9ebb857bd2fc4e4ca1fecc55ed32c7f5b5ed`，395项测试。
  [COMMAND第85节](../../COMMAND.md)含完整拟合、SHA自动读取与GPU复评命令。
- `verification_4090/final_report.json`的normal/remove均来自重新读取全部2400张原图，batch4，
  正常81.875%/去光75.50%，mAP@10=.7264833209、NDCG@10=.7835496037。
  report SHA：`e434d4b52887d2551f9724348f55c1c6c5fc9427657340376c6c31d5bb0a0756`。
  `normal_features.pt`保存原图推理向量及逐图路由；TRAIN统计使用前1600个gallery向量，
  与协议train ID集合核对一致，做L2余弦检索、对角线置负无穷，1515/1600命中。
  原图去光结果不同于初筛缓存75.375%，只引用原图75.50%。
- gallery/query分别路由合格；V/L TEST分别6/5种组合，alpha沿用上述数值。
  首次`verification/`在读取模型前被无效长度SHA拒绝，未运行推理；成功结果在`verification_4090/`，
  自动核对报告SHA与实际best文件后运行，没有关闭校验。复评PID1559863已退出释放CUDA。
  只读复评与自有GPU4训练共享资源，总计仍两张GPU，用时不当作性能基准。

### 数据与比较口径

- 200个SKU，每个SKU有12张同一转台序列的不同视角；按固定哈希选8张TRAIN/gallery、4张QUERY。
  全部1600图库/800查询保留。正确答案是**同一个SKU**，不是同一物体类别。
  TEST商品都已登记，测试是未参与训练的视角，不宣称未见商品或独立拍摄场景泛化。
- `runs/simulation/abo200_enrolled_protocol_20260913/protocol.json`，SHA
  `f1749d5fc22d2dfee6a1333ce2b35e9fa600a070f949eba8420b4def41906dde`。
  TRAIN自检排除查询照片本身、保留同SKU其他7张；每个TEST查询有8张正确图库图。
- 冻结Qwen前64维、L2归一化、余弦检索的R@1=**85.125%=681/800**，未微调。
  原报告`runs/simulation/abo200_enrolled_qwen64_20260913/final_report.json`，源码
  `5f731fae979ef81a09770c1dd721b5cecd8b3392`；manifest与学生一致，不能替换为旧94.375%类别检索。
- 周期TEST选best、跨run选择均存在选择偏差；无单独validation，不能称独立无偏测试。
  此处R@1指Top1是否命中任一正确图库图，不是内部按8个正例计数的`positive_recall_at_1`。

### 光优先起点训练与文件入口（81.25%，保留来源）

服务器根目录：`/DATA/DATA1/guest3/2026OpticsMoE`。下列run均相对
`LightGenV2/tasks/t07_abo_image_retrieval/`，没有另建一份复制网络。

- 主run：`runs/simulation/abo200_optical_pretrain_20260914/`。
  训练源码`cf32e92939f23927dc717255c4849ab38549e819`，完整命令见[COMMAND第77节](../../COMMAND.md)。
  从原`abo200_route_distill_20260913/best.pt`继续，不是从头训练；起点SHA
  `d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a`。
- 外部池`abo_spin_pool50_views12_20260913`含453个额外SKU/5436图，manifest SHA
  `68fd35b6a2308f13e01963eb1233545c44eb07f5caa48ff655dc6fc302b1ed8f`。
  排除目标SKU/spin及筛出的目标精确/近重复图；不把目标QUERY用于loss。
- 外部36轮×100步仅训12份相位（958728参数）；电子、alpha、读出冻结，每轮SHA已核对不变。
  进入目标集后重置Adam/EMA，完成20轮×100步联合微调；每5轮评估live/EMA，按路由合格再R@1/mAP选best。
  一轮是100次采样更新，不等于遍历全部数据一遍。没有在这轮加入教师loss或扩容电子网络。
  外部expert/global基础LR=.002（原.0004的5倍）、36轮预算也不同；不能把提升单独归因于冻结电子。
- 仍为`standalone/model.py`、`frontend.py`和`optics.py`的原独立六次光计算：
  V router/expert/global + L router/expert/global、10cm/原ROI、Top2、同尺度融合alpha边界[.4001,.8]。
  推理无native Transformer/attention，原64维线性头；可训练2782485、冻结29152256，冻结前端没有解冻。
  受保护`optics.py` SHA=`6490c6ee0ccc7501572fbae722aafbd7d4a016425d21af454019e7b60625433d`。
- `best.pt`/`last.pt`是仅有的模型PT；`history.json`记录TRAIN/TEST和每轮相位变化，
  `execution.json`记录真实配置/命令/环境/commit/数据SHA；两个最终报告均已complete。
  增强单因素未提高（新训练最高77.375%），独立3%相位dropout也未提高（77.125%），都选回初始78.125%。
  命令见第78节，未将初始保底分数作为新训练成果。主训练/独立复评/dropout进程均退出，GPU1/3释放。

### 最佳权重与独立复评证据

- 最终选总epoch41=目标epoch5 EMA，不是最后一轮。best SHA256：
  `dcf768878abddd91558533757e404d9ee788cffbc5e0f0162a6f07d8a197eb1d`。
- 独立复评源码`2ee704e1cffbb54c6052afe83c0f131fdce5a47e`，Python3.11.15、torch2.6.0+cu124、
  RTX4090、batch4，完整命令见[COMMAND第80节](../../COMMAND.md)。无需下载完整Qwen模型执行学生。
  `verification/final_report.json` SHA=`f28c7f71240079e2bb7479a6603f5102cd54b1076e9f1bf82a4cb6cba0bdf65c`。
- 正常Hit@5=.91625、Hit@10=.95375、mAP@10=.7200305679563492、NDCG@10=.7788583475781425。
  去光mAP@10=.5263419518849206；无去光单独训练。独立复评不是新的独立测试集，仍披露TEST选模偏差。
- `verification/normal_predictions.csv`和`remove_optical_predictions.csv`保留全部800条预测，
  `normal_features.pt`含按ids对齐的V/L离散Top2掩码。正常与去光在原图上分别编码，不使用旧router状态。
- best的12份相位相对本run起点均改变：expert/global RMS=.1011～.1518 rad，router约.000742/.000310 rad。
  `phase_masks.png`为best相位可视化；没有用生成式图像替代真实权重。原光路/ROI/FP32相位保持不变。
- 本地最佳权重和报告镜像位于原项目根目录
  `LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/abo200_optical_pretrain_20260914/`；
  源码通过Git独立工作树与GitHub分支`experiment/t07-domain-refinement-20260912`同步，未覆盖其他任务源码。

## 历史：替代图检索数据集筛选（2026-09-13，非当前方向）

用户指定冻结Qwen64作为基准，停止追加ABO。研究候选、两组配对初筛结果与边界见[DATASET_SCREENING.md](DATASET_SCREENING.md)。
Grocery81短程适配源码`bc110024`，本地/服务器267项测试通过；两步CUDA检查完成，12份相位更新非零，GPU进程570024退出。
正式20轮run=`runs/simulation/grocery81_adapt_20260913`已完成，PID574285退出；epoch20 live测试24.6278%、训练62.8788%、去光7.5654%。
Qwen64同协议71.9517%，差47.324个百分点，不达标。旧报告去光router统计误读历史缓存，必须忽略；不影响检索分数，勘误见DATASET_SCREENING。
COIL以身份互斥的60训练/40测试物体做专门训练对照，统一入口`standalone.retrieval_adapt`，命令第62节，不改变原数据协议或光学结构。
本地/服务器270项测试与2步CUDA检查通过；源码ceba06e6，正式`runs/simulation/coil100_adapt_20260913`启动PID599950，GPU4 RTX4090。
该20轮现已完成，PID599950退出；epoch10 live测试88.75%、训练98.6765%、去光85.3125%，比Qwen64低10.625个百分点，尚未达标。
最佳权重SHA=`a220f144d6fd7d18bb25e11c647cdaea3b04f4cc964959de7f1d476e7f3f202e`。语言expert2测试未被选择，均衡仍有问题。
独立复评已完成，保存在该run的verification：源码3ce514e2，正常88.75%/去光85.3125%一致，272项服务器回归通过，PID618547退出。
具体勘误/限制看DATASET_SCREENING。Grocery原源码与新COIL源码不混称。本轮自有GPU任务全部释放，没有自动追加训练。
完整命令见COMMAND第61节。保留原六次光捕获、Top2、α>0.4、64维头，没有新增TF/attention。
原ABO最佳及下文证据全部保留，不把新协议与旧分数合并。

## 历史480查询协议最佳：独立复评79.5833%（2026-09-13）

原480-query/120训练商品中心图库，382/480，目标389/480还差7个命中。
固定权重`runs/simulation/readout_subspace_20260912/best.pt`，SHA256
`50a8607eec392c00cf3675533490cb8ef953245af6f5d9cfbc9d616bf7d22701`。
拟合/独立复评源码bbb9876d；RTX4090、batch4，从原图分别重建正常与同权重去光图库，PID393380已退出并释放CUDA。
正常79.5833%、mAP@10=0.7685648975；去光62.7083%，下降16.875个百分点；干净训练leave-product-out99.9306%。
报告/预测/逐类结果/相位图/环境/命令/manifest身份都在该run的`evaluation/`。完整拟合与评估命令见COMMAND第52节。
在下方79.375%起点上，仅用原1440 TRAIN单位描述子/类别均值拟合9维子空间P，
`T=P+0.5*(I-P)`折叠进原线性读出`W'=TW,b'=Tb`；推理描述子不减均值，64维均保留，不新增推理运算或层。
光学相位/传播/ROI、冻结前端、光Router Top2、α不变；训练参数2782485，冻结29152256。
α V=[0.4374556,0.4402898]、L=[0.4295430,0.4281099]，6次捕获、0TF/attention。
拟合矩阵未读TEST特征/标签，但0.5保留率经TEST缓存比较选出；周期TEST/跨run/超参数选择偏差必须披露。
只多1个查询，不宣称显著改善或光相位进一步训动。旧相位打乱/噪声分数属于下方未校准权重，不能挪用。
拟合输出删除旧辅助头与旧选模分数，不能直接套用要求原辅助头/教师坐标的续训profile。
旧权重、数据和报告不覆盖；原manifest及baseline仍见下文。复现环境与依赖沿用独立standalone工程。
本地同路径已下载evaluation的六份结果（不含best.pt）。六文件名→SHA字典按键排序、紧凑JSON编码后的SHA
`cee660b348553d2b1cad179596c88dab11cce08fef3f0738fbb0adf7d733aae5`，两端已核对。
final_report.json SHA=`57c9742f1dbb34e4da505b6ebf4527b250a26ae7b342f79c76162a5136319946`。

## 上一最佳：独立复评79.375%（2026-09-12）

381/480命中，仍为原120训练商品中心图库、同类别相关，目标389/480还差8个命中。
固定权重`runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt`，SHA256
`7c7bc601b1048be13316a9e4863780fa5d49cf2f66b3bf3e3d68ced2d8f172f4`。
来源`domain_joint_curriculum_20260912_gpu1/domain_distill_joint_curriculum`第8轮live，源码9e0d9544；
独立复评f695014b、RTX4090、batch4，正常/去光从原图分别重建图库，报告在固定副本目录的`evaluation/`。
正常79.375%、mAP@10=0.7623252315；同权重去光62.9167%，下降16.4583个百分点；干净原训练99.9306%。
相位图`evaluation/phase_masks.png`，逐图/逐类结果在同目录；执行记录保存环境、命令、数据清单与权重SHA。
复评PID2564306已退出，nvidia-smi确认CUDA释放；在自有GPU1与训练短暂同卡复评，总用卡仍为1、2、4。
α V=[0.4374556,0.4402898]，L=[0.4295430,0.4281099]；6次捕获、光Router Top2、原64维线性头、无TF/attention。
候选metadata及全部冻结frontend张量与78.75%起点一致；optics.py与9e9da3d8逐字节一致。
12份相位均有更新：专家/global圆周RMS约0.0332～0.0419 rad，V/L router约0.00434/0.00238 rad。
第8轮训练Top2选用率V=[0.4770,0.5109,0.5029,0.5092]，L=[0.4844,0.6307,0.4238,0.4611]；
每行合计2，这是训练统计，不冒充测试路由或硬件实测。
完整cap250训练5556图，原教师坐标/辅助头恢复，逐图教师余弦2+商品关系KL0.3，KL前三轮热身；
GT前4轮0.2，第5～8轮恢复至1，基础学习率而非1/4；第8轮已为完整GT1。教师仅训练，学生推理不变。
训练命令见COMMAND第35节，固定权重复评见第20节；原数据manifest仍为下方历史协议相同SHA。
单seed、周期test选best并跨run选择，只比78.75%多3个查询，不宣称显著性或目标达成；16轮训练已完成。
最终best与固定副本SHA相同。该训练final_report额外诊断相位像素打乱50.8333%、轻微专家/global CCD噪声79.7917%。
噪声为固定单seed123、无评估DC、router干净，不替代正常测试或硬件实测；原训练2542439已释放CUDA。
独立复评evaluation的6份结果（不含best.pt）已同步到本地同路径；六文件名→内容SHA字典按键排序、
紧凑JSON编码后的SHA为`e860337d3bc096590046b9a20ba7cab21076eb805b2cafbd541abf895aacfeac`，两端一致。

只读错误分布诊断（已有特征，CPU，无训练/重排/测试拟合）：
`evaluation/diagnostics/error_concentration.json`，使用35055dc7的`analysis.audit_for_meeting.score_features`，
原480/120重新计分得到同样79.375%，冻结Qwen native2048=95.2083%、native64=94.375%。
40个测试商品中21个12视角全对、3个12视角全错；99个错误中51个来自错误最多的5个商品。
99个学生错误里Qwen2048答对79个。错误最集中的商品包括chair、light fixture、wall art；
其中B071PDYXRQ双方均0/12，B071S5RK7P学生0/12而Qwen12/12，不能笼统归因于光或标签。
报告保存两份特征及原manifest SHA、全部40商品与分类别计数。它是test-selected模型的描述性诊断，
不把480相关视角当作480独立商品，不据此删除/改标/单独拟合测试商品；下一步仍检验训练数据覆盖和蒸馏。

训练特征共同分量诊断：`evaluation/diagnostics/train_centering_probe.json`，CPU、原1440训练行，
使用train-only教师缓存及当前最佳保存的训练描述子，不更新学生或读取测试描述子参与计算。
归一化教师前64维的训练均值范数0.77797；比较原教师坐标、重新拟合坐标、减去0.5/1倍该均值再拟合坐标。
在1360个共同有效训练行上，2×逐图教师loss梯度与GT检索NLL+margin梯度的平均余弦依次为
-0.37615、-0.39586、0.03445、0.43107；反向比例依次94.044%、95.882%、45.368%、3.971%。
教师64维原训练leave-product-out命中依次94.1667%、94.1667%、93.8889%、93.4028%，
因此不能单凭梯度更同向就认定泛化更好。GT此处不含proxy CE/光学辅助，梯度只到64维描述子，不代表全参数。
这与光学20%～30%未调制分量无关；若后续尝试，只改变训练教师目标，不改变相位几何或学生推理。
当前尚未将去均值加入正式训练，不能宣称已有性能收益；报告保存源特征/教师缓存/原坐标SHA。

## 上一最佳：独立复评78.75%（历史证据）

378/480命中，原120商品图库/同类相关；还差11个命中。固定副本
`runs/simulation/verify_teacher_first_ep11_20260912_gpu4/best.pt`，SHA256
`67a9c71d321720e5735afc9dc5214dbeeee017b6e4135912f92ed8fdabe3f3df`。
训练来源`domain_teacher_first_20260912_gpu4/domain_distill_teacher_first`第11轮live，源码9e9da3d8；
固定副本与当时训练best的SHA相同。独立复评9c6ff5c4、RTX4090/batch4，结果在该副本目录`evaluation/`。
正常78.75%、mAP@10=0.7613181217；同权重去光61.4583%，下降17.2917个百分点；干净训练100%。
相位图为`evaluation/phase_masks.png`，逐查询和分类别结果均在同目录。复评PID2449797退出且释放CUDA。
alpha V=[0.4372112,0.4398755]、L=[0.4291850,0.4280129]；光路/Top2/六次捕获/原线性读出未变。
12份相位相对77.9167%起点均更新；专家/全局圆周RMS约0.0512～0.0581 rad，两个Router约0.00374/0.00577 rad。
第11轮训练Top2选用率 V=[0.4973,0.4879,0.5303,0.4846]、L=[0.4934,0.6428,0.4088,0.4551]，
每行合计2，是训练全轮选用率，不冒充测试路由或硬件数据。
训练完整cap250、5556图，教师余弦系数2；前4轮GT乘0.2、5～8轮恢复至1，教师只在训练用。
第11轮实际系数教师2/GT1，没有新推理层。仅比前版多4个查询，仍有过拟合，不宣称显著或目标达成。
完整16轮训练现已结束，最终仍选择第11轮live，权重SHA与独立复评副本一致。命令见COMMAND第20节。
最终训练报告额外诊断：相位像素打乱50.2083%；轻微专家/global CCD噪声78.75%（单seed，无评估DC，router干净）。
监督2414473/训练2423765已结束释放GPU4；新6776e07f低学习率续训另存run，不覆盖本结果。

## 上一最佳：独立复评77.9167%（历史证据）

374/480命中，原120商品图库/同类相关。来源为已完成的一轮训练检查：
`runs/smoke/domain_aligned_feature_20260912_gpu1/domain_distill_aligned_feature/artifacts/best.pt`，第1轮live。
SHA256=`4fafe41ec23fcf0cb917f275ddab3be5c1686705c549ca1f58d68aa3c4829d3e`。
训练源码c26f13ec，独立复评3cc59c6d，RTX4090/batch4，复评目录`runs/simulation/verify_feature_ep1_20260912_gpu1`。
正常77.9167%、mAP@10=0.7358022487；去光59.375%，下降18.5417个百分点；干净训练99.8611%。
alpha V=[0.4369651,0.4394923]、L=[0.4290638,0.4280897]；原线性读出与V3/L5残差，光路未改。
检查run另外报告相位打乱52.50%、轻微CCD噪声78.125%，该噪声scope沿用下方说明，不能替代主指标。
训练为原1440图+cap250新增4116图，逐图教师余弦loss0.5/三轮热身（本轮实际系数1/6），不做外部筛选。
教师仅训练；12片相位均更新；无新增TF/attention。这里只训练1轮125步，不冒充16轮正式组已完成。
独立复评进程2401496已退出且释放CUDA。可用COMMAND第20节从同一固定best重算，输出目录须不存在。
相比77.50%仅多2个查询，未宣称显著性，仍是单seed/test-selected；还需15个命中才达到目标。

## 上一最佳：独立复评77.50%（历史证据）

固定原480-query/120-gallery、同类相关，372/480命中；未改测试数据或候选库。
`runs/simulation/verify_strong_ep4_20260912_gpu1/`保存固定`best.pt`、逐样本预测、
`phase_masks.png`、`execution.json`及`final_report.json`。
权重SHA256=`e5c0eaab4c84766b1ee231dd14271e97737604c0dcd675d9e2f4957c6932658d`。
来源`domain_distillation_20260912/domain_distill_strong`第4轮live，训练源码`6ab405fe`；
独立复评源码`5f3c704a`，RTX4090、batch4，正常77.50%、mAP@10=0.7266903108；
同权重去光59.375%，下降18.125个百分点；原训练排除自身商品99.8611%。
alpha V=[0.4369612,0.4394684]、L=[0.4290614,0.4281045]；无TF/attention，Top2、六次捕获。
5556训练图的冻结Qwen关系蒸馏权重0.3，仅训练使用，不增加学生推理模块。
24轮训练现已完成，重新加载的best与独立复评保存副本SHA一致；仍选第4轮live。
完整训练run的final_report补齐相位像素打乱51.0417%、轻微CCD噪声77.0833%；
后者仅专家/全局CCD增益和截断噪声，评估未加入DC/旁路、Router无噪声，不能冒充实测或固定20%零级测试。
12份相位参数均有更新；专家/全局相位相对起点的圆周RMS约0.0266～0.0319 rad。
训练进程2284756及监督进程已退出，GPU释放通过PID/nvidia-smi核查；GPU1随后接续电子7×7对照。
命令见COMMAND第20节；复评时checkpoint可改为本节固定best路径，SHA不变。
结果为单seed、test-selected；训练99.86%与测试77.5%的差距没有消失。

## 历史完整对照：76.25%（2026-09-12）

固定原480-query/120-gallery、同类相关，原manifest SHA如下方；未改变类别或测试名单。
当前权重目录为`runs/simulation/domain_refine_wide_20260912_gpu2/domain_refine_wide/artifacts`，
`best.pt` SHA256=`6f23466a1570a024e5bcf8408ae70a01065399e28e818af5a394dff15bfa850a`。
训练源码`ac737d41`，RTX3090，24×128主batch，从75.2083%起点续训；第16轮EMA为best。
新增2058外部同类商品4116图仅训练，pool SHA=`5348ffec1ba51a547ee1be81341e1c4bbbf1b970b9aeec47e557ddd7888f8488`。
正常Hit@1=76.25%（366/480），mAP@10=0.73225；同权重去光60.625%（下降15.625个百分点），
相位打乱47.9167%，轻微CCD噪声76.0417%，原训练排除自身商品Hit@1=100%。
alpha V=[0.4370494,0.4396843]、L=[0.4291205,0.4279832]。仍有过拟合，不能宣称泛化问题已解决。
训练光相位/Router及电子残差/读出，冻结紧凑Qwen前端；无完整Qwen/attention推理、无教师训练。
完整训练命令在COMMAND第13节，配置/环境/原始命令在同目录execution.json；
训练程序结束时已经重新加载这个best完成正常/去光/打乱/噪声复评，结果写入final_report。
注意：现有CLI evaluate读取`--assets`中的best.pt，旧`standalone_assets_20260910`仍是旧交付权重，
不能用旧assets命令冒充76.25%复评。本轮未覆盖交付包；新CLI已支持`--checkpoint`及必需的
`--expected-checkpoint-sha256`显式固定权重，命令见COMMAND第20节。评76.25%时需替换成上面的
wide best路径/SHA及未使用的输出目录，第20节原样命令对应本页最上方当前最佳候选。
新增试验不覆盖最佳：训练蒸馏（COMMAND14–16）、cap500+1:3采样（17）、现有电子卷积7×7（18）。
7×7方案是电子结构调整，不是改光路；真实4张训练图CPU完整前向与原起点逐值一致，
所有光学权重未改、增15,360参数，源码`2ca45836`，本地/服务器66测试通过；正式成绩待实测。
辅助头“完整恢复”更正见任务README，旧resumeaux会重置类别proxy，不与full版本混报。

## 81%目标：训练期关系蒸馏约束

仍是同一480查询/120原训练商品图库，389命中才达到≥81%。不更换任务、类别或评估名单。
`domain_distill_light/strong`从已完成75.21%best出发，5556张训练图（1440原train+4116外部图）
离线冻结Qwen2048D教师，绝不对val/test生成蒸馏目标；仅向学生传递训练商品间的相似度关系。
学生推理图/alpha>0.4/六次光捕获/optics.py保持不变，教师只在缓存生成子进程存在。
teacher正确性门控和权重0.1/0.3、完整命令见任务README及COMMAND第14节。
此为新训练假设，不将教师95.21%算成学生成绩；运行后以execution/history/final_report为证据。

## 2026-09-12：混合训练75.21%与冻结大模型复评

训练源`09835251`；固定测试480query、原120商品图库、同类12正例、原manifest
`2949a4035150a9f8718f2a6cace164c17394613d24fb9d0234c553bee8d77c97`。
三组40轮都已完成，best包含初始69.7917%，新best均不是epoch=-1。

| runs/simulation中的run | 最佳epoch/权重 | Hit@1 | 同权重去光 | 干净原训练Hit@1 |
| --- | --- | ---: | ---: | ---: |
| domain_target_control_20260911 | 20/live | 70.4167% | 56.8750% | 99.9306% |
| domain_mixed_20260911 | 15/live | 75.2083% | 60.2083% | 99.7222% |
| domain_curriculum_20260911 | 20/EMA | 72.5000% | 60.4167% | 99.7222% |

证据位于`<run>/<profile>/artifacts`；训练检索排除自身整个商品，119候选，不是分类辅助头准确率。
75.21%best SHA256=`0e523f3a08631e0d248032c58534761758841e23702e155df8f2d3f4cd472f85`。
75.21%为361/480，mAP@10=0.71290385；相位像素打乱52.9167%，轻微CCD噪声75.0000%。
原数据/评估不变，新增922商品仅用于训练；α0.4281～0.4395，正常/去光下降15个百分点。
GPU0/1/3各一张RTX4090；全部进程已退出，status记录gpu_context_released=true。
单seed且test周期选模，不宣称统计显著或已解决过拟合，不替换旧交付包。

冻结Qwen重新从图像计算：`runs/simulation/frozen_qwen_20260912`，source09835251，GPU0 RTX4090。
无微调、0训练参数，原尺寸2048D Hit@1=95.2083%（457/480），前64维94.3750%；
中心裁切224平方2048D为93.9583%，64D为92.7083%。主指标复现，其他变体小幅波动，不能称逐位一致。
模型snapshot仍为`9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`；精确指令与预处理见run_manifest。
baseline_report SHA256=`5678c420aa3bc3b5508edee338e20185d95a5e87b1b51ca07b4de295543d43da`。
速度/能耗未测；此完整Qwen仅作baseline，不进入光电学生推理/训练。

下一轮`domain_refine_control / wide / views`协议及完整命令见任务README/COMMAND第13节。
新池2058商品4116图，包含全部旧922商品，排除原200商品；只按metadata标注，不冒充人工核验。
池SHA256=`5348ffec1ba51a547ee1be81341e1c4bbbf1b970b9aeec47e557ddd7888f8488`。
三组起点为75.21%同一SHA，24×128主batch，wide仅改变训练池，views仅增加同商品跨视角cosine约束。
views训练计算量更多，推理结构不变；先CPU测试及双视角GPU短检查，再正式运行。新成绩待实际报告。

## 较大数据预训练 → 10类迁移（2026-09-10）

数据准备commit `21ada451`，训练commit `fcdbb2e8`。不改变独立模型推理结构，固定原alpha，
训练光学相位/Router/电子残差/读出；Qwen前端冻结，无完整Qwen教师。
新增64维归一化类别代理头仅用于训练分类loss，检索评价不调用它；同类别跨商品监督对比。

预训练池位于`runs/simulation/broad_pool_20260910/`，原图仍在根data/abo，不复制/移动。
原始145615商品、576种metadata product_type中，均衡选128类型×48商品×2图：**6144商品、12288图**。
不是1000类，也不是全ABO。清单SHA256：`2c891c4b50c8c8a98fad603a56ccb9a03024061386d236b6a1aa04afe9999c3a8`。
目标manifest SHA256：`2949a4035150a9f8718f2a6cace164c17394613d24fb9d0234c553bee8d77c97`。
排除全部200个目标商品（含train/val/test），共享image ID检查，再用文件SHA与128bit dHash筛查。
商品和image ID交集均0；55个候选商品触发目标精确/近重复筛查；另有1123次池内重复图片跳过。
近重复是启发式，不保证所有语义相近商品变体均已排除。商品类型标签来自元数据，存在粗细粒度混杂。

20项本地测试通过；`runs/smoke/broad_chain_20260910/`完成预训练、迁移、正常/去光完整流程检查。
短检查预训练V/L router raw相位RMS更新0.000131/0.000078，全局相位0.001398/0.001386；alpha更新严格0。
不把两步训练的结果当作性能优化结论。

正式串行run：`runs/simulation/broad_transfer_20260910/`，日志`console.log`。
`artifacts/pretrain/`：20epoch×120steps，batch32；`artifacts/adapt/`：30epoch×48steps，batch40。
采样不保证每epoch覆盖所有图，history记录unique_images。精确配置/命令/环境/权重SHA在各阶段execution.json。
GPU UUID `GPU-1b963983-7909-af6e-0528-f0f0661ab549`，启动PID4093748；只有这一张GPU。
运行时状态以history/final_report为准；尚无目标微调final_report时，不得声称达到了75%。

完整重建/运行命令见任务[COMMAND.md第6节](../../COMMAND.md)。源代码从GitHub checkout该训练commit后执行，
默认只保存各阶段best.pt/last.pt。预训练选择按训练loss对应的EMA快照，无目标test参与；
迁移仍按用户既定test-selected口径，每5epoch测试，并以原70.2083% best作保底。
75%对应同口径480张query至少360张Hit@1正确；达到后还需检查去光差值与相位、专家分布，不能只看辅助分类头。

## 训练优化：teacher_curriculum（独立版，不改变推理结构）

源commit `6d568dfc`；18项本地测试通过，`runs/smoke/curriculum_20260910`完成三阶段小检查。
已完成30epoch，训练候选最高70.00%（epoch3），末轮67.9167%，最终恢复accepted epoch0的70.2083%。
没有获得提升，PID3697546已退出，不占GPU。以下启动信息仅保留作历史证据。
固定alpha的raw参数更新严格为0；V/L router raw相位RMS更新0.000325/0.000237，
全局相位0.003370/0.002285，确认冻结和重新加热均生效。短检查不是新性能结论。

正式run：`runs/simulation/teacher_curriculum_20260910/`；日志`console.log`，结果在`artifacts/`。
初始PID 3697546，GPU UUID `GPU-1b963983-7909-af6e-0528-f0f0661ab549`；单卡。
不要重用非空输出目录、不要覆盖现有70.2083%交付best。动态进度以history/final_report为准，
没有final_report时不能把计划中的30epoch写成已完成。

在服务器源码工作树根目录（需先fetch并checkout该commit）执行：

```bash
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/guest3/miniconda3/envs/xml/bin/python -u -m LightGenV2.tasks.t07_abo_image_retrieval.run finetune \
  --profile teacher_curriculum \
  --assets LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/standalone_assets_20260910 \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --output LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/teacher_curriculum_20260910/artifacts \
  --device cuda --epochs 30 --steps 48 --batch-size 4
```

开始best与数据身份继承下方独立验收；实际SHA、训练配置、源码环境记录于artifacts。
只用1440张训练图及对应教师缓存；test仅按既有用户口径选模（test-selected，不无偏）。
alpha保持原best四个数值，不通过继续降alpha换指标；40样本跨商品batch。
收尾停用特征/关系蒸馏，但分类CE的中心仍取自训练教师缓存。

## 当前独立版本（2026-09-10）

日常命令已移至任务根目录 [COMMAND.md](../../COMMAND.md)。`run.py`默认只运行standalone；
下方历史训练命令明确改为`legacy_run`，仅维护者审计时使用，不能发给接收人当日常入口。

独立代码只使用PyTorch、processor/tokenizer等安装库；不导入其他任务或experiments。
Qwen参数由safetensors CPU按键读取，仅保留patch/position/merger与固定prompt的词表行。
不创建AutoModel/Qwen3VLModel，不加载TF/attention/LM-head，不读取原checkpoint优化器。
首次迁移前端+best+processor+仅训练教师目标共85,580,305字节；原始checkpointSHA保持可追溯。

从仓库根目录执行一次性CPU导出（普通接收人已有assets，不执行）：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.export \
  --qwen /DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --checkpoint LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/polish_phase_reheat_20260910/best_checkpoint.pt \
  --teacher-cache LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/frozen_qwen_20260909/features.pt \
  --output LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/standalone_assets_20260910
```

首轮全量独立评估run=`standalone_verify_20260910`，源commit=`b768403f`：
4090 Hit@1=70.2083%、mAP@10=0.68001116、同权重去光67.2917%；
原4090为70.2083%/0.68003894。测试embedding平均余弦0.99999756，RMS差0.00027684，
并非逐位一致；不篡改原参考特征，不把微小排序差隐藏为“完全相同”。
最终验收：`standalone_isolated_verify_20260910`，commit `172c8092`，隔离目录禁止旧仓库导入，
Hit@1=70.2083%、mAP@10=0.68001116、去光67.2917%。详细检查与资源记录见 [ACCEPTANCE.md](../../ACCEPTANCE.md)。
交付为任务 `releases/t07_abo_standalone_70_20260910.zip`，包含数据/最优权重/独立代码；以下历史段落保持历史含义。

独立微调保留当前训练目标和鲁棒措施，但改成显式模型后随机数消费顺序可能改变；
因此只主张固定权重性能复现与可训练性，不承诺重新训练逐epoch等同历史。
首次独立CPU测试5项通过，原合同加独立测试共16项通过。
GPU默认1张、上限2张；全量评估/微调验收串行执行，完成后检查对应PID退出。

---

# t07_abo_image_retrieval 复现说明入口

## 2026-09-09 新一轮图搜图（六次光传播）

环境：服务器 `/home/guest3/miniconda3/envs/xml/bin/python`；需要 PyTorch CUDA、transformers、Pillow、PyYAML、numpy、matplotlib。
使用 GPU 数字编号前设置 `export CUDA_DEVICE_ORDER=PCI_BUS_ID`，或直接以 GPU UUID 设置 `CUDA_VISIBLE_DEVICES`，避免编号与 nvidia-smi 不一致。
数据保持 `/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data`，不移动原数据。
Qwen 与 warmstart 的服务器绝对路径在 `configs/optical_top2_dc20.yaml`，其他电脑先改这两个位置及 dataset_root。
代码必须先从 GitHub checkout 对应 commit；以下从仓库根目录执行。GPU 编号需按空闲显存调整。

```bash
python -m unittest discover -s LightGenV2/tasks/t07_abo_image_retrieval/tests -v

# 1. 重跑冻结大模型，自动输出 native/square × 2048/64D 四组结果及教师缓存。
CUDA_VISIBLE_DEVICES=1 python -u -m LightGenV2.tasks.t07_abo_image_retrieval.legacy_run \
  --mode baseline \
  --run-dir LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/frozen_qwen_20260909

# 2. 首次先做一轮小检查（评估仍覆盖完整图库/测试集）。
CUDA_VISIBLE_DEVICES=1 python -u -m LightGenV2.tasks.t07_abo_image_retrieval.legacy_run \
  --mode optical --epochs 1 --steps 2 --eval-interval 1 \
  --run-dir LightGenV2/tasks/t07_abo_image_retrieval/runs/smoke/optical_top2_dc20_20260909

# 3. 正式训练；不能复用已有 checkpoint 的 run-dir。
CUDA_VISIBLE_DEVICES=1 python -u -m LightGenV2.tasks.t07_abo_image_retrieval.legacy_run \
  --mode optical --epochs 40 \
  --run-dir LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/optical_top2_dc20_20260909

# 4. 可选的同结构训练对照：只加强训练集语义中心/KD监督，不新增推理头。
CUDA_VISIBLE_DEVICES=4 python -u -m LightGenV2.tasks.t07_abo_image_retrieval.legacy_run \
  --mode optical --epochs 40 \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/optical_top2_dc20_anchor.yaml \
  --run-dir LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/optical_top2_dc20_anchor_20260909

# 5. 固定 best 重评，不重新训练，不需要教师特征缓存。
CUDA_VISIBLE_DEVICES=1 python -u -m LightGenV2.tasks.t07_abo_image_retrieval.legacy_run \
  --mode evaluate \
  --checkpoint LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/optical_top2_dc20_20260909/best_checkpoint.pt \
  --run-dir LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/optical_top2_dc20_reeval_20260909
```

baseline 看 `baseline_report.json`，光学看 `history.json`/`status.json`/`final_report.json`；
最后的相位图为 `best_phase_overview.png`。run_manifest 记录源码 commit、环境、原命令和数据清单 SHA。
`features.pt` 是固定教师缓存，不是可部署的光学权重；部署/后续微调用 `best_checkpoint.pt`。
baseline 2048D 原长宽比的历史 Hit@1=95.2083%，本轮是否复现必须以新报告为准。
训练仅用 train；test 每 5 epoch 参与选模，明确不作为独立泛化估计。

## 继续优化：训练方法 / 电子增强的配对对照

两组都从 `optical_top2_dc20_anchor_20260909/best_checkpoint.pt` 开始（SHA 在配置中强制校验），
重新初始化优化器。固定旧 baseline、训练/测试名单、全 120 商品检索、64D 输出和光路。
只保存 best/last；80 epoch，每 5 epoch 完整测试选 best，不使用验证集。

```bash
# 先同步本次 Git commit；GPU UUID 按空闲显存选择，建议留出至少 10 GB。
python -m unittest discover -s LightGenV2/tasks/t07_abo_image_retrieval/tests -v

# 训练方法组；原推理结构不变。
python -u -m LightGenV2.tasks.t07_abo_image_retrieval.legacy_run --mode optical \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/refine_training.yaml

# 相同训练方法，再扩大电子残差卷积核 + 非线性读出。
python -u -m LightGenV2.tasks.t07_abo_image_retrieval.legacy_run --mode optical \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/refine_electronics.yaml

# 增强模型重评必须传其对应配置，不能用旧的默认线性头配置。
python -u -m LightGenV2.tasks.t07_abo_image_retrieval.legacy_run --mode evaluate \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/refine_electronics.yaml \
  --checkpoint LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/refine_electronics_20260909/best_checkpoint.pt \
  --run-dir LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/refine_electronics_reeval_20260909
```

训练 batch=20：十类各取两个不同训练商品的一张图，48 step/epoch（随机采样，非保证全覆盖）。
轻增强先做与测试一致的方形裁剪，再做 0.94–1.0 裁剪和 ±5% 亮度/对比度；不翻转。
KD 使用同一训练图片的干净视图缓存，明确属于增强一致性目标，不是重新执行教师。
5 epoch 预热 + 余弦学习率，KD 从 0.5 降至 0.1；监督对比/训练类中心监督保留。
电子增强仅在原两路残差内扩大卷积核至 9，末端读出增加 512 隐层的 GELU 修正，
无 attention/Transformer、无教师推理旁路。卷积新增系数、读出修正输出初始化为零，
初始函数保持旧权重行为（浮点运算误差除外）。真实 kernel、读出参数和活跃图见 architecture.json。
仍保留光 Router Top-2、同尺度融合、20%–30% 未调制训练分量、既有 CCD 噪声；像素偏移仍为零。

## 商品图库损失与关系蒸馏（第二轮继续训练）

源模型 `refine_training_20260909/best_checkpoint.pt` 已完成80epoch，Hit@1=69.5833%，
同权重去光67.0833%，mAP@10=0.67616634。SHA256由两份新配置继承并强制核验。
对照的电子增强组完成80epoch，best60：Hit@1=67.0833%，去光66.6667%；不采用它作为续训起点。

```bash
# 每组60 epoch，沿用上述 Python 环境与数据/特征缓存；从仓库根执行。
python -m LightGenV2.tasks.t07_abo_image_retrieval.legacy_run --mode optical \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/refine_gallery.yaml
python -m LightGenV2.tasks.t07_abo_image_retrieval.legacy_run --mode optical \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/refine_gallery_relation.yaml
```

两组推理图完全相同，保持原V卷积核3、L卷积核5、线性64D读出、六次光传播。
每5epoch重新编码1440张干净训练图，建立120个训练商品中心；每个训练batch再以0.5动量更新
访问过的图像特征，中心为逐图L2→按商品平均→L2。训练查询排除自身商品的整个中心，
故每次是119个有效候选、11个同类正商品；最终测试仍是原120候选、12正商品，不改协议。
损失为所有正商品概率之和的负对数，加0.1监督对比；固定教师类中心CE移除，
逐向量KD从0.1退火到0。关系组另加0.5 KL：同一训练查询在完整2048D教师空间中对训练商品
中心的相似度分布，监督学生64D空间的商品分布；教师/学生温度均0.1，自身商品两边都排除。
教师只用已有square缓存的train部分，不加载教师Transformer做学生推理。
所有memory均stop-gradient，查询分支可导；memory不属于部署模型，恢复训练时重建。

训练时显式恢复phase dropout，全图库刷新和test时关闭。更正此前说明：继承的
`FourLayerOpticalReplacement.set_student_train_mode()`本来就会恢复phase dropout，
因此旧训练并没有“测试后不恢复”的缺陷；新profile的显式调用只是重复保障，不改变该行为。
保留20%–30%未调制训练分量、CCD噪声、Router均衡。
仅保存best/last和最终报告、图；test每5epoch选best，仍属于test-selected结果。
训练memory定义另写入每个run的 `training_gallery_contract.json`。

## 2026-09-10：原目标低学习率续训 / 相位再加热

图库两组60epoch均已结束，Hit@1分别68.75%和68.5417%，未替换69.5833%的旧best。
本轮从 `refine_training_20260909/best_checkpoint.pt` 继续，恢复其原损失：0.5监督对比、
1.0训练教师类别中心CE、0.1逐向量KD（固定，不再退火）；无全图库loss或关系KL。
仅30epoch，batch=20，48step/epoch，EMA0.99，5epoch测试一次；best/last两份权重。
相同seed与增强，同一路径/几何/Top-2，无新网络。对照使用上一轮终点附近的学习率；
另一组仅将特征相位LR从0.0004提高到0.004，Router与电子LR不变。

```bash
python -m LightGenV2.tasks.t07_abo_image_retrieval.legacy_run --mode optical \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/polish_low_lr.yaml
python -m LightGenV2.tasks.t07_abo_image_retrieval.legacy_run --mode optical \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/polish_phase_reheat.yaml
```

目标是原480-query Hit@1超过0.70，不改query、图库、标签、指标或评估预处理。
测试仍参与选模，小幅超过阈值不等于统计显著改进；完成后固定best再复评，报告命中张数和去光结果。

新增第三个有界对照 `polish_smoothing.yaml`，与low_lr仅相差anchor CE的label_smoothing=0.1，
推理不变。这基于旧best缓存的训练集诊断：1440训练图查询119个其他训练商品中心，
Hit@1=99.375%（1431/1440），不属于独立测试；未见商品仍为334/480。
不得把训练图的该诊断指标填入论文测试性能。

```bash
python -m LightGenV2.tasks.t07_abo_image_retrieval.legacy_run --mode optical \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/polish_smoothing.yaml
```

## 2026-09-10 完成结果与固定权重复评

本轮保留原推理结构、480张query、120个训练商品中心、Top-2、同尺度融合、训练直流/CCD噪声。
单seed，三个预设30epoch对照；每5epoch测试选EMA best，**不是独立无偏测试，也不是多seed统计**。

| 方案 | best epoch | Hit@1 | mAP@10 | 同权重去光Hit@1 |
| --- | ---: | ---: | ---: | ---: |
| 旧refine_training | 80 | 69.5833% | 0.67616634 | 67.0833% |
| polish_low_lr | 10 | 69.5833% | 0.68140021 | 67.0833% |
| polish_smoothing | 30 | 69.3750% | 0.67446313 | 67.5000% |
| phase_reheat最终恢复best，A100 | 10 | **70.0000%** | **0.67950967** | **67.0833%** |
| phase_reheat固定best，RTX4090复评两次一致 | 10 | **70.2083%** | **0.68003894** | **67.2917%** |

phase_reheat的A100训练周期best为70.0000%（336/480）；RTX4090固定权重复评为337/480。
两次4090逐query CSV完全一致；跨GPU仅query `4959f7d1e5955571` 的Top-1正确性由错变对。
两设备测试embedding平均余弦0.99979985、元素RMS差0.00250140；评估沿用bfloat16 autocast，
与微小数值差影响近邻排序相符，但没有进一步隔离到具体算子。保守主表记70.00%，不挑硬件较高结果充当额外训练提升。
相较旧best仅增加2～3张命中，不能因此宣称统计显著或已经接近95.2083%的冻结baseline。

训练源码：low_lr和phase_reheat为`40544b2b`；smoothing及固定复评为`74da80d0`。
后者只新增可选训练CE label_smoothing，phase_reheat配置未启用，推理图不变。
新best SHA256：`3674981c3499555077c7eadc3072a11e07583675000ec4c012616a96185867f5`。
续训起点为旧best SHA256 `2d588f40a1aa9f09336745b1e14a61c0cca76874f4c6d4f24a3d6163b978f20c`。
V1/V2/L1/L2 alpha分别约0.10270/0.10415/0.08809/0.08742；这是融合系数，不等于性能贡献。
同权重去光的A100和4090差值均为2.9167个百分点；不另训纯电网络。
三组训练及两次独立复评均已complete。最佳相位raw参数相对此轮起点的RMS变化约0.0502～0.1037，
确实发生更新；这不是包裹相位的弧度距离，也不把训练末轮live权重变化当作best变化。
best epoch的live训练Top-2选择计数：V=[478,473,488,481]，L=[495,481,489,455]；
份额均约23.7%～25.8%，本轮训练未见明显单专家集中。这不是EMA best在test上的使用率。

服务器唯一结果根目录：
`/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/`。

- `polish_phase_reheat_20260910/`：best/last权重，训练history、配置、最终报告和`best_phase_overview.png`、`comparison.png/pdf`。
- `polish_phase_reheat_reeval_epoch10_20260910/`：首次4090固定复评，`final_report.json`、逐query CSV、检索特征。
- `polish_phase_reheat_reeval_repeat_20260910/`：第二次独立4090进程复评，相同权重SHA和指标。
- `polish_low_lr_20260910/`与`polish_smoothing_20260910/`：失败对照，保留审计证据，不替换best。

从仓库根目录运行（output必须使用不存在的新run目录；不需重训/教师特征缓存）：

```bash
CUDA_VISIBLE_DEVICES=GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/guest3/miniconda3/envs/xml/bin/python -m LightGenV2.tasks.t07_abo_image_retrieval.legacy_run \
  --mode evaluate \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/polish_phase_reheat.yaml \
  --checkpoint LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/polish_phase_reheat_20260910/best_checkpoint.pt \
  --run-dir LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/phase_reheat_fixed_check_new
```

GPU UUID是本服务器4090；在其他机器需替换为当地空闲GPU。数据与冻结Qwen前端仍需按前文准备。

## 2026-09-11：严格高alpha最新证据与独立审阅版

`high_alpha_retrieval_20260910/artifacts`：30epoch完成，source `e59fc45fbec17938e76f4bfd527e5540003bb255`；
epoch15 EMA，Hit@1=68.9583%，mAP@10=0.675709，同权重去光64.5833%，去光下降4.375个百分点。
alpha约0.430–0.440，约束区间[0.4001,0.8]；没有选择旧低alpha保底。仍为train120/val40不用/test40商品、
480query×120训练商品中心、test-selected，数据清单与冻结baseline一致。
best SHA256=`814893fb430a73ce529acd7a1a80e264d7ed0253a37658df936cc40e34d6265b`；
manifest SHA256=`2949a4035150a9f8718f2a6cace164c17394613d24fb9d0234c553bee8d77c97`。

独立整理版在根 `LightGenPublic/tasks/t07_abo_image_retrieval/`，只用紧凑冻结前端，不加载完整Qwen；
源码29e16bed完成隔离全量复评：1440train+480test的64维特征与原代码逐位一致、最大绝对差0。
Python3.11.15、torch2.6.0+cu124、单卡4090。此为固定权重数值复现，不是从头重训保证。
另执行一epoch一步的真实联合续训冒烟并验证12片相位更新；训练/复评/冒烟进程均已退出，GPU释放。

完整命令/模型/资产要求见独立版 `COMMAND.md`、`README.md`、`docs/VERIFICATION.md`；
组会分析见 `docs/GROUP_MEETING.md`。原始细粒度审计在本任务 `runs/simulation/meeting_audit_latest_20260911/artifacts`，
独立复评在审阅任务 `runs/fixed_equivalence_20260911`。新的代码ZIP是仿真/续训内部审阅包，不是硬件SDK包。

## 2026-09-11：SAM与输入裁剪修正（运行中，非最终结果）

唯一结果目录为本任务 `runs/simulation/generalization_20260911/`，源码
`c1f4b4716ee001e962ef0d33816f854ff6c81d52` 已同步GitHub，完整启动命令见 [COMMAND第9节](../../COMMAND.md)。
单卡4090、Python3.11.15/torch2.6.0+cu124，原始数据、split、480-query/120训练商品中心及上节源best的SHA256不变。
按 `preserve_sam → preserve_adam → preserve_fullfield_sam` 串行，每组独立从同一68.9583%高alpha权重开始，
不是前一组接下一组。各30epoch×64steps，batch40，评估batch4，每5轮比较live/EMA测试Hit@1再mAP，
仅保存best/last；test-selected且单seed，不是独立测试估计。前25轮联合训练，末5轮仅读出。

三组输入都从中心裁切改为等比完整缩放、白色补边224²，训练/图库/测试一致；禁止裁剪和旋转增强。
SAM组rho前三轮0.01/0.02/0.03，两次前向重放同一随机噪声/dropout，参数精确恢复后按第二次梯度更新。
第三组另外将L端CCD全场汇聚为77×224，而非pool224²后截前77行；V端、光路、参数数量均不改。
不新增TF、attention或教师。alpha>0.4、原有光学噪声/未调制分量/路由均衡保留。
第一组与旧成绩还同时存在适配学习率/训练变化，不能把提升全部归因于裁剪；SAM对照之间配置匹配，
但同step计算约两倍，不是等GPU时间。更改后的输入/读出写入checkpoint，旧合同成绩不能作为新合同保底。

本地/服务器32项测试通过。`runs/smoke/generalization_20260911`（源码fc11abdd）完整通过两步SAM及最终
正常/去光/相位打乱/噪声复评，12片相位更新非零，损失有限；结束后PID3295086/3295089退出、GPU恢复空闲。
后续源码仅修正审计字段与队列顺序，不改变数值计算。正式队列PID3329183、第一组PID3329218为启动记录；
当前活动进程/完成指标以队列status.json为准。正式提升待实测，不把冒烟结果填入性能表。

## 2026-09-11：完整输入结果与干净训练准确率记录

源码c1f4b471的三组已完成：`generalization_20260911/preserve_adam/artifacts` epoch25 live为
Hit@1=69.7916667%（335/480）、mAP@10=0.68241708，去光57.2916667%，相位打乱47.2916667%，
轻微CCD噪声69.375%。best SHA256=`c8509b44fbc0f7bcd6e1f0507376b6964bf483fca8205308407b790a295a100f`。
SAM和仅L全场SAM都是68.9583333%，尚无额外Hit@1收益；不能把新的权重重新适配过程说成从50%提升到69%的独立突破。
相对旧68.9583%仅多4张命中，test-selected/单seed，不宣称显著性。V/L全场组为另一未完成实验。

源码13c4b2bc用各自最终best保存的检索特征做CPU复核，按sample顺序和原manifest SHA校验；
查询训练集时排除自身整个商品，而不是只排除当前图片。训练1440查询×119候选、每查询11正例；
测试480查询×120候选、每查询12正例。前者仍是见过的训练商品，不代表新商品泛化。
原高alpha/完整输入AdamW/完整输入SAM/仅L全场SAM的干净训练Hit@1分别为
99.8611%/99.9306%/99.7222%/99.5139%，训练-测试差距分别30.9028/30.1389/30.7639/30.5556个百分点。
原始证据`runs/simulation/overfitting_audit_20260911/summary.json`，SHA256
`1985dde0b501fbfdff00814d6a1502516fe54736f56de963ecc89d22b68268d7`；每组附CSV、PNG、PDF及权重/特征SHA。
旧epoch没有checkpoint时不补造干净训练值，图中仅best有干净训练点。绘图命令见COMMAND第11节。

新反过拟合训练同样从69.7917%起点：增强+电子weight decay对照，与额外phase dropout=5%（Router2%）对照，
源码13c4b2bc；不改变推理网络/Top2/alpha>0.4/光路，只在训练使用。每5轮复用本就编码的训练特征
记录匹配live/EMA权重的干净训练Hit@1，测试同样关闭增强和dropout；不额外占用完整Qwen或GPU。
配置、可执行命令和依赖顺序见COMMAND第11节；43项CPU测试通过。新分数待真实训练完成。

## 历史审计说明（保留）

[历史 baseline 方法审计](BASELINE_METHODS.md)保留了旧运行的模型、预处理及评估定义。

本目录集中保存 baseline 及主方法的可复现性证据；2026-09-08 建立入口时尚未复现，2026-09-09 的新一轮结果见上方说明与 [本轮证据](RUN_20260909.md)。
当前任务结构和已有结果见 [任务README](../../README.md)。不得因为存在本文件就声称已复现。

后续每个正式结果需在这里记录：

1. baseline定义，冻结/训练的参数，预处理、输出头与主方法差异。
2. 原始数据版本、train/test清单与SHA256、标签生成方法、指标与选模口径。
3. 源码commit、完整命令、依赖环境、模型及checkpoint的SHA256和获取位置。
4. 固定权重复评与从头重训分别报告；标明样本数、seed、运行ID、结果和误差。
5. 速度/能耗的硬件、计时边界、功率积分口径；未测的不得填估计值冒充实测。

原始日志及逐样本结果留在本任务runs，文档只引用。参考 [SALICON复现说明](../../../t03_saliency/reports/reproduction/README.md)。
# 目标相关扩充协议（2026-09-11）

当前新增试验入口为COMMAND第12节及`standalone/domain_expansion.json`，三组共享同一配置，仅数据域顺序不同。
初始best来自`generalization_20260911/preserve_adam/artifacts/best.pt`，SHA256
`c8509b44fbc0f7bcd6e1f0507376b6964bf483fca8205308407b790a295a100f`，固定测试69.7917%。
旧target manifest SHA256为`2949a4035150a9f8718f2a6cace164c17394613d24fb9d0234c553bee8d77c97`。
pool实际SHA和候选排除统计写在pool/report.json，每组execution.json复制池合同、源码commit、命令和环境。
原spin视图与新增listing图片属于不同数据域，不宣称新增图片也是12视角spin数据。
不合并原val、不改类别、不删除难例、不增加评估候选；原baseline与现有包保留。
预训练顺序候选实际是高alpha已训练模型先做10轮外域适配，再30轮混合；对照为40轮直接混合及40轮原数据。
三组周期评估同一原图库/测试集，best包含起点。所有新指标须等真实训练报告，不能用起点69.79%冒充提升。
