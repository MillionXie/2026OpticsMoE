# T19：九项 ResNet-20 小模型 baseline

## 2026-10-23 接续七项，LGVQ优先

ABO双向各30轮已完成，两个方向均选第25轮固定复评：H@1分别96.6667%和95.0000%；
第30轮分别62.7083%和80.0000%。两者独立列明，均为TEST选模的开发指标，不隐藏末轮波动。
来源：`easy100_single30_shared_20261023`及其两个formal run；自有GPU进程释放检查通过。

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
