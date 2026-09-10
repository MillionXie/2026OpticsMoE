# 额外无标签图像：预训练数据准备

状态（2026-09-10）：**19999张教师目标已导出并通过实际缓存读取器校验；尚无额外图像训练的CC结果**。
原两组60轮已结束；后期来源的40轮对照已提前停止并释放GPU，详见文末停止记录。
较早来源的80轮配对方案准备中，最多两张卡。

## 为什么检查这条路线

在同一10000张SALICON图像上反复续训的已完成best为0.86204960，目标0.88尚未达到。
可以尝试将现有教师对额外未标注图像的输出用于预训练，再回原SALICON真值微调，
让原光电网络接触更多视觉变化，而不增加推理网络/参数。
依据是[Noisy Student，CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.html)
利用教师伪标签与额外无标签数据、学生训练扰动的思路。该论文是ImageNet分类，
这里不是其完整复现，也不能据此承诺显著性CC会提升。具体预训练损失/预算待当前对照后确定。

**公平性必须披露：若采用本路线，学生多用了无标签预训练图像，不能再声称与原baseline
使用完全相同的训练数据预算。**原10000图训练结果与新方案分行保存，不覆盖baseline。
教师只在离线生成目标时运行，不能加入学生推理；所有既有光路、Top2、alpha及轻量结构约束不变。

## 已有服务器数据与重叠风险

服务器已有 `data/COCO2017/train2017` 118287张，未下载或移动任何数据。
只按COCO数值ID核查发现：与SALICON train重叠10000张，**与SALICON public-test重叠4376张**。
因此不能直接拿整个COCO train2017预训练。排除SALICON train与val全部15000个ID后，
剩103911个候选。初步seed17042选择20000张的现有文件共3248018579字节（约3.25GB），
这些是已有磁盘内容，不会复制成另一份数据集。

正式准备器进一步对完整SALICON train/val文件及选中图像计算SHA256：

1. 要求排除目录完整含10000/5000张，拒绝错误/缺失目录和重复数值ID。
2. 排除全部训练、测试数值ID，固定seed从排序候选中抽样，再按ID排序。
3. 排除与训练/测试文件字节完全相同、或选中集合内重复的文件；不会为补齐数量偷偷改变抽样。
4. 保存实际数量、每个原图的SHA与文件名、ID哈希、排除原因和Git commit；不保存额外图像副本。

仅ID初选的20000个ID哈希（还不是内容去重后的正式manifest）：
`beb2014d0029c7cda21543b8964ea8fbb5e864bf2cb5189cd975985f1a3e4624`。
哈希口径是数值排序、十进制ID逐行、末尾换行。不能与旧流程无末尾换行的sample_id哈希直接比较。
上述排除不保证发现不同ID且重新编码的近重复图片；该边界须保留。教师导出时还必须验证图像
可解码、文件SHA与manifest一致，且不能把伪标签当成原始眼动真值。

## CPU准备命令

在服务器仓库已同步、已测试的源码版本中运行；输出必须不存在。该命令不导入torch、不使用GPU，
只顺序读取文件计算哈希；它不是预训练入口。

```bash
python -m LightGenV2.tasks.t03_saliency.prepare_unlabeled_pool \
  --coco-root /DATA/DATA1/guest3/2026OpticsMoE/data/COCO2017/train2017 \
  --salicon-root /DATA/DATA1/guest3/2026OpticsMoE/data/SALICON \
  --count 20000 --seed 17042 \
  --output /DATA/DATA1/guest3/2026OpticsMoE/cache/qwen3_vl_embedding_2b_salicon_lightgen/coco20k_pretrain_20260910/image_manifest.json
```

后续教师预测、光电预训练和真值微调应分别有身份清单；只在空出当前两张卡之一后执行。
预训练仅使用上述排除后的额外图像；正式微调仍只使用10000张SALICON train，5000张public-test
仅按既有已披露的公开测试选模口径评估。完成标准独立复评之前不得更新正式成绩。

## CPU准备已完成

代码 `051a9da53a7580d6d8fc823e8d8af95faa8152cd`：本地准备器3项测试通过，服务器T03全套
115项通过（26.20秒，既有matplotlib弃用警告13项），先push GitHub再运行实际数据准备。
2026-09-10 13:21 CST完成；20,000张ID候选又发现1张内容重复（COCO ID410810），
仅从候选清单排除，**没有删除原图**。最终19999张，现有原图3247811485字节。

正式清单就是上方命令的 `image_manifest.json`：

- manifest SHA256：`d0600cf3a4eb8e1bd0d6dc415a237c2c5fe6ce88ca3a77385d26243d44874182`
- 最终19999个ID SHA256：`1ab59f56a3304dca39829e5b14e59c35a218128d303baf480dc15de1a987d60a`
- 排除用train ID SHA256：`15e2eca8377f91111cfc35148306c6c11ba7ac99e6fcdc1359eee3b1170f2f96`
- 排除用test ID SHA256：`643e536bcfb4ba21c77f1be6a13fdf374b1e43499223044117a6ad63e61dce80`

全部采用本页带末尾换行的数值ID口径。CPU准备进程正常退出，未使用第三张GPU。
这证明图像池与当前训练/测试ID、文件字节的隔离，不证明预训练已执行或提升性能。

## 教师目标导出（待空出GPU后执行）

`export_unlabeled_teacher`与`unlabeled_data`是额外图像专用入口，**不修改**原
`TrainTeacherMaps`只能读取有序SALICON train身份的限制。再次核对完整排除集合的ID哈希，
对每张选中图片将同一份字节先做SHA检查再解码，RGB转224×224 BICUBIC，与原无增强预处理一致。
导出时运行已完成的冻结Qwen24＋同规格头教师；这不属于学生推理结构。
原始RGB、人工真值、原SALICON缓存均不改写。

```bash
# 仅在本助手已有训练空出一张卡后运行；GPU0只是预定编号，需先查实时占用。
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m LightGenV2.tasks.t03_saliency.export_unlabeled_teacher \
  --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_sam_spatialcc_kd2.yaml \
  --checkpoint LightGenV2/tasks/t03_saliency/runs/simulation/qwen_aligned_head_staged_seed42/best_checkpoint.pt \
  --image-manifest /DATA/DATA1/guest3/2026OpticsMoE/cache/qwen3_vl_embedding_2b_salicon_lightgen/coco20k_pretrain_20260910/image_manifest.json \
  --image-manifest-sha256 d0600cf3a4eb8e1bd0d6dc415a237c2c5fe6ce88ca3a77385d26243d44874182 \
  --batch-size 16 \
  --output /DATA/DATA1/guest3/2026OpticsMoE/cache/qwen3_vl_embedding_2b_salicon_lightgen/coco20k_pretrain_20260910/teacher_logits.pt
```

目标缓存约2.01GB（19999×1×224×224个FP16 logits，另有小型身份元数据）。无logits非有限值或
FP16溢出才发布完整缓存；已有PT/partial/JSON时拒绝覆盖。失败partial保留供检查，不当作完整数据使用。
完成后JSON记录缓存SHA、教师SHA、图像清单SHA、源码版本、预处理、图像数量和
`ground_truth_available=false`。每个ID显式命名为`unlabeled/coco2017/xxxxxxxxxxxx`。
读取器验证缓存SHA、教师来源、清单、ID顺序、形状/dtype和数值，再按ID取值；
不能把它直接塞入旧的SALICON train缓存入口。尚需独立的无标签预训练流程，不能用伪fixation冒充眼动标签。

### 导出/读取入口的验证状态

源码 `61b1de32cb6184db33d5792014508fae3eb6cf9e`：服务器CPU全套T03测试128项通过
（26.59秒，13项既有matplotlib弃用警告），已确认该commit在GitHub分支历史中。
真实图像清单的19999张全部逐项核对字节SHA、解码、RGB及224×224预处理与ID顺序，
117.57秒完成，CPU检查进程正常退出。预定教师权重也已实查SHA：
`531c4a330fc05e2c27b037784588716d248a29d1ab8da951d443165dcce552f8`。
这些不是GPU教师前向验证：**完整教师导出仍待执行，尚无该额外图像缓存或预训练结果**。

## 下一轮：保留真值的额外图像辅助训练（待缓存完成）

现有同集教师预热对照在恢复GT前出现明显测试下降，恢复GT后尚未超越原best。
因此额外图像路线不再设计成长期关掉GT：每个优化步骤始终有一批SALICON train真值图像，
另取一批排除重叠后的COCO图片，只用缓存教师输出监督。没有人为生成的fixation或NSS真值。

记原监督任务损失为 `S=GT_KL+1.5*(1-GT_CC)+.25*(1-GT_SIM)-.1*GT_NSS+2*KD_labeled`，
额外图像空间CC损失为 `U`，各批次的router均衡/importance/CCD工作点正则为`R_l/R_u`：

`L = S + lambda*U + (R_l+R_u)/2 + phase_DC_regularizer`。

GT系数不随额外batch大小缩小。额外图像系数计划从0.6开始；不是把两种标签拼成一份伪真值。
每次SAM第一/第二次计算都使用相同两批图，共四次学生前向、一次optimizer/EMA更新。
路由损失必须在对应前向后立即保存，不能让额外批次覆盖先前路由状态。
相位、router以及原电子参数均继续训练，没有新增推理/训练模块参数。训练计算量增大，
不能把它与普通两次前向SAM称为相同训练FLOPs；正式推理仍是原单图光电架构。

额外loader独立seed，循环游标跨监督epoch保留，不在每个epoch重复只抽前半池；
完整走完池后才重新shuffle，保留末尾不足batch的图像。日志分别记录监督样本数、
额外样本数与loader重启次数。GT CC/NSS只对有真值的SALICON计算，
额外池只报告教师空间损失，不冒充测试成绩。

`moe_alpha40_extra_control.yaml`固定40轮、从完成best SHA87ad出发、原精修学习率
（E1e-5、phase2e-4、router2e-5、head2e-5、CFFN5e-5）、31轮起小步精修。
对照`unlabeled_distillation.weight=0`精确走旧SAM入口，不改变旧profile。
额外组需在缓存真实导出后写入image manifest及cache的SHA、weight=.6和独立output_dir；
**当前不提供缺失缓存SHA的可运行正式配置，不得跳过校验强行运行**。
新分支已实现为`semisupervised.py`，由原`run.py`按显式配置调用；现阶段不代表已训练或验证有提升。

两个新试验只能在现有两卡作业结束、核查释放后排队启动。需先用真实图像和教师缓存做
短更新验证，再跑完整40轮与5000张public-test选模。若有改善仍需独立复评和光学审计。

### 保留GT分支的实现验证

源码 `2659f0e01922486a2842c18b6d0a2ac7bc610949`，服务器T03全套136项测试通过
（28.27秒，13项既有matplotlib弃用警告），已push GitHub。测试覆盖GT梯度系数仍为1、
两批正则各为1/2、四次SAM前向复用同一对batch、一次优化器更新、循环数据游标及关闭，
并验证不允许同时关掉GT/开启教师单独预热或无效SHA。

另在CPU用真实完整学生与SHA87ad权重做一次短更新检查：SALICON训练记录0作监督批，记录1只提供
另一张图像和已有train教师输出；这是**训练代码冒烟检查，不是正式无标签池训练**，不计性能结果。
没有使用测试图片、没有生成/覆盖checkpoint、没有占用第三张GPU，进程已正常退出。
SAM实际optimizer step=1，loss有限（1.10317647）；结构标签保持原cffn_d1不变，
alpha=0.43072152/0.44106743。router raw参数RMS更新约1.23e-5，四专家约1.83e-4–1.97e-4，
global约1.92e-4。这里是raw参数量，不是弧度；其中包含DC正则梯度，不能据此断言一批数据激活了全部专家。
这证明真实网络上的混合更新可运行；正式额外池教师缓存导出和GPU短验证仍待执行。

## 正式额外缓存已完成

代码 `2659f0e01922486a2842c18b6d0a2ac7bc610949`，按上方完整命令在GPU0导出，
全部19999张成功，导出PID9801正常退出，GPU0显存恢复12MiB。
`teacher_logits.pt`实际2007761116字节（约2.01GB），SHA256：
`232e02d243a3d58b8d5cc48557da8f566f77a81e7020968a8a84c0c85d7c305c`。
配套`teacher_logits.json`保存教师SHA、图像清单SHA、预处理和源代码身份。
随后独立CPU读取器重新计算缓存SHA、检查19999个ID顺序/排除集合、全部FP16有限值及形状，
首末样本按ID取回形状为`[2,1,224,224]`。没有使用测试图像生成这份训练缓存。

正式额外组配置为`moe_alpha40_extra_coco20k.yaml`，继承40轮控制配置，
仅增加`unlabeled_distillation.weight=.6`及上述锁定缓存；两者推理架构完全相同。
GPU1的控制组PID17303，代码2659f0e0；其数据仍只有10000张SALICON train。
下列是两组复现命令，额外组须先完成真实缓存短更新检查后启动；运行目录必须尚未被另一作业使用。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_extra_control.yaml --phase all
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_extra_coco20k.yaml --phase all
```

只用现有两张卡；缓存导出结束才能交接GPU0，不额外占卡。目标.88仍未达到。

### GPU短更新与正式启动（2026-09-10）

源码`9c40515ea1b36ed0f5c855774f11a168f65331cf`：T03全套137项CPU测试通过
（32.68秒，13项既有警告），已确认GitHub分支包含该commit。新增配置测试初次因工作树cache为
符号链接而比较未resolve路径失败；已只修正测试路径比较，未改训练逻辑或放松实际数据校验。

使用该正式配置、原87ad初始化、真实额外缓存，在GPU0做一次完整SAM更新：
固定seed42、workers0、诊断batch2，SALICON train前2条为GT批，额外批来自正式独立shuffle池；
其余损失/光学扰动保持正式设置，map KD=2、extra KD=.6。挂钩检查optimizer更新1次、学生前向4次、
24个原生Vision Transformer模块的调用次数为0，并执行EMA更新。
loss=1.01628029，alpha=.43072152/.44106743；router raw RMS更新约1.29e-5，
专家约1.79e-4～1.98e-4，global约1.92e-4。所有指标/更新有限；raw RMS不是弧度，
包含DC正则梯度，不能用它宣称一批激活了全部专家。没有保存checkpoint，没有改变原始best。
诊断PID53989正常退出，GPU0回到12MiB后才启动正式额外组。

- GPU0额外组PID63224：`moe_alpha40_extra_coco20k_seed42`，代码9c40515e。
- GPU1控制组PID17303：`moe_alpha40_extra_control_seed42`，代码2659f0e0。

两源码版本的T03差异仅新额外profile、测试和文档，训练实现相同；各run保存实际commit。
正式两组均batch32、评估batch48、workers2，40轮，源SHA87ad，推理参数不变。
当前只是启动记录，不代表已完成或提升。后续看`metrics/training_history.csv`、
`training_report.json`和重载best的完整5000图评估；有实质提升再独立float64复评。

## 较早权重的额外图像对照（准备中，未启动）

后期87ad来源的额外组第1/5轮CC为.86218145/.86201662，同轮普通控制为
.86204380/.86179060：有减轻下降的迹象，但尚无持续提升；两组仍运行，不是最终结果。
若后期来源持续平台，下一对照使用已有较早的`moe_alpha40_refine_weakaug_seed42`，
best SHA已重新核验为`de477b8c13c46c50cb9f17eb0b62bc577aaefef5512887e1e17a86d0affb5eea`。
这是此前反复精修之前、历史CC约.85468765的来源，不是随机初始化，也不是从其optimizer精确续训。

- `moe_alpha40_extra_early_control.yaml`：80轮、无额外图像。
- `moe_alpha40_extra_early_coco20k.yaml`：同样80轮，额外图像空间CC监督系数.6。

两组相同来源、初始化函数、GT损失、SAM .05、EMA .995、batch32、测试batch48、seed42；
教师监督2恒定，关闭在线增强，始终保留GT。E/phase/router/CCD/head/CFFN学习率为
3e-5/5e-4/5e-5/3e-5/5e-5/2e-4；1–60轮联合训练，61–80轮小步精修，不新增电子模块。
这些学习率沿用此前已验证的较早来源训练量级，不是在运行中的任务上偷偷修改。
源权重早于当前CFFN，因此只对当前网络已有的6912个空间DW参数作恒等初始化；
**相对当前部署候选不增加任何推理参数**。原光相位、alpha和解码头严格加载，不重置。
需要真实网络验证源模型与初始化后模型在eval下输出一致，并检查新增恒等参数能更新。

这是同一较早来源内部的配对对照，不能只与后期40轮比较就把差异归因于数据；
额外组仍多用19999张图和约两倍学生训练前向计算，公平性披露不变。

```bash
# 仅在原作业结束或明确记录停止、核查显存释放后，由空出的两张卡顺序启动。
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_extra_early_control.yaml --phase all
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_extra_early_coco20k.yaml --phase all
```

本节为待验证/排队方案，不代表已启动80轮，不得占用第三张卡。只保留best/last，正式结论仍需完整测试复评。

### 较早来源验证

代码`2cae020507babb6f7e37578d38e62106ba06df76`的全套138项T03 CPU测试通过
（27.97秒，13项既有警告），已确认该commit在GitHub分支历史中。
CPU真实网络、两张训练图检查：将de477源权重加载到旧无CFFN网络与当前恒等CFFN网络，
eval输出最大绝对误差为0。再用2张真实训练图＋2张实际额外池图像做一次SAM更新，
loss=1.03229702，两个现有CFFN卷积RMS更新均约1.999e-4；alpha=.43447757/.44157824。
只是初始化/梯度诊断，不是正式测试精度；没有保存或覆盖checkpoint，没有使用GPU。

## 后期40轮试验提前停止与资源清理

2026-09-10 16:12 CST：额外组第10轮CC=.86172154，同轮普通控制=.86147447；
控制第15/20轮进一步降至.86127592/.86110137。额外组best仍第1轮.86218145，
没有持续逼近.88的迹象，因此结束该后期来源路线，转向前述较早来源试验。
这属于根据公开测试表现进行的人工调整，具有选择偏差。

- 控制PID17303：最后完整epoch24，best epoch0（源权重）；第25轮中止。
- 额外PID63224：最后完整epoch10，best epoch1；第11轮中止。

两者均**未完成预定40轮**，没有`training_report.json`；不能伪称正常完成。
只对已核验命令/工作目录的两个自有PID发送SIGTERM。best/last文件均保留并已CPU重载成功，
没有删除run或原图。此处训练中保存的best指标尚未独立float64复评，不替换正式87ad候选。

停止后文件SHA256（best / last）：

- 控制：`695a6b21e3876a71a9fe53cb805ae285986e9838e624f180628ae8b08bab48ab`
  / `0203f0ff7c39701577a31fc25405717a792a7d3a5f8ebf2152f2b5687530f5ac`。
- 额外：`8684126c2b02d6cf4f4185c1b78be85f7f61ae466a34fbd78b81042f61f362ce`
  / `ae23e226c2feedfe54416ee28d1dc38ea53b159262d028893b0888ba7443eb0f`。

清理发现额外加载器的两个fork子进程252373/252374在父进程结束后仍存活（PPID=1，
命令和cwd均属于该额外run），GPU0仍显示原父PID约7298MiB。核对后只终止这两个自有子进程，
随后所有上述PID消失，GPU0/1回到12/25MiB。没有终止其他任务，没有reset显卡。

为后续运行改用额外DataLoader的`spawn`上下文（workers=0时不设置多进程上下文），
避免在学生已初始化CUDA之后fork继承驱动资源；不改变数据顺序种子、张量处理或推理结构。
[PyTorch 2.6多进程说明](https://docs.pytorch.org/docs/2.6/notes/multiprocessing.html)
提醒CUDA运行时不支持fork方式启动CUDA子进程，并提示fork后台线程风险；本次资源残留是现场观测，
不把文档泛化为所有DataLoader都一定泄漏。新增测试验证实际spawn取样及正常close后的worker退出。
即使使用spawn，异常终止后仍须核查自有CPU子进程与显存，不能承诺所有信号下自动清理。
