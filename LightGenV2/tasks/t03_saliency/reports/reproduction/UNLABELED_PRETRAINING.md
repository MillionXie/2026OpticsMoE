# 额外无标签图像：预训练数据准备

状态：只准备、审计图像池，**尚未生成教师预测、尚未训练、没有新CC结果**。
当前两张GPU仍用于教师预热/联合续训对照，不能因本方案占用第三张卡。

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
