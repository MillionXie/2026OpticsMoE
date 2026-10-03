# SALICON：不改结构的训练方法优化

## 已完成起点与目标

服务器仓库：`/DATA/DATA1/guest3/2026OpticsMoE`。
任务目录：`LightGenV2/tasks/t03_saliency`。以下路径均相对仓库根目录。

| 已完成run（runs/simulation/下） | 最佳CC | epoch | 两层alpha |
|---|---:|---:|---|
| moe_staged_alpha_free_seed42 | 0.8521266821 | 80 | 0.369679 / 0.409575 |
| moe_staged_alpha_ge040_seed42 | 0.8513403416 | 95 | 0.435896 / 0.442169 |
| qwen_aligned_head_staged_seed42 | 0.8896847576 | 80 | 不适用 |

证据是每个run的`selected_checkpoint_test_evaluation.json`及`metrics/training_history.csv`。
三组源码为49157bf567c995d766ad62a58c2f1ac0ca745b89。冻结Qwen新头baseline已完成，
不是零样本模型；同规格适配器197184参数、解码头85412参数，不代表整体网络参数/历史预算一致。
本轮希望缩小0.03834 CC差距，不预先声称能够追平。

## 训练学习率问题与修复边界

旧的`vision2_hybrid_dense.settings.apply_vision2_hybrid_settings`在读完training后，
用optimization覆盖了电子/相位/router LR。因此上一轮实际基础LR为1e-4/1e-4/5e-5。
本轮T03配置显式设置`training.learning_rate_source: task_training`，以training为最终值。
旧配置默认仍为legacy_optimization，保障原结果可复现，也不影响其他任务。
运行的`resolved_config.yaml`新增`effective_optimizer_learning_rates`；每轮history记录真正的LR。

## 四组配置（只改训练）

四组均从已完成ge040的best起步，保持原始alpha，不重置gate；新建优化器而非精确恢复optimizer。
source：`LightGenV2/tasks/t03_saliency/runs/simulation/moe_staged_alpha_ge040_seed42/best_checkpoint.pt`。
SHA256：`f3c95ef24a7bcf56674815cee87bd6f1157394969a4101edfa6cdd983903de31`，加载前强制检查。
请勿在同一个源run重新训练覆盖该文件。

| config后缀 | 相位基础LR | CC权重 | 最小裁剪面积比例 | 亮度/对比度jitter |
|---|---:|---:|---:|---:|
| control | 1e-4 | 0.5 | 0.90 | 0.10 |
| reheat | 3e-3 | 0.5 | 0.90 | 0.10 |
| cc | 3e-3 | 1.5 | 0.90 | 0.10 |
| weakaug | 3e-3 | 1.5 | 0.98 | 0.03 |

相邻两组只有表中对应训练因素改变。其余相同：电子3e-5、router2e-4、CCD读出5e-5、
头1e-4；100epoch，前5epoch固定电子/alpha，6–70联合余弦退火，71–100低LR精修。
硬均衡保持0.1，软均衡0.08；KL1.0/SIM0.25/NSS0.1保持。无教师蒸馏。
不是把control称为历史run的完全复刻，而是四组共同续训协议的低相位LR对照。

严格不变：冻结Qwen前端、不执行原生Transformer/attention；两层光电主体、同尺度融合、
alpha≥0.4、光router四专家Top2、478有效区、224专家、17um/10cm、mean_only CCD。
20%–30%相干零级分量、原读出噪声、相位dropout保留；像素位移仍为0。
无新读出头/分支、无推理集成或TTA、无测试图逐图强度拟合。

## 运行顺序

1. 获取GitHub已发布源码，确认工作树和配置。使用xml环境，准备既有SALICON数据、冻结Qwen本地快照及上述source。
2. 检查空闲GPU；每个命令使用一张空闲卡，不停止其他任务。以下GPU编号只是本次启动规划，换机器需重新确认。
3. 从仓库根目录运行，保持日志和run一一对应：

```bash
conda activate xml
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
TASK=LightGenV2/tasks/t03_saliency
mkdir -p "$TASK/runs/simulation/refine_alpha40_20260908"
CUDA_VISIBLE_DEVICES=0 nohup python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_refine_control.yaml" --phase all > "$TASK/runs/simulation/refine_alpha40_20260908/control.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_refine_reheat.yaml" --phase all > "$TASK/runs/simulation/refine_alpha40_20260908/reheat.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_refine_cc.yaml" --phase all > "$TASK/runs/simulation/refine_alpha40_20260908/cc.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_refine_weakaug.yaml" --phase all > "$TASK/runs/simulation/refine_alpha40_20260908/weakaug.log" 2>&1 &
```

不要重复运行到已有输出目录。只有best/last两份模型权重；初始epoch0也参与best保存，确保不丢失原候选。
产物目录为`runs/simulation/moe_alpha40_refine_<后缀>_seed42`。
最终JSON记录CC/KLD/SIM/NSS/AUC/MAE、alpha、专家份额以及相对起点的相位变化
（sigmoid相位转弧度后的圆周RMS，超过0.01rad的像素比例）。相位移动不等于性能必然变好。

## 评价与复现

SALICON官方train2014 10000训练、val2014 5000作public test；无独立validation。
每5epoch及首末epoch按最高public-test CC选模，需披露单seed与选模偏差。
标签、224输出、源图sigma19、逐图CC平均的口径不变，详见本目录README。
run_manifest记录命令和Git SHA，environment记录环境，initialization_report记录source SHA。
不能用仅启动的run更新论文成绩。正式保留候选还需检查专家没有明显坍缩，以及实测鲁棒性。

## 本轮执行记录

实现提交c95e01da；13项T03测试在服务器CPU全部通过。实际启动版本为包含它的
a9cadb1318dfae294521838807f257a584e2fe00（另一任务T04独立提交），已发布GitHub。
control/reheat/cc/weakaug分别使用物理GPU0/2/5/6；0和6与其他小显存任务共享，未终止其他进程。

## 2026-09-09最终复评与下一轮

上一轮四组均完成100epoch，5000张选定权重复评：

| run后缀 | CC | best epoch |
|---|---:|---:|
| refine_control_seed42 | 0.8526296228 | 60 |
| refine_reheat_seed42 | 0.8524844288 | 20 |
| refine_cc_seed42 | 0.8528320684 | 30 |
| refine_weakaug_seed42 | 0.8546876516 | 20 |

完整run前缀为`moe_alpha40_`。证据均在各run的selected_checkpoint_test_evaluation.json。
weakaug alpha=0.434479/0.441579；专家份额22.74/27.10/22.44/27.72%，没有明显全局坍缩。
专家相位相对起点圆周RMS=0.255/0.259/0.452/0.452rad，确有变化。
但weakaug末轮train CC=0.87488、test CC=0.85262，低于epoch20，延长训练没有继续提升泛化。
距同规格头Qwen0.88968476仍差0.0349971，不能声称追平。

### 新协议：正则化及温和退出的蒸馏

固定起点`moe_alpha40_refine_weakaug_seed42/best_checkpoint.pt`，
SHA `de477b8c13c46c50cb9f17eb0b62bc577aaefef5512887e1e17a86d0affb5eea`。
四组配置前缀`moe_alpha40_generalize_`：

- regularized：EMA=0.995/step，电子/读出/头AdamW decay=0.01，相位/router decay=0；裁剪0.95、jitter0.05。
- aligned：同上，但关闭图像增强。它是缓存蒸馏所需的严格像素对齐对照。
- kd020：aligned + 教师密度KL权重0.2，从epoch1线性衰减，到epoch50为0。
- kd060：同上，初始蒸馏系数0.6；epoch50–100均为0，避免教师长期限制学生。

真实标签损失仍为KL1.0 + CC1.5 + SIM0.25 - NSS0.1；光学正则/均衡不变。
全组100epoch；相位LR降到5e-4，电子3e-5、router2e-4、读出5e-5、头1e-4；
不再重复固定电子warmup，1–60联合退火，61–100精修。
EMA只平均同一个网络的权重，best保存EMA权重，非多模型输出集成；推理参数量和光场次数不变。
last保存live权重和EMA状态；它的test_metrics对应EMA而非live，字段已显式区分。
history的train/alpha是live训练权重，test是EMA；正式alpha/router值以best复评为准。

蒸馏教师为新同规格头冻结Qwen，权重SHA
`531c4a330fc05e2c27b037784588716d248a29d1ab8da951d443165dcce552f8`。
只生成train2014的10000张无增强预测，不生成/读取test教师图。
缓存为单个约1.0GB float16 PT及manifest，按完整sample_id严格检查，禁止将未同步变换的缓存用于增强图像。
教师本身曾以public-test选模，所以此方案仍有已披露的测试选模偏差，不称为独立盲测。
教师网络仅用于一次离线训练缓存，最终光电推理没有教师/attention/Transformer/新支路。
所有方案仍有alpha≥0.4、光Router Top2、同尺度融合、20%–30%零级分量和既有读出噪声，ROI不变。

### 新命令（仓库根目录，先确认GPU空闲）

```bash
conda activate xml
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
TASK=LightGenV2/tasks/t03_saliency
mkdir -p "$TASK/runs/simulation/generalize_alpha40_20260909"
# 第一步仅生成训练图缓存；完成后再启动KD组。
CUDA_VISIBLE_DEVICES=6 python -u -m LightGenV2.tasks.t03_saliency.export_teacher_maps --config "$TASK/configs/moe_alpha40_generalize_kd020.yaml" --checkpoint "$TASK/runs/simulation/qwen_aligned_head_staged_seed42/best_checkpoint.pt" --output "$TASK/runs/simulation/generalize_alpha40_20260909/teacher_train_logits.pt"
# 两组无教师对照可与缓存生成并行；KD两组须等缓存及manifest都完成。
CUDA_VISIBLE_DEVICES=0 nohup python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_generalize_regularized.yaml" --phase all > "$TASK/runs/simulation/generalize_alpha40_20260909/regularized.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_generalize_aligned.yaml" --phase all > "$TASK/runs/simulation/generalize_alpha40_20260909/aligned.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_generalize_kd020.yaml" --phase all > "$TASK/runs/simulation/generalize_alpha40_20260909/kd020.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_generalize_kd060.yaml" --phase all > "$TASK/runs/simulation/generalize_alpha40_20260909/kd060.log" 2>&1 &
```

移机时除源checkpoint/数据/冻结前端，还需传训练缓存及manifest才能做蒸馏；普通推理不需要缓存。
不覆盖已有run。逐epoch日志、配置、Git SHA、缓存SHA、best/last仍在各run内集中保存。

本轮实现提交`18c6d0a01196bf34ab617eda363bd787b8d53768`，17项T03测试通过后发布GitHub并同步服务器。
已完成10000张train-only教师缓存，SHA256
`a45a90fe1dc029961464304373473d1271594128fc7b7f2ac80775f8638dd60e`。
regularized/aligned/kd020/kd060分别在物理GPU0/1/2/3启动，A100仅用于已经完成的教师缓存生成。
该轮现已完成：regularized/aligned/kd020/kd060的5000张完整public-test CC分别为
0.8546876669 / 0.8569809561 / 0.8572896766 / 0.8581201639。
regularized保留warmstart(epoch0)，其余三个best均在epoch5。
证据为各run的`selected_checkpoint_test_evaluation.json`，不是预测值。

## 2026-09-09：同步增强、感受野及分阶段微调

共同源：`moe_alpha40_generalize_kd060_seed42/best_checkpoint.pt`，
SHA256 `36333e6ba013cac3a2801bce4babcc2fb5cd36adf02969e019437b40ceebaffd`。
该权重本轮CC=0.85812016；最终epoch100测试CC反降至0.84992395，因此本轮是30epoch短周期精调，
不继续简单延长100epoch。源权重属于EMA选定权重，不恢复旧优化器动量；本轮重新建立EMA。

四组配置/run前缀均为`moe_alpha40_rfstage_`，run后缀为`_seed42`：

| 后缀 | 唯一试验变量（相对于control） |
|---|---|
| control | 共同对照：不增强，3×3卷积，前20epoch联合退火、21–30精修 |
| flip | 50%概率水平翻转，同步变换图像、GT density、fixation、教师logits |
| kernel5 | 两级电子深度卷积改5×5；旧3×3居中、外圈补0，其余张量严格加载 |
| staged | epoch1–5冻结electronic组的梯度及更新，6–20联合退火，21–30精修 |

共同优化器：电子LR=1e-5、相位2e-4、Router5e-5、CCD读出2e-5、显著性头3e-5。
EMA=.995/step；电子/读出/头weight decay=.01，相位/Router为0。
GT损失保持KL1+CC1.5+SIM.25−NSS.1，train-only教师KL系数全程为.2。
除flip外关闭增强；本轮不裁剪、不改亮度/对比度、不做测试时增强或多模型集成。
batch=48，test batch=64，workers=4；每组重新评估warmstart，epoch1及每2epoch完整测试。
测试集仍为官方val2014的5000张，不设独立验证集，明确属于public-test选模；不宣称盲测。
只保留best和last，保留源best，不覆盖历史run。

### 增强对齐边界

flip是主进程中的train-only操作，发生于无增强加载之后、Qwen预处理之前。
以同一sample ID及同一随机翻转标志同步翻转三张目标图；翻转保持GT密度总和。
教师图是原图教师预测的同步镜像，属于等变性训练约束，**不是宣称Qwen对翻转严格等变**。
禁止未同步的随机裁剪/亮度扰动与此缓存混用；test loader始终不增强。
教师仍只需既有约1GB train缓存，推理不引入教师模型。

### 感受野及冻结边界

kernel5保持电子宽度192、MLP192→384→192不变，只将两个depthwise卷积核3→5。
增加`2×192×(25−9)=6144`参数，无attention/Transformer/新分支，光学mask/ROI不变。
新权重architecture以`_ek5`结尾；旧源必须显式启用`expand_kernel_on_warmstart`，
只允许两个指定卷积核中心补零，其他shape/key差异拒绝加载。
新外圈初始为0但可训练，应检查实际梯度和训练后变化。

staged的electronic组包括公共输入投影、振幅编码、两级电子残差、融合系数及相应norm；
前5epoch这些参数`requires_grad=False`，不会暗中积累Adam动量。光学相位、Router、CCD读出和
最终显著性头继续训练，第6epoch重新启用电子组。alpha始终处于原有≥.4范围。

### 操作命令（仓库根目录）

先检查GPU剩余显存/利用率，不停止其他任务。以下GPU编号只是示例，实际以资源检查为准。
如主工作树有其他任务未提交改动，用GitHub提交建立独立worktree，不覆盖源码；
该worktree的数据、缓存和T03 runs通过明确symlink指向原仓库。`cache/qwen`需指向实际HF缓存。

```bash
conda activate xml
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
TASK=LightGenV2/tasks/t03_saliency
python -m pytest "$TASK/tests" -q
# 逐个在各自终端运行，或nohup后台运行；stdout写到对应run/train.log。
CUDA_VISIBLE_DEVICES=0 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_rfstage_control.yaml" --phase all
CUDA_VISIBLE_DEVICES=3 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_rfstage_flip.yaml" --phase all
CUDA_VISIBLE_DEVICES=4 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_rfstage_kernel5.yaml" --phase all
CUDA_VISIBLE_DEVICES=5 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_rfstage_staged.yaml" --phase all
```

完成后首先看各run的`selected_checkpoint_test_evaluation.json`、`training_report.json`、
`metrics/training_history.csv`及`best_visualization/`。报告CC、KLD/SIM/NSS、alpha、专家负载和相位变化，
不把训练CC当测试结果。本节是实验协议，不是已取得提升的结果声明。

实现`21fc0c14`已发布GitHub；服务器CPU回归22项通过。control/flip/staged在物理GPU0/3/5
启动，工作树固定为`2026OpticsMoE/.worktrees/t03_rfstage`，源仓库其他未提交任务未被修改。
三组完整warmstart复评均约0.85812008。
kernel5首次启动发现旧电子模块使用显式F.pad，新卷积又内置padding导致尺寸错误；
没有完成训练或产生best。已改为更新模块的kernel_size且卷积padding=0，并将测试改为经过
真实ElectronicResidualMLPBlock的完整前向/反向，而非孤立Conv2d测试。
失败日志保留在`moe_alpha40_rfstage_kernel5_seed42/train.log`，修正后的正式run明确为
`moe_alpha40_rfstage_kernel5_paddingfix_seed42`，不覆盖失败记录；配置文件名不变。
修复提交`dea0f494`已发布GitHub，服务器22项测试重新通过；kernel5使用独立固定工作树
`2026OpticsMoE/.worktrees/t03_rfstage_ek5`在GPU4重新启动，并完成5000张warmstart复评，CC仍约0.858120。
源码留在固定worktree，数据/缓存共享，所有正式run仍写回原仓库T03的runs/simulation。
旧3×3训练工作树不在运行中切换提交；此处两个worktree的功能差异仅为kernel5补边修复及其测试/说明。
