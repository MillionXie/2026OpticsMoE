# T07 独立工程：日常操作顺序

以下从本任务文件夹或解压后的工程根目录执行。**不需要2026OpticsMoE、T01、experiments、完整Qwen权重或网络访问。**
当前支持固定ABO检索指令、一张224×224商品图；不是任意prompt的大模型服务。

## 1. 环境与文件

先激活已有Python环境，不创建/覆盖conda环境。根据本机显卡先安装适配的CUDA版PyTorch，再执行：

```powershell
python -m pip install -r requirements.txt
python run.py verify --assets assets
```

`assets`包含best.pt、processor、train_targets.pt和SHA256清单。完整包另有`data`，内含
`data/abo_similarity10_manifest.csv`以及它引用的图像；若数据已在其他位置，后续仅修改`--data`，不移动原数据。
不要修改assets内容；图像尺寸、prompt、朝向和相位物理参数不通过猜测更改。

## 2. 固定best：完整评估与去光对照

```powershell
python run.py evaluate --assets assets --data data --device cuda --batch-size 4 --output runs/simulation/evaluate_01
```

全量1440训练视角重建120个图库中心，480测试图全部查询；然后同权重去光复评。
输出`final_report.json`、逐图CSV、`retrieval_features.pt`、`phase_masks.png`。
`--output`必须是新目录，不覆盖已有结果；CPU调试可改`--device cpu`，但全量会慢。
这不是硬件采集命令；新版核心保留实测CCD注入接口，但本次包不声称已含设备SDK/自动播放流程。

## 3. 可选：独立微调

```powershell
python run.py finetune --assets assets --data data --device cuda --epochs 30 --steps 48 --batch-size 4 --output runs/simulation/finetune_01
```

`--batch-size`是评估batch；训练固定10类×2个不同商品=20。`--steps`是每epoch训练步数。
使用包内**训练集**教师目标，不加载完整Qwen；训练光学相位、Router、电子残差及读出，前端冻结。
保存best.pt和last.pt，EMA=0.99，每5epoch完整test选best；这是test-selected而非独立无偏估计。
新微调为fresh optimizer继续训练，不承诺与历史训练随机轨迹逐步相同。历史70%是固定best复现目标，
不是承诺任意微调都提升。先保留原assets；别用微调last直接替换交付best。

## 4. GPU使用规则

默认一个进程只用一张GPU，无DDP/多卡自动扩张。服务器先查看`nvidia-smi`，用UUID选择空闲卡：

```bash
CUDA_VISIBLE_DEVICES=空闲GPU的UUID python run.py evaluate --assets assets --data data --output runs/simulation/evaluate_02
```

日常只用一张；同一操作者最多两张，勿占用他人GPU进程。命令完成或Ctrl+C会退出进程，释放CUDA上下文；
结束后用`nvidia-smi`确认自己的PID消失。不要为清理显存杀别人的进程。

## 5. 可选：任务内教师预热 + 联合训练（新试验，不替换正式best）

需最新 Git 代码（旧 ZIP 未包含此新增 profile）：

```powershell
python run.py finetune --profile teacher_curriculum --assets assets --data data --device cuda --epochs 30 --steps 48 --batch-size 4 --output runs/simulation/teacher_curriculum_01
```

训练 batch=40（10类×4个不同商品）；`--batch-size 4`仍然只控制评估。默认30epoch包含4epoch预热、
22epoch光电联合、4epoch无蒸馏收尾。前端始终冻结，不添加TF/attention，不加载大模型。
仅使用原训练集教师缓存，并非新增外部数据集预训练。配置在`standalone/curriculum.json`。
alpha固定为输入best的四个实际系数（约0.087～0.104，不是0.4）；预热只更新电子，随后相位也更新。
联合阶段75%的batch保留原未调制/CCD噪声；收尾25%，其余为干净仿真。推理图和原评估口径不变。
每轮记录全部batch的专家选择比例、相位/电子参数更新、alpha及独立训练样本数。
开始先评估/保存epoch0为保底best，之后用EMA候选test选模；只保存best.pt、last.pt，不保证优化一定提升。
收尾关闭的是逐样本特征蒸馏和关系KL；CE分类锚点仍来自训练教师缓存，不是完全无教师训练。

## 6. 较大 ABO 子集预训练，再迁移到当前10类

这才是增加商品/类型覆盖的预训练；不同于第5节在同一小训练集上蒸馏。
不改变部署结构：仍是固定Qwen前端、V/L光Router Top-2和原光电网络。预训练辅助类别头只算训练loss，
检索/推理完全不用；不加载完整Qwen，也不使用教师缓存。alpha固定为当前best，不进一步降低。

服务器数据路径可按实际修改。先准备一次清单（CPU，不复制原始图片）：

```bash
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.prepare_broad_abo \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --output LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/broad_pool_20260910 \
  --categories 128 --products-per-category 48
```

每类型最多48商品、每商品2图；实际保留数量以report.json为准，不是保证有128类。
排除全部目标200商品（含train/val/test）、共享image ID、文件SHA及128位dHash近重复。
近重复筛查有误差，不能说已证明语义相近的所有商品变体完全独立；目标test不用于预训练选模。

清单检查后，选一张空闲GPU串行完成预训练和迁移：

```bash
CUDA_VISIBLE_DEVICES=空闲GPU的UUID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.broad_transfer \
  --mode chain \
  --assets LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/standalone_assets_20260910 \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/broad_pool_20260910 \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --output LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/broad_transfer_20260910/artifacts \
  --pretrain-epochs 20 --adapt-epochs 30 --batch-size 4
```

工作目录为源码仓库根目录。`pretrain/`下20epoch×120steps、训练batch32；`adapt/`下30epoch×48steps、
训练batch40。每epoch并不强制覆盖全部图；采样ID数写入history。配置在`standalone/broad_transfer.json`。
全程相位、Router、电子残差、读出均更新，Qwen前端/alpha冻结；监督对比+训练用类别分类，EMA。
预训练50% batch、目标微调25% batch保留原噪声/20%～30%未调制分量，其余干净；光路/ROI不变。
预训练以训练loss选择EMA快照（并非外部验证最优）；目标微调按现有test口径选best，并包含原70.21%保底。
每阶段只留best.pt/last.pt；目标最后全量正常/去光复评输出final_report、逐图CSV和相位预览。
75%是优化目标，不是本命令已取得的结果；原ZIP/best不会被覆盖。

## 7. 严格alpha>0.4，分阶段训练与增强

从仓库根目录执行；先检查`nvidia-smi`，仅选一张有足够余量的卡，不停止其他人的进程。
下面显式指定训练用GPU UUID；示例UUID需按实际检查结果设置。

```bash
export CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.broad_transfer \
  --mode adapt --profile high_alpha \
  --assets LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/standalone_assets_20260910 \
  --checkpoint LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/standalone_assets_20260910/best.pt \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --output LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/high_alpha_aug_20260910/artifacts \
  --adapt-epochs 60 --batch-size 4
```

`--batch-size 4`是评估batch，训练batch由`high_alpha.json`定义为40。
冒烟检查请用独立`runs/smoke/`输出，并追加`--adapt-epochs 6 --steps 1`，覆盖冻结和联合两个阶段。
已有输出目录会报错，不能覆盖旧结果。源权重低alpha只用于热启动，转换后alpha初始0.45。
前5epoch优先光学学习，后续联合；EMA每5epoch测试选优，测试集参与选模必须如实披露。
`execution.json`记录源码/配置/初始权重/数据哈希；`history.json`包含alpha和专家选择比例；
`parameter_updates.json`记录参数变化；`final_report.json`包含正常/去光/相位打乱/轻噪声和圆周相位变化。
所有训练结束或失败后检查执行日志中的PID已退出，及`nvidia-smi`中该PID已释放。

## 8. 从67.92%高alpha权重继续：跨商品图库检索目标

仍在仓库根目录执行，先检查GPU；默认只使用一张。以下示例选择GPU1，确认余量后使用。
不需要大ABO预训练池或完整Qwen，不改变正式测试图库。

```bash
CUDA_VISIBLE_DEVICES=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.broad_transfer \
  --mode adapt --profile high_alpha_retrieval \
  --assets LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/standalone_assets_20260910 \
  --checkpoint LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/high_alpha_aug_20260910/artifacts/best.pt \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --output LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/high_alpha_retrieval_20260910/artifacts \
  --adapt-epochs 30 --batch-size 4
```

训练batch40、64steps/epoch，每轮额外刷新仅训练商品的图库；评估batch4。
代码拒绝低alpha源checkpoint；不重置已训练的高alpha。前25轮联合、末5轮仅读出头。
每5轮分别保存live/EMA的测试指标，只有选中的最好权重写best.pt（不额外保存EMA文件）。
高alpha的67.92%起点保留在候选中，若续训没有提高必须如实说明，不能将保底成绩说成新提升。
快速检查使用独立`runs/smoke/`目录、`--adapt-epochs 1 --steps 2`，会覆盖图库构建、反传和两种选模。
输出指标/去光/相位打乱/噪声/相位变化文件与第7节一致；保留全部旧结果，不覆写旧ZIP。

## 9. 完整商品输入 + SAM + 全场语言CCD读出：单卡串行

从仓库根执行，先检查GPU。以下GPU1 UUID是当前服务器已检查的卡，其他服务器必须替换。
源权重固定为68.9583%高alpha版本，不需要完整Qwen或原ABO大预训练池。

```bash
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d \
  --assets /DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/standalone_assets_20260910 \
  --checkpoint /DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/high_alpha_retrieval_20260910/artifacts/best.pt \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --output /DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/generalization_20260911 \
  --epochs 30 --steps 64
```

按 `preserve_sam → preserve_adam → preserve_fullfield_sam` 顺序执行，优先检查SAM是否有效；参数在 `standalone/generalization.json`，
继承`high_alpha.json`和`retrieval_training.json`。学习率、增强、采样、数据、源权重都匹配，SAM增量rho0.03。
每组30轮，SAM同step需两次反传，**不是等GPU时间对照**。状态统一看output下status.json；各组console.log、
artifacts/history.json、final_report.json保留完整数值，失败队列停止，不会一直启动失败任务。

快速实跑检查使用另一个 `runs/smoke/generalization_20260911` 输出，并加：
`--profiles preserve_fullfield_sam --epochs 1 --steps 2`。
此检查会全量编码/评估，但只更新两步；不把冒烟的准确率当最终优化成绩。
每组进程结束才启动下一组；若卡被其他人占用，队列停止并记录原因，不杀别人的进程。

## 10. V/L都保留完整CCD：接在现有队列之后

以下是额外单组，不修改第9节已运行的三组；源权重仍是原68.9583%版本，不接上一组训出的best。
从包含本profile的仓库根运行；`--after-queue` 必须指向已存在的第9节status.json。
等待期间仅CPU轮询；前队列失败则本组也停止，前队列完成且GPU空闲才开始，不会抢占他人GPU。
不需要等待时可省略 `--after-queue`，但不得同时重复启动同一组。

```bash
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d \
  --assets /DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/standalone_assets_20260910 \
  --checkpoint /DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/high_alpha_retrieval_20260910/artifacts/best.pt \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --output /DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/generalization_fullfield_both_20260911 \
  --profiles preserve_fullfield_both_sam --epochs 30 --steps 64 \
  --after-queue /DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/generalization_20260911/status.json
```

输出仍为status.json及profile下console.log/artifacts，只有best/last；测试、相位和去光分析同第9节。
新配置V全场pool196×224、L全场pool77×224，完整强度图参与汇聚，输出仍为196×192和77×192。
35项CPU测试覆盖旧模式不变、V/L完整场读出、SAM恢复及排队依赖；真实训练成绩仍待完成。

## 11. 训练准确率曲线、完整物体增强与相位dropout

从仓库根执行。下面变量只缩短路径，不移动数据。当前GPU预算为一张，必须先检查占用；不要重复启动。
先做1epoch/2steps检查，正式队列等检查成功后再启动；若还有第10节任务在跑，检查先等它完成。

```bash
T07_RUNS=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval/runs
T07_DATA=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d \
  --assets "$T07_RUNS/simulation/standalone_assets_20260910" \
  --checkpoint "$T07_RUNS/simulation/generalization_20260911/preserve_adam/artifacts/best.pt" \
  --target "$T07_DATA" --output "$T07_RUNS/smoke/antioverfit_20260911" \
  --profiles regularized_phase05 --epochs 1 --steps 2 \
  --after-queue "$T07_RUNS/simulation/generalization_fullfield_both_20260911/status.json"

python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d \
  --assets "$T07_RUNS/simulation/standalone_assets_20260910" \
  --checkpoint "$T07_RUNS/simulation/generalization_20260911/preserve_adam/artifacts/best.pt" \
  --target "$T07_DATA" --output "$T07_RUNS/simulation/antioverfit_20260911" \
  --profiles regularized_phase05 regularized_control --epochs 30 --steps 64 \
  --after-queue "$T07_RUNS/smoke/antioverfit_20260911/status.json"
```

两组都从69.7917%起点独立训练；配置`standalone/regularization.json`。SAM本次不启用，
两组只相差额外相位dropout。已有的光噪声与均衡保留，不改ROI、alpha>0.4、六次光计算或推理头。
实时看各队列`status.json`和各组`artifacts/learning_curves.png`、CSV、history；曲线只连接已测epoch，
不会把未测干净训练集准确率补成98%之类的估计。

旧结果补算（纯CPU，校验原始manifest、sample顺序与最终test性能，不修改旧run）：

```bash
CUDA_VISIBLE_DEVICES='' python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.learning_curves \
  --data "$T07_DATA" --output "$T07_RUNS/simulation/overfitting_audit_20260911" \
  --artifacts "$T07_RUNS/simulation/high_alpha_retrieval_20260910/artifacts" \
  "$T07_RUNS/simulation/generalization_20260911/preserve_adam/artifacts" \
  "$T07_RUNS/simulation/generalization_20260911/preserve_sam/artifacts" \
  "$T07_RUNS/simulation/generalization_20260911/preserve_fullfield_sam/artifacts"
```

`clean_best_train_test.json`记录最终best的干净训练/测试准确率与权重/特征SHA。
旧过程只有batch日志和best/last，不能恢复中间每个epoch的完整干净准确率；图中明确标注这一限制。
# 12. 目标相关商品扩充（实验室 Linux 服务器）

本节为独立新协议，不改变原始split或测试图库。先同步已推送的源码，再在仓库根目录执行。
本次授权最多3张卡，以下每个队列仅占一张；UUID必须通过nvidia-smi确认空闲后填写。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
TARGET=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
ABO=/DATA/DATA1/guest3/2026OpticsMoE/data/abo
POOL=$T07/runs/simulation/domain_pool_20260911
ASSETS=$T07/runs/simulation/standalone_assets_20260910
BEST=$T07/runs/simulation/generalization_20260911/preserve_adam/artifacts/best.pt

# CPU准备：不要覆盖已有POOL；失败时先检查报错/类别覆盖，不降低测试排除要求。
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.prepare_broad_abo \
  --target "$TARGET" --abo "$ABO" --output "$POOL" \
  --target-types-only --categories 10 --products-per-category 100 --minimum-products 4 --views 2

nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
# 用确认空闲的GPU UUID替换这一行。没有填写前不要执行后面的队列。
T07_GPU=GPU_REPLACE_WITH_IDLE_UUID
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --output "$T07/runs/simulation/domain_mixed_20260911" \
  --profiles domain_mixed --epochs 40 --steps 64
```

另外两组将profile和输出名同时换为`domain_curriculum`、`domain_target_control`；相同40轮预算。
队列不导入torch，子进程退出释放CUDA；SIGTERM/中断只终止自己的子进程，失败不自动重开。
初次运行先在`runs/smoke/`使用`--epochs 1 --steps 1`验证；为覆盖数据，自动steps仍可能大于1。
查看每组`status.json`、`<profile>/console.log`及`<profile>/artifacts/`中的
`execution.json`、`history.json`、`learning_curves.csv/png`、`final_report.json`。
`history.data_coverage`区分真实商品覆盖和重复视角，初始保底成绩不是新增收益。
每个阶段按原固定test评估live与EMA；按test选best，存在选择偏倚。

## 13. 75.21%起点：扩大池 / 同商品跨视角一致性（2026-09-12）

这部分在Linux源码仓库根目录执行；不是旧ZIP入口。原评估协议不变，不合并validation。
使用已推送的本轮源码，先激活服务器xml环境。原数据和资源不移动。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
TARGET=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
ABO=/DATA/DATA1/guest3/2026OpticsMoE/data/abo
ASSETS=$T07/runs/simulation/standalone_assets_20260910
BEST=$T07/runs/simulation/domain_mixed_20260911/domain_mixed/artifacts/best.pt
POOL=$T07/runs/simulation/domain_pool250_20260912

# CPU准备只执行一次；服务器此池已生成，直接跳过。旧池/旧run不得覆盖。
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.prepare_broad_abo \
  --target "$TARGET" --abo "$ABO" --output "$POOL" --target-types-only \
  --categories 10 --products-per-category 250 --minimum-products 4 --views 2

nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
T07_GPU=GPU_REPLACE_WITH_CONFIRMED_IDLE_UUID
# 先验证双视角前向/反向；覆盖全池会自动增加steps，不是只有一步。
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --profiles domain_refine_views --epochs 1 --steps 1 \
  --output "$T07/runs/smoke/domain_refinement_20260912_gpu4"

# 短检查成功/释放GPU后，正式同视角一致性组：
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --profiles domain_refine_views --epochs 24 --steps 128 \
  --output "$T07/runs/simulation/domain_refine_views_20260912"
```

另外两组保持同一个BEST和24×128主batch预算：

| profile与run名 | `--pool` |
| --- | --- |
| `domain_refine_control` / `domain_refine_control_20260912` | `$T07/runs/simulation/domain_pool_20260911`（旧922商品池） |
| `domain_refine_wide` / `domain_refine_wide_20260912_gpu2` | `$T07/runs/simulation/domain_pool250_20260912`（2058商品池） |

仅在不同且经检查空闲的GPU上并行，最多三张；否则在一张卡串行。进程结束会检查CUDA PID释放。
views比wide每主batch多一次同商品另一张图的前向/反向，因此训练FLOPs并不相等；推理成本完全相同。
`history.json`新增`view_consistency_weight`、`losses.view_consistency`、`paired_unique_images`，
`unique_images`仍只数主视角，路由计数也是主视角，避免双视角被算成Top4。
对照保留75.21%起点；epoch=-1表示未提升。每4轮按test选live/EMA，只保留best/last。

冻结完整Qwen的独立参照（本次已执行，不要覆盖同名run或把此入口当作学生训练）：

```bash
CUDA_VISIBLE_DEVICES="$T07_GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -u -m LightGenV2.tasks.t07_abo_image_retrieval.legacy_run --mode baseline \
  --run-dir "$T07/runs/simulation/frozen_qwen_20260912" \
  --cache "$T07/runs/simulation/frozen_qwen_20260912/features.pt"
```

该baseline模块是历史全模型评估入口，依赖仓库旧后端；独立学生仍只用standalone，不调用它。
若日后改变测试名单、图库、类别或预处理，另建baseline run/cache重新推理，不复用这个成绩。

## 14. 训练期语义关系蒸馏（目标81%，光路不改）

使用最新Git源码，先执行第13节变量定义（POOL为250池，BEST为已完成75.21%权重）。
缓存构建是唯一加载完整Qwen的进程；学生训练、推理仅用standalone。
必须等已有同卡队列完成，等待进程不创建CUDA上下文。以下为本次GPU1的有界串行方案：

```bash
T07_GPU=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d
QWEN=/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
KD_SMOKE=$T07/runs/smoke/domain_distillation_20260912

# 先全量生成仅训练缓存，再检查一轮学生蒸馏；两者串行、不同子进程。
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --teacher-model "$QWEN" \
  --profiles build_teacher_cache domain_distill_light --epochs 1 --steps 1 \
  --output "$KD_SMOKE" \
  --after-queue "$T07/runs/simulation/domain_refine_control_20260912/status.json"

# 上述完成后，用相同起点分别跑轻/较强蒸馏，绝不以第一组last继续第二组。
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" \
  --teacher-cache "$KD_SMOKE/build_teacher_cache/artifacts/cache.pt" \
  --profiles domain_distill_light domain_distill_strong --epochs 24 --steps 128 \
  --output "$T07/runs/simulation/domain_distillation_20260912" \
  --after-queue "$KD_SMOKE/status.json"
```

监督进程不导入torch，子进程退出后检查nvidia-smi中该PID已消失，再进入下一项。
没有空闲卡时安全停止，不自动挤占别人的训练。后台运行时使用nohup并将console.log放在run父目录。
仅保留各组best/last；缓存保存一次，由两个蒸馏组共用。缓存约22MiB张量，加身份信息；以实际报告为准。
`history.losses`增加relation_kd、teacher_correct_fraction、teacher_confidence，
`execution.training_only_teacher`记录缓存SHA/图像身份。最终推理不需要teacher缓存或2B权重。
最少389/480个命中才达标；当前各组尚未完成，不保证蒸馏一定提高性能。

补充0.6强度对照：确认上面的KD_SMOKE已`complete`、缓存SHA与README一致，
且本人的任务少于三张卡，才可以在第三张空闲卡启动。它没有新的推理层；仅KL系数不同。
下面GPU2是本次已释放的卡，执行前必须再次查nvidia-smi，不得终止别人的进程腾卡。

```bash
nvidia-smi -i 2 --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
T07_GPU=GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" \
  --teacher-cache "$KD_SMOKE/build_teacher_cache/artifacts/cache.pt" \
  --profiles domain_distill_stronger --epochs 24 --steps 128 \
  --output "$T07/runs/simulation/domain_distillation_stronger_20260912_gpu2"
```

## 15. 续训时恢复训练辅助头（不是新推理网络）

**口径更正**：本节旧`domain_distill_resumeaux`加载后还会把类别proxy初始化为原训练类别均值，
所以它实际仅保持光学辅助分类头的恢复。新`domain_distill_resumeaux_full`才保留全部辅助参数；
需要复现完整恢复时将下面profile改成该名称，且必须另选未使用的output目录、记录新源码commit。
不要在运行中的worktree修改源码，也不要覆盖旧run。新版本execution额外记录
`category_proxy_initialization=preserved_checkpoint`（旧组为`target_training_class_means`）。

沿用第13、14节变量。BEST必须是75.2083% live起点，不能替换成EMA best；程序核对固定SHA。
与0.3 strong相比只恢复其已保存的训练辅助头，其他配置和学生初始参数相同，优化器仍全新。
先等GPU4现有任务成功结束，再检查恢复/前向/反向/最终复评；不占用第四张卡。

```bash
T07_GPU=GPU-1b963983-7909-af6e-0528-f0f0661ab549
AUX_SMOKE=$T07/runs/smoke/domain_resumeaux_20260912_gpu4
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" \
  --teacher-cache "$KD_SMOKE/build_teacher_cache/artifacts/cache.pt" \
  --profiles domain_distill_resumeaux --epochs 1 --steps 1 --output "$AUX_SMOKE" \
  --after-queue "$T07/runs/simulation/domain_refine_views_20260912/status.json"

python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" \
  --teacher-cache "$KD_SMOKE/build_teacher_cache/artifacts/cache.pt" \
  --profiles domain_distill_resumeaux --epochs 24 --steps 128 \
  --output "$T07/runs/simulation/domain_resumeaux_20260912_gpu4" \
  --after-queue "$AUX_SMOKE/status.json"
```

恢复只涉及checkpoint中的`auxiliary_training_head`，不新增模型层，部署不需要该辅助头。
`execution.auxiliary_initialization`应为`restored_pinned_live_checkpoint`；旧对照为`fresh_random`。
原1轮检查不能当作正式训练提升，正式组依然从BEST独立开始。

## 16. 教师监督温度校准（训练统计给出0.03）

沿用第13、14节变量。同一75.2083%起点、5556图缓存、原测试协议。
与0.3 strong只差教师概率温度0.03；学生损失温度仍0.1，光Router推理温度不变。
等待GPU2的0.6对照结束，先做1轮检查，再独立做24轮，不增加同时使用的显卡数量。

```bash
T07_GPU=GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b
SHARP_SMOKE=$T07/runs/smoke/domain_sharpteacher_20260912_gpu2
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" \
  --teacher-cache "$KD_SMOKE/build_teacher_cache/artifacts/cache.pt" \
  --profiles domain_distill_sharpteacher --epochs 1 --steps 1 --output "$SHARP_SMOKE" \
  --after-queue "$T07/runs/simulation/domain_distillation_stronger_20260912_gpu2/status.json"

python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" \
  --teacher-cache "$KD_SMOKE/build_teacher_cache/artifacts/cache.pt" \
  --profiles domain_distill_sharpteacher --epochs 24 --steps 128 \
  --output "$T07/runs/simulation/domain_sharpteacher_20260912_gpu2" \
  --after-queue "$SHARP_SMOKE/status.json"
```

配置审计：`execution.common_config.relation_teacher_target_temperature=0.03`，
`relation_teacher_temperature=0.1`。未设置target_temperature的旧对照保持原有数值行为。

## 17. cap500训练池＋降低原商品重复比例（无教师）

沿用第13节变量。以下池已在服务器生成，不需重复生成，原始图片仍在data/abo。
池含3390商品/6780图；原任务120训练商品/1440图不删，原val/test不参与训练。
此组从已验证76.25%best继续，不能误用其他组BEST变量；每类采1原＋3外部商品，仍40图/batch。
167步/轮、24轮：较128步对照训练预算增加，不将数据和采样变化混称为单因素消融。

```bash
POOL500=$T07/runs/simulation/domain_pool500_20260912
DATA_BEST=$T07/runs/simulation/domain_refine_wide_20260912_gpu2/domain_refine_wide/artifacts/best.pt
DATA_SMOKE=$T07/runs/smoke/domain_pool500_mix13_20260912_gpu4
T07_GPU=GPU-1b963983-7909-af6e-0528-f0f0661ab549

# 只有另一台机器尚无此池时才执行；已有目录会拒绝覆盖。
# python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.prepare_broad_abo \
#   --target "$TARGET" --abo "$ABO" --output "$POOL500" --target-types-only \
#   --categories 10 --products-per-category 500 --minimum-products 4 --views 2

python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$DATA_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL500" --profiles domain_refine_pool500_mix13 \
  --epochs 1 --steps 1 --output "$DATA_SMOKE" \
  --after-queue "$T07/runs/simulation/domain_resumeaux_20260912_gpu4/status.json"

python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$DATA_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL500" --profiles domain_refine_pool500_mix13 \
  --epochs 24 --steps 167 --output "$T07/runs/simulation/domain_pool500_mix13_20260912_gpu4" \
  --after-queue "$DATA_SMOKE/status.json"
```

检查轮同样自动扩展到167步，保证商品覆盖；正式训练不从检查last继续。
`history.data_coverage.mixed_target_products_per_class=1`；旧对照仍为2，默认随机采样序列保持不变。
模型/光路/ROI/光Router Top2/alpha边界都没有因本实验修改，最终评估仍480 query与120商品图库。

## 18. 扩大现有电子卷积感受野（不改光路）

沿用第13节的ASSETS/TARGET/ABO，POOL仍cap250，不是cap500。
只改两个Vision depthwise卷积3×3→7×7；层数、分支数、Language、读出头、相位/光学布局不变。
自动将旧卷积居中零填充；新增15,360参数，不能只复制新模型源码却丢掉checkpoint metadata。
本组排在GPU1现有蒸馏队列后面，检查成功再正式运行；不是占第四张GPU。

```bash
CONTEXT_BEST=$T07/runs/simulation/domain_refine_wide_20260912_gpu2/domain_refine_wide/artifacts/best.pt
CONTEXT_SMOKE=$T07/runs/smoke/domain_context7_20260912_gpu1
POOL=$T07/runs/simulation/domain_pool250_20260912
T07_GPU=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$CONTEXT_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --profiles domain_refine_context7 \
  --epochs 1 --steps 1 --output "$CONTEXT_SMOKE" \
  --after-queue "$T07/runs/simulation/domain_distillation_20260912/status.json"

python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$CONTEXT_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --profiles domain_refine_context7 \
  --epochs 24 --steps 128 --output "$T07/runs/simulation/domain_context7_20260912_gpu1" \
  --after-queue "$CONTEXT_SMOKE/status.json"
```

检查轮自动125步以覆盖训练商品。正式仍从同一76.25%起点，不从检查last继续；
source commit、metadata、alpha及去光结果必须一起报告，不能把电子修改隐藏成纯训练技巧。
