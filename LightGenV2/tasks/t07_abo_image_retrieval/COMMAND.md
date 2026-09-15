# T07 独立工程：日常操作顺序

**先选目的，不要把所有章节顺序执行。**第1–4节是旧独立交付包操作；当前最佳固定权重复评用第20节。
第5节以后保留历史优化对照；最新服务器训练候选为第23–26节，各自独立，不需要重跑前面的所有实验。
带`LightGenV2.tasks...`的服务器命令必须从2026OpticsMoE仓库根目录执行，并使用指定Git版本与训练资产。

第1–4节从本任务文件夹或解压后的工程根目录执行。**旧独立包运行不需要2026OpticsMoE、T01、experiments、完整Qwen权重或网络访问。**
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

## 19. 训练图库类别数量归一化（不改变推理）

当前cap250训练图库不同类45～262商品，测试图库各类12商品；此组只消除gallery NLL的数量先验。
仍用76.25%原3×3模型起点、cap250池、每类2原+2外部、24×128步；无教师，不与7×7组合。
原始余弦排名和top-negative margin不变，不能在测试阶段按标签筛图库。
等待GPU2教师温度组成功完成再运行，任何依赖失败都停止，不抢别人的卡。

```bash
BALANCED_BEST=$T07/runs/simulation/domain_refine_wide_20260912_gpu2/domain_refine_wide/artifacts/best.pt
BALANCED_SMOKE=$T07/runs/smoke/domain_balanced_20260912_gpu2
POOL=$T07/runs/simulation/domain_pool250_20260912
T07_GPU=GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$BALANCED_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --profiles domain_refine_balanced \
  --epochs 1 --steps 1 --output "$BALANCED_SMOKE" \
  --after-queue "$T07/runs/simulation/domain_sharpteacher_20260912_gpu2/status.json"

python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$BALANCED_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --profiles domain_refine_balanced \
  --epochs 24 --steps 128 --output "$T07/runs/simulation/domain_balanced_20260912_gpu2" \
  --after-queue "$BALANCED_SMOKE/status.json"
```

检查轮自动125步；正式组独立从同一best开始。新增`gallery_class_balance=true`仅影响训练loss，
未打开此选项的所有旧组保持原行为；部署不需要本损失函数或训练图库扩充数据。

## 20. 明确指定最新权重的独立复评（不改assets）

默认`evaluate --assets ...`仍读assets/best.pt；要评训练目录中的新best，必须同时传checkpoint及其SHA。
这两个新参数仅用于evaluate，不会覆盖assets、修改模型或启动训练。读取前后再次核对SHA，
运行中的best若刚好被更新会拒绝评估；先确认新的epoch/SHA，再另建结果目录，不绕过校验。
下面明确复评已固定并独立核验的joint_curriculum第8轮live候选（79.375%），不是旧assets权重。
完整16轮联合课程仍在继续，使用固定副本，不读取会变化的训练best。
GPU1须先确认资源允许，不终止别人的进程；输出目录须尚不存在。也可选择另一张已获准且空闲的GPU。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
T07_GPU=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d
nvidia-smi -i "$T07_GPU" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
CUDA_VISIBLE_DEVICES="$T07_GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.cli evaluate \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --expected-checkpoint-sha256 7c7bc601b1048be13316a9e4863780fa5d49cf2f66b3bf3e3d68ced2d8f172f4 \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --device cuda --batch-size 4 --output "$T07/runs/simulation/recheck_joint_ep8_20260912_gpu1"
```

复评重新从原图生成120商品图库、查询全部480测试图，再同权重去光；输出逐图预测/特征、
相位图、clean train排除自身商品指标、execution/final_report。最终报告记录实际评估权重SHA。
在无GPU机器可用`--device cpu`，但速度/数值可能与CUDA bfloat16不同，不能冒充同硬件逐位复现。

## 21. 单模型权重平均对照（CPU生成，再独立复评）

两份模型必须来自相同初始化且合同一致；本组固定各50%，不搜索测试样本专属权重。
平均相位raw以及电子参数，冻结前端不变。最终推理一次，六次光捕获，不是双模型投票。
输出路径必须不存在；父checkpoint不覆盖，构建报告不会继承父模型准确率。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
AVG=$T07/runs/simulation/weight_average_strong_wide_20260912
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.weight_average \
  --left "$T07/runs/simulation/verify_strong_ep4_20260912_gpu1/best.pt" \
  --left-sha256 e5c0eaab4c84766b1ee231dd14271e97737604c0dcd675d9e2f4957c6932658d \
  --right "$T07/runs/simulation/domain_refine_wide_20260912_gpu2/domain_refine_wide/artifacts/best.pt" \
  --right-sha256 6f23466a1570a024e5bcf8408ae70a01065399e28e818af5a394dff15bfa850a \
  --right-weight .5 --output "$AVG"

# 仅在确认该GPU资源允许时运行；SHA直接读取构建报告，不手工转录。
AVG_SHA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' "$AVG/average_report.json")
CUDA_VISIBLE_DEVICES=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.cli evaluate \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$AVG/best.pt" --expected-checkpoint-sha256 "$AVG_SHA" \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --device cuda --batch-size 4 --output "$AVG/evaluation"
```

以evaluation/final_report与逐图预测判断，不以两个父模型命中的并集或平均分宣称提升。

## 22. 新增训练图的教师一致性对照

仅筛选额外训练图，原1440训练图全保留；不改原数据文件、测试集或图库。
教师缓存先按完整cap250池校验，之后子集化；不需要重新加载Qwen，也不新增推理层。
一轮检查通过后，正式16轮独立从同一个77.50%best开始，不从检查last继续。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
AGREE_SMOKE=$T07/runs/smoke/domain_teacher_agreement_20260912_gpu4
AGREE_BEST=$T07/runs/simulation/verify_strong_ep4_20260912_gpu1/best.pt
AGREE_CACHE=$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt
POOL=$T07/runs/simulation/domain_pool250_20260912
T07_GPU=GPU-1b963983-7909-af6e-0528-f0f0661ab549
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$AGREE_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --teacher-cache "$AGREE_CACHE" \
  --profiles domain_distill_teacher_agreement --epochs 1 --steps 1 --output "$AGREE_SMOKE" \
  --after-queue "$T07/runs/simulation/domain_pool500_mix13_20260912_gpu4/status.json"
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$AGREE_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --teacher-cache "$AGREE_CACHE" \
  --profiles domain_distill_teacher_agreement --epochs 16 --steps 128 \
  --output "$T07/runs/simulation/domain_teacher_agreement_20260912_gpu4" \
  --after-queue "$AGREE_SMOKE/status.json"
```

对照属于教师偏好的训练数据子集，不证明被筛出图标签错了；不得把它们从论文测试协议删除。

## 23. 逐图特征蒸馏（不改学生推理）

完整cap250池，教师前64维归一化，原1440训练图拟合正交坐标变换；矩阵仅用于教师目标。
不改学生初始化/前端/相位布局。每图余弦损失0.5，教师类别错误时关闭该图教师损失；关系KL关闭。
不做数据筛选、不叠加7×7卷积；仍从固定77.50%原3×3模型开始。等待GPU1卷积对照结束。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
FEATURE_SMOKE=$T07/runs/smoke/domain_aligned_feature_20260912_gpu1
FEATURE_BEST=$T07/runs/simulation/verify_strong_ep4_20260912_gpu1/best.pt
FEATURE_CACHE=$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt
POOL=$T07/runs/simulation/domain_pool250_20260912
T07_GPU=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$FEATURE_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --teacher-cache "$FEATURE_CACHE" \
  --profiles domain_distill_aligned_feature --epochs 1 --steps 1 --output "$FEATURE_SMOKE" \
  --after-queue "$T07/runs/simulation/domain_context7_20260912_gpu1/status.json"
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$FEATURE_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --teacher-cache "$FEATURE_CACHE" \
  --profiles domain_distill_aligned_feature --epochs 16 --steps 128 \
  --output "$T07/runs/simulation/domain_aligned_feature_20260912_gpu1" \
  --after-queue "$FEATURE_SMOKE/status.json"
```

alignment PT仅用于复核训练目标，可查看rotation、fit_sample_ids和源SHA；实验室推理只用模型best及原前端。

## 24. 逐图特征蒸馏 + 小型串联电子读出

仅最终头由LN384/Linear64改为LN384/Linear128/ReLU/Linear64；增加32,896参数，无新增分支。
签名成对初始化保持旧函数；metadata的retrieval_head=relu128保证重载时不会误用线性头。
其他配置与第23节相同，不改光学。先检查，再从同一77.50%源权重独立训练16轮。
以下在仓库根目录、含该profile的Git提交运行；本次使用GPU2，等待原均衡loss对照结束，最多三张GPU。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
ASSETS=$T07/runs/simulation/standalone_assets_20260910
TARGET=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
ABO=/DATA/DATA1/guest3/2026OpticsMoE/data/abo
MLP_BEST=$T07/runs/simulation/verify_strong_ep4_20260912_gpu1/best.pt
MLP_CACHE=$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt
POOL=$T07/runs/simulation/domain_pool250_20260912
MLP_SMOKE=$T07/runs/smoke/domain_feature_mlp_20260912_gpu2
T07_GPU=GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$MLP_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --teacher-cache "$MLP_CACHE" \
  --profiles domain_distill_feature_mlp --epochs 1 --steps 1 --output "$MLP_SMOKE" \
  --after-queue "$T07/runs/simulation/domain_balanced_20260912_gpu2/status.json"
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$MLP_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --teacher-cache "$MLP_CACHE" \
  --profiles domain_distill_feature_mlp --epochs 16 --steps 128 \
  --output "$T07/runs/simulation/domain_feature_mlp_20260912_gpu2" \
  --after-queue "$MLP_SMOKE/status.json"
```

勿把这个串联读出MLP与光学编码/CCD后处理混淆；相位大小、SLM布局、传播距离和六次捕获均未改。

## 25. 教师优先、逐步恢复GT的训练课程

从已核验77.9167%线性头权重出发。教师特征系数2；最终GT损失前4轮乘0.2，5–8轮恢复到1。
光学辅助分类/物理正则保持原系数，所有原GT标签保留。无新增推理模块、无新光路。
每轮live/EMA测试选best；属于test-selected，不是独立无偏测试。检查后正式16轮独立重启同一起点。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
ASSETS=$T07/runs/simulation/standalone_assets_20260910
TARGET=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
ABO=/DATA/DATA1/guest3/2026OpticsMoE/data/abo
TEACHER_FIRST_BEST=$T07/runs/smoke/domain_aligned_feature_20260912_gpu1/domain_distill_aligned_feature/artifacts/best.pt
TEACHER_FIRST_CACHE=$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt
TEACHER_FIRST_SMOKE=$T07/runs/smoke/domain_teacher_first_20260912_gpu4
POOL=$T07/runs/simulation/domain_pool250_20260912
T07_GPU=GPU-1b963983-7909-af6e-0528-f0f0661ab549
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$TEACHER_FIRST_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --teacher-cache "$TEACHER_FIRST_CACHE" \
  --profiles domain_distill_teacher_first --epochs 1 --steps 1 --output "$TEACHER_FIRST_SMOKE" \
  --after-queue "$T07/runs/simulation/domain_teacher_agreement_20260912_gpu4/status.json"
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$TEACHER_FIRST_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --teacher-cache "$TEACHER_FIRST_CACHE" \
  --profiles domain_distill_teacher_first --epochs 16 --steps 128 \
  --output "$T07/runs/simulation/domain_teacher_first_20260912_gpu4" \
  --after-queue "$TEACHER_FIRST_SMOKE/status.json"
```

history记录实际`supervised_loss_scale`与`teacher_feature_weight`，不只记录原始loss值。

## 26. 狭长输入的有限展宽对照

`contain_min_half`仅将原图长宽比超过2:1的输入适度非等比缩放至2:1，再白色补边到224²；不裁切。
其余图与原contain_white相同。原图文件/标签/图库不变，光路/相位布局/固定token数不变。
新输入合同必须重新评起点；不能复用原预处理分数。与第23节同样教师loss0.5，但每轮测试。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
ASSETS=$T07/runs/simulation/standalone_assets_20260910
TARGET=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
ABO=/DATA/DATA1/guest3/2026OpticsMoE/data/abo
ASPECT_BEST=$T07/runs/smoke/domain_aligned_feature_20260912_gpu1/domain_distill_aligned_feature/artifacts/best.pt
ASPECT_CACHE=$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt
ASPECT_SMOKE=$T07/runs/smoke/domain_bounded_aspect_20260912_gpu1
POOL=$T07/runs/simulation/domain_pool250_20260912
T07_GPU=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$ASPECT_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --teacher-cache "$ASPECT_CACHE" \
  --profiles domain_distill_bounded_aspect --epochs 1 --steps 1 --output "$ASPECT_SMOKE" \
  --after-queue "$T07/runs/simulation/domain_aligned_feature_20260912_gpu1/status.json"
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu "$T07_GPU" --assets "$ASSETS" --checkpoint "$ASPECT_BEST" --target "$TARGET" \
  --abo "$ABO" --pool "$POOL" --teacher-cache "$ASPECT_CACHE" \
  --profiles domain_distill_bounded_aspect --epochs 16 --steps 128 \
  --output "$T07/runs/simulation/domain_bounded_aspect_20260912_gpu1" \
  --after-queue "$ASPECT_SMOKE/status.json"
```

这不是校正CCD的透视变换，也不改变SLM尺寸；若采用该模型，实验输入生成必须遵循其metadata预处理。

## 27. 固定光学网络，只拟合原线性读出

这是一次训练集闭式拟合，不使用epochs/steps，不重复完整epoch训练。将队列放在GPU2小MLP训练之后，
等待时不占CUDA。仅改变原Linear384→64的weight/bias；不增加层，不改相位/前端/残差/alpha。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
ASSETS=$T07/runs/simulation/standalone_assets_20260910
TARGET=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
ABO=/DATA/DATA1/guest3/2026OpticsMoE/data/abo
RIDGE_BEST=$T07/runs/smoke/domain_aligned_feature_20260912_gpu1/domain_distill_aligned_feature/artifacts/best.pt
CACHE=$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt
POOL=$T07/runs/simulation/domain_pool250_20260912
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b --assets "$ASSETS" --checkpoint "$RIDGE_BEST" \
  --target "$TARGET" --abo "$ABO" --pool "$POOL" --teacher-cache "$CACHE" \
  --profiles domain_distill_readout_ridge --output "$T07/runs/simulation/domain_readout_ridge_20260912_gpu2" \
  --after-queue "$T07/runs/simulation/domain_feature_mlp_20260912_gpu2/status.json"
```

`execution.json`保存起点SHA、教师缓存身份和训练内部留出商品；`train_readout_features.pt`只含训练图特征。
`final_report.json`记录四个系数的训练留出余弦、选定系数、原测试及去光指标。只输出一份拟合完成的
`best.pt`（没有迭代epoch/周期权重）；test不参与本次系数选择。若要采用，仍须按第20节固定SHA独立复评。

## 28. 训练位置增强，测试输入和光路不变

本组使用原77.9167%权重，不使用第27节退步的ridge模型，也不使用小MLP。
50%训练图完整缩至90%～100%后随机白边放置，50%不改变位置；关系KL蒸馏0.3。
正式16轮，每轮评估，best/last；无新推理结构、无物体裁切。队列依赖已完成的GPU2读出检查。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
ASSETS=$T07/runs/simulation/standalone_assets_20260910
TARGET=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
ABO=/DATA/DATA1/guest3/2026OpticsMoE/data/abo
BEST=$T07/runs/smoke/domain_aligned_feature_20260912_gpu1/domain_distill_aligned_feature/artifacts/best.pt
CACHE=$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt
POOL=$T07/runs/simulation/domain_pool250_20260912
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b --assets "$ASSETS" --checkpoint "$BEST" \
  --target "$TARGET" --abo "$ABO" --pool "$POOL" --teacher-cache "$CACHE" \
  --profiles domain_distill_position_jitter --epochs 16 --steps 128 \
  --output "$T07/runs/simulation/domain_position_jitter_20260912_gpu2" \
  --after-queue "$T07/runs/simulation/domain_readout_ridge_20260912_gpu2/status.json"
```

## 29. 从已核验78.75%稳态续训（保留训练辅助头/教师目标）

只接受下列固定第11轮live副本和已记录SHA的教师坐标矩阵；传错checkpoint/矩阵会拒绝运行。
矩阵来自训练特征，非光相位的k空间变换；部署不依赖它或辅助头。优化器重新建立，不声称精确恢复旧训练状态。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
ASSETS=$T07/runs/simulation/standalone_assets_20260910
TARGET=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
ABO=/DATA/DATA1/guest3/2026OpticsMoE/data/abo
BEST=$T07/runs/simulation/verify_teacher_first_ep11_20260912_gpu4/best.pt
CACHE=$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt
POOL=$T07/runs/simulation/domain_pool250_20260912
ALIGNMENT=$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-1b963983-7909-af6e-0528-f0f0661ab549 --assets "$ASSETS" --checkpoint "$BEST" \
  --target "$TARGET" --abo "$ABO" --pool "$POOL" --teacher-cache "$CACHE" --teacher-alignment "$ALIGNMENT" \
  --profiles domain_distill_teacher_continue --epochs 12 --steps 128 \
  --output "$T07/runs/simulation/domain_teacher_continue_20260912_gpu4" \
  --after-queue "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/status.json"
```

`execution.json`记录辅助头恢复、category proxy保留和教师矩阵来源；每轮记录实际监督系数与正常test。
训练结束再重载best完成正常/去光复评，不是每轮都测去光。
该组没有新增网络层或改变光学传播。未超过当前best时不替换正式候选。

## 30. 同起点SAM续训对照（不增加推理网络）

与第29节保持相同起点/教师矩阵/12轮128步/学习率，只更换为SAM+AdamW训练更新。
SAM半径0.02、3轮热身，共用全体活跃参数L2范数；两次前向复用相同随机噪声。
约增加一倍前后向成本，因此这是等更新次数，不是等算力对照。仅best/last，不生成周期PT。
在仓库根目录、已激活xml环境执行；不要在已有输出目录重复运行，也不要同时启动第29节的第二份任务。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
ASSETS=$T07/runs/simulation/standalone_assets_20260910
TARGET=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
ABO=/DATA/DATA1/guest3/2026OpticsMoE/data/abo
BEST=$T07/runs/simulation/verify_teacher_first_ep11_20260912_gpu4/best.pt
CACHE=$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt
POOL=$T07/runs/simulation/domain_pool250_20260912
ALIGNMENT=$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d --assets "$ASSETS" --checkpoint "$BEST" \
  --target "$TARGET" --abo "$ABO" --pool "$POOL" --teacher-cache "$CACHE" --teacher-alignment "$ALIGNMENT" \
  --profiles domain_distill_teacher_continue_sam --epochs 12 --steps 128 \
  --output "$T07/runs/simulation/domain_teacher_continue_sam_20260912_gpu1" \
  --after-queue "$T07/runs/simulation/domain_bounded_aspect_20260912_gpu1/status.json"
```

等待阶段不占CUDA，前序退出后再次检查GPU空闲；被别人使用时停止队列，不抢占。
`history.json`中的`sam_rho`和`sam_loss_gap`确认实际执行；正式结论仍需固定best独立复评。

## 31. 同起点降低硬标签损失（不叠加SAM）

对应第29节的配对训练对照：前12轮仅最终CE/SupCon/图库NLL和margin乘0.5；
教师余弦仍2、光学辅助及正则不变。不改变样本、标签、评估图库或光路。不要自行扩大epochs冒充相同日程。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
ASSETS=$T07/runs/simulation/standalone_assets_20260910
TARGET=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
ABO=/DATA/DATA1/guest3/2026OpticsMoE/data/abo
BEST=$T07/runs/simulation/verify_teacher_first_ep11_20260912_gpu4/best.pt
CACHE=$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt
POOL=$T07/runs/simulation/domain_pool250_20260912
ALIGNMENT=$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b --assets "$ASSETS" --checkpoint "$BEST" \
  --target "$TARGET" --abo "$ABO" --pool "$POOL" --teacher-cache "$CACHE" --teacher-alignment "$ALIGNMENT" \
  --profiles domain_distill_teacher_continue_softgt --epochs 12 --steps 128 \
  --output "$T07/runs/simulation/domain_teacher_continue_softgt_20260912_gpu2" \
  --after-queue "$T07/runs/simulation/domain_position_jitter_20260912_gpu2/status.json"
```

比较第29/30/31节时均从固定78.75%权重开始，不串接彼此best。GPU型号不同，若出现新高，
须固定权重SHA后按第20节在同一RTX4090、batch4进行独立复评，不将小幅硬件数值波动记作突破。

## 32. 训练检索损失FP32对照（模型前向与推理精度不变）

与第29节相同起点/数据/损失系数/更新次数，仅关闭训练商品图库loss内部autocast。
单纯写`query.float()`不足以阻止外层autocast将矩阵乘法重新转成BF16，因此本组显式限定精度作用域。
不包含SAM、标签减半、教师去均值或任何网络结构变化。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
ASSETS=$T07/runs/simulation/standalone_assets_20260910
TARGET=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
ABO=/DATA/DATA1/guest3/2026OpticsMoE/data/abo
BEST=$T07/runs/simulation/verify_teacher_first_ep11_20260912_gpu4/best.pt
CACHE=$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt
POOL=$T07/runs/simulation/domain_pool250_20260912
ALIGNMENT=$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-1b963983-7909-af6e-0528-f0f0661ab549 --assets "$ASSETS" --checkpoint "$BEST" \
  --target "$TARGET" --abo "$ABO" --pool "$POOL" --teacher-cache "$CACHE" --teacher-alignment "$ALIGNMENT" \
  --profiles domain_distill_teacher_continue_fp32gallery --epochs 12 --steps 128 \
  --output "$T07/runs/simulation/domain_teacher_continue_fp32gallery_20260912_gpu4" \
  --after-queue "$T07/runs/simulation/domain_teacher_continue_20260912_gpu4/status.json"
```

原方法保持可复现。检查`execution.json/common_config/gallery_loss_full_precision=true`，
不要将此设置描述为提高光学仿真精度或修改部署；它仅改变训练图库损失的数值计算。

## 33. 扩充训练池＋逐图教师监督（250/500配对）

两组均从固定78.75%开始、12轮×250步，使用原1440训练图重拟合教师坐标，不传第29节的旧矩阵。
500池每轮至少250步才能覆盖外部商品；控制组也用250步，避免把更多更新次数当作数据收益。
仅缓存生成进程加载完整冻结Qwen；不进入学生推理。原测试/图库、输入和六次光计算均不变。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
ASSETS=$T07/runs/simulation/standalone_assets_20260910
TARGET=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
ABO=/DATA/DATA1/guest3/2026OpticsMoE/data/abo
BEST=$T07/runs/simulation/verify_teacher_first_ep11_20260912_gpu4/best.pt
CACHE=$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt
POOL250=$T07/runs/simulation/domain_pool250_20260912
POOL500=$T07/runs/simulation/domain_pool500_20260912
QWEN=/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda

# GPU1：等SAM完成，再生成500池缓存（复用5551图），随后同卡训练。
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d --assets "$ASSETS" --checkpoint "$BEST" \
  --target "$TARGET" --abo "$ABO" --pool "$POOL500" --teacher-model "$QWEN" \
  --reuse-teacher-cache "$CACHE" \
  --reuse-teacher-cache-sha256 aa5a5a952c0a96836b4b035d7905ca600f9082dc0bd26b5396cdab4cea36f2db \
  --profiles build_teacher_cache domain_distill_refit500 --epochs 12 --steps 250 \
  --output "$T07/runs/simulation/domain_refit500_20260912_gpu1" \
  --after-queue "$T07/runs/simulation/domain_teacher_continue_sam_20260912_gpu1/status.json"

# GPU4：另一个终端运行，同样先等待该卡前序结束；控制组复用已有250池教师缓存。
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-1b963983-7909-af6e-0528-f0f0661ab549 --assets "$ASSETS" --checkpoint "$BEST" \
  --target "$TARGET" --abo "$ABO" --pool "$POOL250" --teacher-cache "$CACHE" \
  --profiles domain_distill_refit250 --epochs 12 --steps 250 \
  --output "$T07/runs/simulation/domain_refit250_20260912_gpu4" \
  --after-queue "$T07/runs/simulation/domain_teacher_continue_fp32gallery_20260912_gpu4/status.json"
```

缓存的`final_report.json/reuse`记录精确复用/新增/未入新池数量。旧原图、缓存与模型不被覆盖。
两组新拟合的`teacher_feature_alignment.pt`仅用于训练解释与复现，不是相位权重、也不是部署依赖。

## 34. 最终线性读出64→256维（没有新增光学层或分支）

从固定78.75%线性64维权重转换，新192维初始化为零；所有非读出权重原样保留。
教师使用已有2048维训练缓存的前256维，不需要再加载或运行Qwen。与第33节refit250相同12×250步。
教师矩阵重新拟合为256×256，仅使用原训练图；因此不要传旧64维`--teacher-alignment`。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
ASSETS=$T07/runs/simulation/standalone_assets_20260910
TARGET=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
ABO=/DATA/DATA1/guest3/2026OpticsMoE/data/abo
BEST=$T07/runs/simulation/verify_teacher_first_ep11_20260912_gpu4/best.pt
CACHE=$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt
POOL=$T07/runs/simulation/domain_pool250_20260912
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b --assets "$ASSETS" --checkpoint "$BEST" \
  --target "$TARGET" --abo "$ABO" --pool "$POOL" --teacher-cache "$CACHE" \
  --profiles domain_distill_readout256 --epochs 12 --steps 250 \
  --output "$T07/runs/simulation/domain_readout256_20260912_gpu2" \
  --after-queue "$T07/runs/simulation/domain_teacher_continue_softgt_20260912_gpu2/status.json"
```

`model_audit`应为descriptor_dimension=256、trainable_parameters=2856405、六次捕获、Top2且无TF/attention。
只以新权重实际复测结果决定是否采用，不能继承64维权重成绩充当训练收益。
本候选训练使用上述入口，旧通用`cli finetune`的64维train_targets不能直接用于256维模型。

## 35. 联合教师课程（仅训练方法，不扩维、不加推理结构）

固定78.75%线性64维起点；完整cap250，逐图教师余弦2＋商品关系KL0.3，GT先减弱后恢复。
恢复原教师矩阵/辅助头；学习率为base，16×128步。它不是第34节256维读出的叠加版本。
队列等待GPU1扩充池结束再检查显卡空闲，不抢占他人进程、不额外使用第四张卡。

```bash
T07=LightGenV2/tasks/t07_abo_image_retrieval
ASSETS=$T07/runs/simulation/standalone_assets_20260910
TARGET=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
ABO=/DATA/DATA1/guest3/2026OpticsMoE/data/abo
BEST=$T07/runs/simulation/verify_teacher_first_ep11_20260912_gpu4/best.pt
CACHE=$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt
POOL=$T07/runs/simulation/domain_pool250_20260912
ALIGN=$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d --assets "$ASSETS" --checkpoint "$BEST" \
  --target "$TARGET" --abo "$ABO" --pool "$POOL" --teacher-cache "$CACHE" --teacher-alignment "$ALIGN" \
  --profiles domain_distill_joint_curriculum --epochs 16 --steps 128 \
  --output "$T07/runs/simulation/domain_joint_curriculum_20260912_gpu1" \
  --after-queue "$T07/runs/simulation/domain_refit500_20260912_gpu1/status.json"
```

测试仍480查询/120商品图库，live/EMA周期test选best，最终同权重去光，best/last而非周期PT。
启动和排队不是涨分证据；看该run的history/final_report后，再决定是否固定新高做独立复评。

## 36. 逐位置视觉教师缓存接口检查（4图CPU smoke）

仅离线教师使用Qwen视觉Transformer，学生推理结构完全不变。本命令只在CPU生成4张训练图的smoke缓存；
不会抢占第四张GPU。它标为不完整，正式训练加载器会拒绝，不能冒充全量训练缓存。
真实全量缓存约1.04GiB，正式生成与训练使用第37节的单卡队列，不要额外抢卡。

```bash
ROOT=/DATA/DATA1/guest3/2026OpticsMoE
T07=$ROOT/LightGenV2/tasks/t07_abo_image_retrieval
QWEN=/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.vision_teacher \
  --target "$ROOT/data/abo_similarity10_data" --abo "$ROOT/data/abo" \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --assets "$T07/runs/simulation/standalone_assets_20260910" --model "$QWEN" \
  --device cpu --batch-size 4 --max-images 4 \
  --output "$T07/runs/smoke/vision_teacher_cpu_probe_20260912"
```

输出`cache.pt`和`final_report.json`；已经存在的输出目录会拒绝覆盖。
输入必须是与学生一致的干净完整图，不能将位置教师直接用于随机裁剪/翻转后的图。
缓存只含训练图片的49×2048 FP16单位向量，不含教师网络、语言decoder或测试图片；不进入部署模型。

## 37. 训练期逐位置视觉监督（部署仍为原六次光计算）

与普通teacher_continue配方相同，另外每4步对干净对应图做一次V-only教师监督。
光学噪声沿用当步状态；主训练仍有增强，不把增强图错误地与干净位置特征匹配。
本轮从原78.75%的64维best开始，不用256维候选；原480-query/120-gallery完全保持。
先等GPU2的读出扩维实验结束，再生成完整5556图视觉缓存，然后退出教师进程、在同卡开始学生训练。

```bash
ROOT=/DATA/DATA1/guest3/2026OpticsMoE
T07=$ROOT/LightGenV2/tasks/t07_abo_image_retrieval
QWEN=/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_teacher_first_ep11_20260912_gpu4/best.pt" \
  --target "$ROOT/data/abo_similarity10_data" --abo "$ROOT/data/abo" \
  --pool "$T07/runs/simulation/domain_pool250_20260912" --teacher-model "$QWEN" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles build_vision_teacher_cache domain_distill_vision_patch --epochs 16 --steps 128 \
  --output "$T07/runs/simulation/domain_vision_patch_20260912_gpu2" \
  --after-queue "$T07/runs/simulation/domain_readout256_20260912_gpu2/status.json"
```

这两个cache不是同一个：原`--teacher-cache`是最终2048维检索教师；新视觉cache是49×2048位置特征。
队列自动向学生传递完整视觉cache。单独续训时需`--vision-teacher-cache`指向完整cache.pt，4图smoke缓存会被拒绝。
训练日志`vision_patch_supervision`在128步中应有32次更新；第3轮后全步平均权重0.2（实际每次辅助更新0.8）。
教师仅生成缓存时加载，不出现在学生训练/推理模块。最终只保留best/last，同权重正常/去光及原协议指标均要报告。

## 38. 已有效教师优先配方的一次训练随机种子对照

仅改变训练随机流为123；默认42保持旧实现。仍从原77.9167%开始，与原teacher_first的seed42结果比较，
不是从78.75%继续。模型/光路/Top2/alpha约束不变，原480-query/120-gallery不变；不是重分train/test。
种子也影响新建训练辅助头，但不改变加载的学生起始权重。仅一个额外seed，不做三次重复或集成推理。
先确认GPU4当前refit250的status.json存在；队列会等待它结束并核验GPU空闲，再启动，不杀其他任务。

```bash
ROOT=/DATA/DATA1/guest3/2026OpticsMoE
T07=$ROOT/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-1b963983-7909-af6e-0528-f0f0661ab549 \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/smoke/domain_aligned_feature_20260912_gpu1/domain_distill_aligned_feature/artifacts/best.pt" \
  --target "$ROOT/data/abo_similarity10_data" --abo "$ROOT/data/abo" \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --profiles domain_distill_teacher_first --epochs 16 --steps 128 --seed 123 \
  --output "$T07/runs/simulation/domain_teacher_first_seed123_20260912_gpu4" \
  --after-queue "$T07/runs/simulation/domain_refit250_20260912_gpu4/status.json"
```

只保留best/last；完整训练随机种子记录在execution.json和final_report.json。固定评估扰动种子不改。
各轮test选模和跨seed选择均属于test-selected，不得将挑出的最佳seed称为均值或无偏泛化结果。

## 39. 扩充商品每个4视角的训练数据对照

仍保留原train/test和120商品图库；只改变外部训练池。新池要求4张不同字节图片，
因可用性/去重条件而改变部分外部商品，不能称作严格同商品纯视角消融。已生成时跳过第一条，不能覆盖旧目录。
在已同步源码根目录执行（f695014b或后续）；数据路径继续指向原服务器目录，不移动数据。

```bash
ROOT=/DATA/DATA1/guest3/2026OpticsMoE
T07=$ROOT/LightGenV2/tasks/t07_abo_image_retrieval
POOL4=$T07/runs/simulation/domain_pool250_views4_20260912
CUDA_VISIBLE_DEVICES='' python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.prepare_broad_abo \
  --abo "$ROOT/data/abo" --target "$ROOT/data/abo_similarity10_data" --target-types-only \
  --categories 10 --products-per-category 250 --minimum-products 20 --views 4 --hamming-threshold 4 \
  --output "$POOL4"
```

已准备结果：1986外部商品/7944图，与原训练合计2106商品/9384图；清单SHA及类别计数见任务README。
教师缓存严格按ID+图片SHA复用旧缓存；只有新增训练图进行完整Qwen前向，教师进程退出后才开始学生训练。
学生沿用teacher_first、seed42、16轮×128步，从同一77.9167%起点开始；不是从79.375%续训，不混入其他改动。

```bash
ROOT=/DATA/DATA1/guest3/2026OpticsMoE
T07=$ROOT/LightGenV2/tasks/t07_abo_image_retrieval
QWEN=/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/smoke/domain_aligned_feature_20260912_gpu1/domain_distill_aligned_feature/artifacts/best.pt" \
  --target "$ROOT/data/abo_similarity10_data" --abo "$ROOT/data/abo" \
  --pool "$T07/runs/simulation/domain_pool250_views4_20260912" --teacher-model "$QWEN" \
  --reuse-teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --reuse-teacher-cache-sha256 aa5a5a952c0a96836b4b035d7905ca600f9082dc0bd26b5396cdab4cea36f2db \
  --profiles build_teacher_cache domain_distill_teacher_first --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_teacher_first_views4_20260912_gpu1" \
  --after-queue "$T07/runs/simulation/domain_joint_curriculum_20260912_gpu1/status.json"
```

GPU1原队列完成且资源允许才接续；等待不占CUDA。只保存best/last与训练日志，最后正常/去光重评原协议。
扩充图不进入测试图库，原冻结Qwen baseline的95.2083%仍是同一测试协议，不能另换更容易的测试分数比较。

## 40. 从已核验79.375%继续联合课程（不增加推理结构）

这是第二次16×128 warm restart：恢复当前最佳模型和训练辅助头，复用原教师坐标，
重新创建AdamW/余弦学习率调度及GT课程。仍使用旧cap250二视角池，不叠加第39节数据变化。
使用新增`domain_distill_joint_restart`，明确绑定下面79.375%权重SHA；旧第35节profile绑定78.75%，不能混用。
seed42；等待GPU4的seed123对照结束，检查空闲才启动，不抢占其他用户。
新增训练预算不保证提升；480-query/120-gallery、六次光捕获和全部光学几何不变。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-1b963983-7909-af6e-0528-f0f0661ab549 \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_restart --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_best_restart_20260912_gpu4" \
  --after-queue "$T07/runs/simulation/domain_teacher_first_seed123_20260912_gpu4/status.json"
```

起点SHA=`7c7bc601b1048be13316a9e4863780fa5d49cf2f66b3bf3e3d68ced2d8f172f4`。
该run已排队时不要再次执行；查看其status.json及子目录history/final_report。

旧`domain_joint_restart_20260912_gpu4`队列因源SHA不匹配，在等待阶段主动取消，未训练；不要重用这个目录。

## 41. 同起点联合蒸馏、全程较弱GT（训练方法对照）

与第40节相同起点/原cap250二视角池/教师坐标/基础学习率；唯一区别是最终GT损失整个16轮乘0.2。
教师余弦2、关系KL0.3、光学辅助与物理正则不变。第17轮若延长则恢复GT1；这里仅16×128，seed42。
原测试/图库不改，不移除类别或困难样本。3090训练若超过最佳，需在4090固定权重核验。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_restart_softgt --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_softgt_20260912_gpu2" \
  --after-queue "$T07/runs/simulation/domain_vision_patch_20260912_gpu2/status.json"
```

等待不占CUDA，依赖失败则不启动；不重用已有输出目录。只保存best/last，无周期PT。

## 42. 训练教师目标去除一半公共分量（已完成，没有新高）

第39节已完成并释放GPU1，本组已用0ad91cf8接续，监督2901717/学生2901720；初始79.375%。
完整16轮现已结束，best回退初始79.375%，去光62.9167%；CUDA已释放，GPU1接续第45节。
第40～41节继续收尾，没有超出3卡预算。本方法不修改光学直流/相位/CCD后处理，
只改变训练教师前64维目标；均值仅来自原1440训练行，外部图用同一均值，重新拟合teacher-only坐标。
与第40节保持同一起点/数据/损失权重/学习率/GT课程。不传旧`--teacher-alignment`；新均值与坐标不参与推理。
CPU梯度诊断不是性能验收；正式结果仍须正常/去光及固定权重复评原480-query/120-gallery。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --profiles domain_distill_joint_centerhalf --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_centerhalf_20260912_gpu1" \
  --after-queue "$T07/runs/simulation/domain_teacher_first_views4_20260912_gpu1/status.json"
```

须使用包含该profile的新commit，旧35055dc7尚无此功能；以execution记录实际源码SHA。
等待/空闲检查不抢GPU，输出目录必须不存在；只保留best/last，不生成周期相位PT。

## 43. 只解冻现有V→L merger最后一个线性层（已完成，没有新高）

第40节已完成并释放GPU4，已用f8a927e7接续；监督2938984/学生2938990。旧35055dc7没有该profile。
完整16轮现已结束，best回退初始79.375%，去光62.9167%；CUDA已释放，未替换正式全前端冻结模型。
恢复同一79.375%和训练辅助头；只将原`frontend.merger_fc2`的8,390,656个参数加入优化器，
学习率为原电子学习率的0.05倍。其它前端冻结，原推理层数/参数总量/光路不变；可训练参数增加须披露。
原电子基础LR=1.5e-5，merger基础LR=7.5e-7，再按同一热身/余弦调度；本地/服务器166项测试已通过。
用FP32主权重避免小更新被BF16舍入吞掉，输出仍BF16；检查初始与固定79.375%是否一致。
不加teacher centering，不加入新分支/TF/attention；仍16×128、原cap250二视角、原480/120评估。
教师仅训练，结束后报告正常/同权重去光；任何新高均独立复核，不能只看训练正确率。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-1b963983-7909-af6e-0528-f0f0661ab549 \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_merger --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_merger_20260912_gpu4" \
  --after-queue "$T07/runs/simulation/domain_joint_best_restart_20260912_gpu4/status.json"
```

等待阶段不占CUDA；依赖失败或GPU被其他人使用则停止，不抢占或杀其它任务。只保存best/last。

## 44. 按类别边际概率蒸馏，不强制同类内部商品排序（已完成、无提升；仅供复现）

已完成训练集诊断及170项两端测试，确认GPU2空闲后用1b052415启动，监督3039650/学生3039658。
16轮完成，最终选择初始权重（epoch=-1），正常79.375%、去光62.9167%，不采用此改动。GPU2已释放。
诊断仅说明原训练bank中同类内部项占关系KL的16.69%，不证明性能会改善。此配方从第40节79.375%起点出发，
将训练商品关系KL改成类别概率之和上的KL，聚合softmax/logsumexp使用FP32。
仍保留原商品Top1门控和置信度、排除自身商品、教师特征余弦2及GT课程；前端保持冻结。
无推理分类/类别筛选、无TF/attention或新光层，原数据/480-query/120-gallery不变。
不叠加42节去均值或43节merger解冻。只保留best/last，任何提升都要独立复核正常/去光。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_categorykd --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_categorykd_20260912_gpu2" \
  --after-queue "$T07/runs/simulation/domain_joint_softgt_20260912_gpu2/status.json"
```

GPU2为空才开始，不挤占他人任务；3090上出现新高需4090固定权重复核。已存在该run时不要重复执行。

## 45. Router整幅相位平移π/2的初始化对照（完成无新高，仅供复现）

仅给V/L两个router的整幅224×224相位加π/2取模，存成原格式raw sigmoid参数。
已用2421826f启动，监督3144821/学生3144831；两端174项测试及真实4图CPU检查通过，完整GPU初始复评仍79.375%。
16轮完成，选择平移后初始权重（epoch=-1），正常79.375%、去光62.9167%；不采用，正式最佳仍未平移。旧进程/CUDA已释放。
目的是减少约一半像素处于sigmoid边界的情况；不是改传播、ROI、相位编码或增加网络。
理想CCD有全局相位不变性，但含未调制/bypass光时不等价；保留现有训练噪声，并明确这项初始化差异。
必须重新计算完整原协议初始分数，不能因为理论近似不变就抄用79.375%。
训练更新统计从平移后的起点算起，不把固定π/2初始化当作学习量；不要用BF16余弦做精确等价性检查，先转FP32。
从固定原权重启动，恢复同一教师坐标和辅助头；前端冻结，不叠加去均值、merger或类别KL。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_routerorigin --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_routerorigin_20260912_gpu1" \
  --after-queue "$T07/runs/simulation/domain_joint_centerhalf_20260912_gpu1/status.json"
```

只有第42节完成并确认GPU1空闲后才能接续；不要修改正在训练的源worktree，不覆盖原权重/输出。
使用包含该新profile的已发布源码；正常与去光评估仍固定480查询/120图库，best/last之外不存周期PT。

## 46. 前4轮只更新相位，再恢复联合训练（完成无新高，仅供复现）

从固定79.375%起点开始。前4轮只更新专家/global/router相位，alpha、所有电子参数及训练辅助头均冻结，
但保留到相位的梯度路径；第5轮恢复联合训练，学习率沿用原余弦时钟。
不改变任何光学传播、ROI、输入/输出合同或网络结构。仍是原480查询/120训练商品图库、原教师和cap250二视角池。
不要与第45节相位平移混用；这是原始79.375%相位初始化的单独训练顺序对照。
先核对GPU4空闲；最多3张卡，不能挤占其他人的任务。正式执行前源码必须测试并push GitHub。
已用d1b5c6cf启动，监督3227016/学生3227019；两端180项测试通过，完整初始评估79.375%。
首轮权重已核验：12份相位更新，所有非相位及辅助头逐值不变；证据见该run的`artifacts/phase_only_freeze_verification.json`。
16轮完成，selected_epoch=-1，正常79.375%、去光62.9167%，仍选择初始权重，不采用本组。
监督3227016/学生3227019退出且释放CUDA，第49节已接续GPU4；不要重启本run。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-1b963983-7909-af6e-0528-f0f0661ab549 \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_phasefirst --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_phasefirst_20260912_gpu4"
```

只保存best/last；history中的`phase_only_warmup`记录阶段。正式采用前要独立复核新高与同权重去光结果。

## 47. 更强的逐图特征蒸馏（完成无新高，仅供复现）

从第40节固定79.375%权重开始，只把教师64维余弦损失权重2改成8，其余同joint_restart。
没有新增推理网络或改变光路，也不叠加第45/46节。先确认GPU2空闲，不挤占他人任务；最多3张GPU。
源码必须通过测试并同步GitHub。新高必须独立4090复评正常/去光，不能把蒸馏强度当作光贡献占比。
已用18c4e400启动，监督3308242/学生3308245；两端181项测试通过，完整初始评估79.375%，只改变教师余弦权重。
16轮完成，selected_epoch=-1，正常79.375%、去光62.9167%，未改善；原进程及CUDA已释放，不采用本组。
第50节已排队接续GPU2，不要重复运行本目录。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_feature8 --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_feature8_20260912_gpu2"
```

## 48. Router按物理相位弧度做Adam更新（完成无新高，仅供复现）

训练前向、光路、ROI、Top2、alpha及保存格式不变；仅两个Router采用弧度坐标Adam及角度EMA。
只在backward和optimizer.step之间临时转换，不能在该上下文内前向或保存。
不叠加第45节初始化平移、第46节相位优先或第47节强教师。固定原79.375%权重，初始评估需重算。
先完成测试并同步GitHub、真实输入梯度检查，再按下列命令接续第45节；依赖未完成时不分配CUDA。
已用a28a8438启动监督3357893等待依赖；两端187项测试及真实4图CPU一步检查通过，不要重复排队。
第45节已正常完成并释放GPU1，当前学生3378938已接续，execution确认radians配置；不占第四张卡。
完整初始原协议评估79.375%；等待训练后成绩，不把该初始值报成训练提升。
16轮已完成，selected_epoch=-1，正常79.375%、去光62.9167%，未产生新高，不采用。
监督3357893/学生3378938退出且释放CUDA，第51节已接续GPU1；不要重启此目录。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_routerradian --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_routerradian_20260912_gpu1" \
  --after-queue "$T07/runs/simulation/domain_joint_routerorigin_20260912_gpu1/status.json"
```

部署不需要新的训练优化器：checkpoint仍是FP32 raw-sigmoid相位。仅保留best/last，新高独立复核正常/去光。

## 49. 仅扩宽现有电子残差MLP（完成无新高，仅供复现）

16轮完成，selected_epoch=-1，正常79.375%、去光62.9167%；末轮EMA/live77.9167%。旧PID/CUDA释放，不采用。

现有四个192→384→192 MLP改为192→768→192，新增591360电子参数，无新增层/分支或光学尺寸。
复制隐藏单元并平分输出权重作保函数初始化，但必须重算完整初始评估，不能继承79.375%。
训练dropout/RNG会变化；不叠加第46—48节。固定原数据、教师、初始权重与Top2/alpha>0.4，正常及去光都要报告。
先测试、同步GitHub并检查真实输入，再接续GPU4的第46节；不占第四张卡，不重复启动已有run。
源码abb36acd，两端191项测试及真实4图CPU检查通过；监督3444725/学生3456323已接续GPU4。
完整初始重算Hit@1=79.375%，尚无训练后新高；正式最佳不变。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-1b963983-7909-af6e-0528-f0f0661ab549 \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_mlp768 --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_mlp768_20260912_gpu4" \
  --after-queue "$T07/runs/simulation/domain_joint_phasefirst_20260912_gpu4/status.json"
```

## 50. 原结构联合续训的seed123对照（完成无新高，仅供复现）

16轮完成，selected_epoch=-1，正常79.375%、去光62.9167%；末轮EMA76.0417%、live76.6667%。旧PID/CUDA释放，不采用。

只改变第40节joint_restart的训练seed，保持原384宽MLP、原raw Router优化器、固定79.375%起点及原协议。
监督3488475使用已发布源码abb36acd，第47节完成并释放GPU2后已接续学生3517887，不启动第四张卡。
seed123及GPU2 CUDA上下文已核验，完整初始分数需重新计算。
不能叠加第49节扩宽或第48节弧度坐标；评估不拼接不同seed预测，不改图库/标签，仅best/last。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_restart --epochs 16 --steps 128 --seed 123 \
  --output "$T07/runs/simulation/domain_joint_seed123_20260912_gpu2" \
  --after-queue "$T07/runs/simulation/domain_joint_feature8_20260912_gpu2/status.json"
```

新高需在4090独立复核正常及同权重去光；保留跨run/test选模偏差说明，不把不同seed的最佳值当作均值。

## 51. 仅提高Router弧度优化步长（完成无新高，仅供复现）

16轮完成，selected_epoch=-1，正常79.375%、去光62.9167%；末轮EMA77.9167%、live78.75%。旧PID/CUDA释放，不采用。

相对第48节仅提高两个Router的初始学习率0.0002→0.002；不是提高专家/global或电子学习率。
仍用原79.375%起点、384宽MLP、原教师坐标和16×128seed42；无推理/光路变化，不叠加第49节。
先同步已测试源码并做真实输入检查，再排到第48节之后；等待不占CUDA，不重复启动同名run。
完整初始分数要重新计算，实际学习率见`execution.json/optimizer_initial_rates_by_kind`。
源码e78dab7a，两端200项测试及真实4图CPU一步检查通过；监督3567958等待第48节完成，不占CUDA。
第48节已正常完成并释放显存，学生3595623已接续GPU1。完整初始评估79.375%，实际Router初始LR=0.002已核验。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_routerradian_fast --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_routerradian_fast_20260912_gpu1" \
  --after-queue "$T07/runs/simulation/domain_joint_routerradian_20260912_gpu1/status.json"
```

只存best/last，新高需独立正常/去光复评；大步长导致性能下降也必须保留记录，不覆盖原最佳。

## 52. 将TRAIN类别子空间校准折叠进现有读出（已独立复评79.5833%）

源码bbb9876d，独立RTX4090/batch4正常79.5833%、mAP@10=0.7685648975、同权重去光62.7083%。
复评PID393380已退出且释放CUDA；下面完整命令保留供复现，已有输出不覆盖。当前目标81%未达到。

仅改原线性读出的weight/bias，其他张量包括所有相位不变；不是新增推理层或类别候选筛选。
0.5保留率已经通过TEST缓存探索选择，需披露该偏差。缓存382/480不能作为实际BF16前向成绩。
拟合只占CPU，源码先测试并同步GitHub；输出目录存在时禁止覆盖。旧辅助头不继承，不直接套用旧的恢复辅助头训练profile。
已用bbb9876d完成CPU拟合，两端207项测试通过；下面拟合命令仅供复现，当前目录已存在，不要重复运行。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.readout_calibration \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --expected-checkpoint-sha256 7c7bc601b1048be13316a9e4863780fa5d49cf2f66b3bf3e3d68ced2d8f172f4 \
  --features "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/evaluation/retrieval_features.pt" \
  --expected-features-sha256 9ca01f0ee51c85bbffe7fd804ce173bdb31af3aa39a82f6c9816b4f84bfffb5d \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --retention 0.5 --output "$T07/runs/simulation/readout_subspace_20260912"
```

拟合报告状态为`fitted_not_evaluated`，不代表完整训练/评估成功。
**复跑前重新确认指定GPU空闲；原复评已结束，不能假定GPU4仍空闲。**仍最多3张卡，不挤占其它任务。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
T07_CALIBRATION_SHA=$(python -c "import json; print(json.load(open('$T07/runs/simulation/readout_subspace_20260912/calibration_report.json'))['checkpoint_sha256'])")
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 \
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.cli evaluate \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --checkpoint "$T07/runs/simulation/readout_subspace_20260912/best.pt" \
  --expected-checkpoint-sha256 "$T07_CALIBRATION_SHA" \
  --device cuda --batch-size 4 --output "$T07/runs/simulation/readout_subspace_20260912/evaluation"
```

复评原480-query/120-gallery及同权重去光；只有独立final_report是真实前向结果，缓存预测不替代复评。

## 53. 仅扩大既有视觉残差卷积感受野（完成无新高，仅供复现）

源码6c1320c2已同步GitHub，两端215项测试与真实4图CPU输出/外圈梯度检查通过。
监督419922/学生419941在GPU2 RTX3090完成16轮，已退出并释放CUDA。selected_epoch=-1，正常79.375%、去光62.9167%，无新高，不采用。

V的两个depthwise卷积3×3→13×13，L仍causal5、MLP仍384；新外圈补零，新增61440电子参数而不加层/分支。
从**未校准79.375%**起点恢复辅助头/教师坐标，不套用第52节校准后的权重。光学传播/ROI/Top2/α>0.4不变。
先完成测试、GitHub同步与真实输入CPU检查，再检查GPU2空闲启动；所有初始指标重新计算，不继承旧成绩。
原cap250二视角、教师2/KL0.3/GT课程/基础LR/raw Router、16×128seed42。旧context7不算同起点配对对照。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_vision13 --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_vision13_20260913_gpu2"
```

仅best/last，独立正常与去光复评后才能采用；扩大电子感受野的收益不能全部归于光。

## 54. 仅训练时随机缩减近纯白留边（已完成无新高，仅供复现）

源码1dab4190已同步GitHub，两端228项测试及真实训练224图/梯度检查通过；监督437698/学生437702在GPU5 RTX3090完成16轮并退出、释放CUDA。
最终selected_epoch=-1，正常79.375%、同权重去光62.9167%，回退起点；未优于正式79.5833%，不采用。
全1440训练图裁框检查无阈值非白像素丢失，364张满足条件；测试和网络结构仍保持原样，不能提前填入新分数。

原V3/L5/MLP384、光路/Top2/α>0.4和测试预处理均不变，起点为未校准79.375%权重。
50%概率尝试保守白边裁框/等比放大，另一半不做本项变换；具体阈值见README和`white_margin_box`。
任一通道<254的像素须全部保留在裁框内；不是物体分割器，不处理CCD，不改标签/样本/测试/图库。
先测试、同步GitHub，审计真实训练224图和梯度，再确认GPU5空闲启动。本助手加本组共两张GPU，不干预其他任务。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-d53ce4c8-272d-c2fb-dc09-f182d586c4eb \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_whitezoom --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_whitezoom_20260913_gpu5"
```

不叠加第53节，初始和最终评估仍从原测试/图库图像重新编码；只存best/last，新高单独4090正常/去光复核。

## 55. 训练时仅投影冲突的教师梯度（已完成未超过正式最佳，仅供复现）

源码8be14e72已同步GitHub，两端240项测试及真实8图CPU反向检查通过；监督454819/学生454823在GPU4 RTX4090完成16轮并退出、释放CUDA。
最终第16轮EMA正常79.375%、mAP@10=0.770077629、同权重去光62.2917%；不是初始回退，仍未超过正式79.5833%。
前8轮约95%–99%批次触发投影，日志证明开关生效；更高排序mAP不替代目标Hit@1，也不证明该训练一定有泛化收益。

原数据/光学/推理不变；既有损失分成primary与teacher，只投影教师在共享活动参数上与primary冲突的分量。
这不是完整PCGrad；GT-only辅助头不投影，不叠加SAM/额外视图/弧度优化，日志见README说明。
先测试并同步GitHub，再做真实训练图/缓存教师的CPU反向检查；确认GPU4空闲才启动。
本助手已有GPU2、GPU5，该组最多使总占用达到授权的三张卡；绝不终止其他人的任务。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-1b963983-7909-af6e-0528-f0f0661ab549 \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_teacherproject --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_teacherproject_20260913_gpu4"
```

初始完整复评必做；参数量/alpha/光路SHA须核验，最终报告正常和同权重去光；只保留best/last。

## 56. Language CCD全幅电子汇聚（已完成，未采用；以下为历史复现命令）

16轮完成，选中第14轮EMA正常71.4583%、同权重去光63.75%，干净TRAIN99.3056%；低于第52节正式最佳，不采用。
最终报告在下面run的`domain_distill_joint_languagefull/artifacts/final_report.json`。监督487163/学生487166已退出并确认CUDA释放；不重复启动原输出目录。
源码24858b7b已同步GitHub，两端245项测试及真实4图CPU前后向检查通过；监督487163/学生487166已在GPU2 RTX3090启动。
12份相位梯度有效，权重没有在CPU检查中被修改；执行审计确认参数量/前端冻结/Top2/alpha及光学源码SHA不变。
初始完整复评57.7083%（277/480）、干净TRAIN留商品外79.0972%；旧权重对新池化有明显失配，需要重新适配，不是最终成绩。
原mask/光路/ROI/Top2/alpha不变；仅language专家和global的CCD电子池化直接输出77×224，不再取前77/224行。
不新增参数、层或分支；这是电子读出接口变化，需重新测初始化成绩。Vision不改，不与第54/55节叠加。
起点是未校准79.375%版本，不是第52节；原辅助头和教师坐标保持固定。源码先测试、推送GitHub，再独立worktree运行。
下面GPU2须在启动前重新确认空闲，最多三张属于本任务的GPU；不终止其他人的进程。目录存在禁止覆盖。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_languagefull --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_languagefull_20260913_gpu2"
```

初始、最终均重新编码原480测试和120商品图库，记录干净训练性能与同权重去光；只留best/last。

## 57. 仅降低原商品重复采样（已完成、未采用；历史复现命令）

16轮完成，selected_epoch=-1：正常79.375%/去光62.9167%为初始权重回退，不是训练提升；监督494597/学生494600及CUDA均已释放。
源码888d08d6已同步GitHub，两端247项测试、实际5556训练图SHA/配额/产品覆盖检查通过；监督494597/学生494600在GPU0 RTX4090运行。
执行审计确认原光学、参数量、前端冻结、Top2/alpha及原prefix CCD读出不变，未叠加其它对照。
初始化完整复评79.375%，与起点一致；后续训练成绩仍待实际产生。
保持原prefix CCD读出及全部光路，不与全幅读出/梯度投影/白边增强叠加。起点仍为未校准79.375%及原辅助头/教师坐标。
每batch40图从每类2原+2外改成1原+3外；成员/标签/测试/图库不变，原训练干净评估仍用全部1440图。
先测试、同步GitHub，并检查实际池的每类配额/产品覆盖；只在GPU空闲、且本任务不超过三卡时启动。
实际选择启动前空闲的GPU0；以后复跑仍必须重新检查，不能据此判断它一直空闲；旧输出不覆盖。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_mix13 --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_mix13_20260913_gpu0"
```

只存best/last；有新高也先独立正常/同权重去光复核，不能用训练提升或类别路由指标替代原Hit@1。

## 58. 轻量适配已有patch输入卷积（用户转向新数据集，已停止）

完成第8轮后用户确认换数据集，向已核实监督517418发SIGTERM，监督回收学生517421；ps及nvidia确认释放。
保留best/last及原日志，状态failed_or_interrupted，不是16轮完成。以下为历史命令，当前不要启动。
源码b69785ab已同步GitHub，两端256项测试通过；真实4图CPU初始化/梯度/patch单步更新检查通过。
监督517418/学生517421已在GPU1 RTX4090启动；启动时GPU0/1/2三组，第56节完成后GPU2已释放，不自动补满资源。
执行审计确认patch基础LR7.5e-7、原光学源码SHA/Top2/alpha、4356373训练参数和27578368冻结参数。
GPU完整初始化复评79.375%，与未校准起点一致；正式训练新成绩仍待产生。
实际第4轮last的CPU审计证实12份相位、patch weight/bias均更新，其余9份前端张量逐位未变；这不是性能通过条件。
证据为本run的`artifacts/patch_training_update_check.json`，记录被读取last的SHA；没有额外周期PT或CUDA进程。
只解冻已有patch Conv3d的两个参数，不增加推理层/分支。其它前端冻结；相位正常参与原训练，光学源码/ROI/Top2/alpha不变。
1573888参数从冻结转为训练；总参数量不变。FP32主权重、原BF16计算接口，patch基础LR7.5e-7。
原未校准79.375%起点和joint_restart原2:2采样/损失，单独对照，不叠加第55–57节。
先通过测试并发布源码，检查真实训练图的初始化输出与梯度，再核验GPU1空闲启动；不超过三张属于本任务的卡。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue \
  --gpu GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/verify_joint_ep8_20260912_gpu1/best.pt" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --abo /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --pool "$T07/runs/simulation/domain_pool250_20260912" \
  --teacher-cache "$T07/runs/smoke/domain_distillation_20260912/build_teacher_cache/artifacts/cache.pt" \
  --teacher-alignment "$T07/runs/simulation/domain_teacher_first_20260912_gpu4/domain_distill_teacher_first/artifacts/teacher_feature_alignment.pt" \
  --profiles domain_distill_joint_patch --epochs 16 --steps 128 --seed 42 \
  --output "$T07/runs/simulation/domain_joint_patch_20260913_gpu1"
```

初始成绩重新完整复评；只存best/last，新高须独立正常/去光复核。报告须注明patch已训练，不能称整个Qwen输入头冻结。

## 59. 替代数据集配对初筛：COIL-100（与ABO成绩分开）

先读`reports/reproduction/DATASET_SCREENING.md`；60训练物体身份与40测试物体分开，图库160图、查询320图，角距至少30度。
这是**自定义受控实例检索**，不是官方SOP或原ABO类别检索，也不等于真实电商效果。初筛不训练，两者均64维；光电沿用原权重/光路。
从仓库根运行。原始官方zip为130688843字节，SHA由prepare命令验证；不把图片或zip提交Git/实验室发布包。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
COIL=/DATA/DATA1/guest3/2026OpticsMoE/data/coil100_source
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen prepare-coil \
  --archive "$COIL/coil-100.zip" --data "$COIL" \
  --output "$T07/runs/simulation/coil100_protocol_20260913"

# 先nvidia-smi确认GPU0为空闲；两条命令串行复用一张卡，不在占用卡上启动。
CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 \
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen optical \
  --data "$COIL" --manifest "$T07/runs/simulation/coil100_protocol_20260913/protocol.json" \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/readout_subspace_20260912/best.pt" \
  --expected-checkpoint-sha256 50a8607eec392c00cf3675533490cb8ef953245af6f5d9cfbc9d616bf7d22701 \
  --output "$T07/runs/simulation/coil100_optical_transfer_20260913" --batch-size 4

# 完成后再次检查上一个PID已释放CUDA；Qwen独立进程，不混入光电模型。
CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 \
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen qwen64 \
  --data "$COIL" --manifest "$T07/runs/simulation/coil100_protocol_20260913/protocol.json" \
  --model /DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --output "$T07/runs/simulation/coil100_qwen64_20260913" --batch-size 4
```

每组输出`final_report.json`、逐图预测和64维特征。比较前必须确认两份报告的manifest SHA相同、都为320-query/160-gallery、同物体相关性。
差距用`100*(Qwen64 Hit@1 - Optical Hit@1)`，单位百分点；去光用同权重直接移除，不另训练。
只在新结果支持时开展训练池适配；当前不得把“下载成功”“脚本通过”写成达到10/4个百分点目标。

## 60. Grocery81自然图搜标准图初筛（不改变COIL或ABO协议）

使用服务器已有官方划分文件，全部2640训练图预留、2485测试查询、81标准图图库；val不用。
这是**细类别检索，不是未见SKU检索**。不使用文本描述、不按粗类别过滤图库、不合并类别。
仍用第59节的冻结Qwen64与原光电权重，先不训练。初筛源码需先测试并同步GitHub。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
GROCERY=/DATA/DATA1/guest3/2026OpticsMoE/data/GroceryStoreDataset/dataset
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen prepare-grocery \
  --data "$GROCERY" --output "$T07/runs/simulation/grocery81_protocol_20260913"
```

随后分别执行第59节的`optical`/`qwen64`命令，只替换：

- `--data`为`$GROCERY`；
- `--manifest`为`$T07/runs/simulation/grocery81_protocol_20260913/protocol.json`；
- `--output`分别为`$T07/runs/simulation/grocery81_optical_transfer_20260913`和`$T07/runs/simulation/grocery81_qwen64_20260913`。

每次启动前重新检查指定GPU为空闲；COIL当时用GPU0不代表该卡现在仍空闲。运行完检查自己的PID和CUDA已释放，不杀其他任务。
数据文件变动/缺图/重复SHA会报错，先调查，不强行放宽校验生成漂亮结果。预留训练不代表已经训练完成。

## 61. Grocery81短程光电适配（完整命令）

数据协议已由第60节固定；不改数据划分，不加载完整Qwen，不改光路。先进入包含本提交的干净Git工作树，使用xml环境。
下面GPU4仅是本次运行设备，重跑前必须用nvidia-smi确认空闲；已有run目录不得覆盖，新复跑应换明确run_id。
训练每批16图，`--batch-size 4`仅控制评估；测试每5轮选best，存在选模偏差，不称独立测试。

```bash
cd /DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t07_screen_20260913
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 \
/home/guest3/miniconda3/envs/xml/bin/python -u -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/GroceryStoreDataset/dataset \
  --manifest "$T07/runs/simulation/grocery81_protocol_20260913/protocol.json" \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/readout_subspace_20260912/best.pt" \
  --expected-checkpoint-sha256 50a8607eec392c00cf3675533490cb8ef953245af6f5d9cfbc9d616bf7d22701 \
  --output "$T07/runs/simulation/grocery81_adapt_20260913" \
  --epochs 20 --steps 100 --eval-every 5 --classes-per-batch 8 --batch-size 4
```

首次CUDA检查使用同一完整命令，仅把epochs/steps改成1/2、output改为`$T07/runs/smoke/grocery81_adapt_20260913`；
它依然全量评估，但仅2次梯度更新，不当作正式成绩。成功且PID释放后再执行正式命令。
`history.json`记录每轮loss/batch检索及周期全量train/test；最终`final_report.json`含best的正常/去光，`phase_masks.png`用于浏览。
`phase_update_last.json`和`phase_update_best.json`分别量化最后/最佳相位相对起点的圆周RMS，不混称best一定学动。
只保留best/last；其中last含优化器/EMA供审计，当前入口是fresh continuation，不宣称自动精确断点恢复。
公开标准图库参与训练拟合；测试自然照片只用于定期选模，没有在训练批中使用。干净训练检索与测试检索使用相同81图图库。
Qwen64基准71.9517%来自`grocery81_qwen64_20260913`，不是ABO的94.375%；当前还没有宣称适配达标。

已运行的Grocery bc110024版本入口名为`grocery_transfer`。当前同一训练引擎整理为`retrieval_adapt`，不保留重复实现；
逐提交复现原run应checkout bc110024并使用execution.json中的原命令，不能用新源码冒充原提交。

## 62. COIL训练物体适配（40测试物体不参与拟合）

沿用第59节固定协议。60个TRAIN物体各0/90/180/270度形成240张训练参考图，其余4080张作训练query。
训练目标为同物体多正例NLL+SupCon；测试仍为40个其他物体、320查询/160图库，最小角距30度。
不使用TEST图库训练。原光路/Top2/α>0.4/64维输出不变，无新增推理头或TF/attention。
每5轮TEST选live/EMA最佳，明确test-selected；训练图库240、测试160，角度分布也不同，train/test差值不全是过拟合。

```bash
cd /DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t07_screen_20260913
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi
# 确认GPU4空闲、上组自己的PID退出后执行；不占用他人的卡。
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 \
/home/guest3/miniconda3/envs/xml/bin/python -u -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/coil100_source \
  --manifest "$T07/runs/simulation/coil100_protocol_20260913/protocol.json" \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/readout_subspace_20260912/best.pt" \
  --expected-checkpoint-sha256 50a8607eec392c00cf3675533490cb8ef953245af6f5d9cfbc9d616bf7d22701 \
  --output "$T07/runs/simulation/coil100_adapt_20260913" \
  --epochs 20 --steps 100 --eval-every 5 --classes-per-batch 8 --batch-size 4
```

首次CUDA检查把epochs/steps改1/2、output改为`$T07/runs/smoke/coil100_adapt_20260913`，不当正式成绩。
`fitting_manifest.json`记录训练query/参考图身份与SHA；COIL参考图source_split必须为train。
只存best/last，最终正常/同权重去光、路由和相位变化。Qwen64同协议基准99.375%。

COIL20轮结束后，独立新进程复核已固定的epoch10最佳权重（仍需先确认指定GPU空闲）：

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 \
/home/guest3/miniconda3/envs/xml/bin/python -u -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen optical \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/coil100_source \
  --manifest "$T07/runs/simulation/coil100_protocol_20260913/protocol.json" \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/coil100_adapt_20260913/best.pt" \
  --expected-checkpoint-sha256 a220f144d6fd7d18bb25e11c647cdaea3b04f4cc964959de7f1d476e7f3f202e \
  --output "$T07/runs/simulation/coil100_adapt_20260913/verification" --batch-size 4
```

新评估入口读取checkpoint训练历史：该权重是本协议训练/test-selected，并非旧ABO直接迁移。
只做固定权重推理，不再次训练；manifest顺序与冻结Qwen一致。目录存在时不覆盖原结果。

## 63. ABO配对抗过拟合、SHAPE与OFF（2026-09-13）

本轮最多4张卡，不代表必须占4张。COIL不追加；原ABO/Grocery失败证据不删除。
ABO两组共同15轮×64步、seed42、原64维/Top2/alpha>0.4/6次采集。全物体增强、独立专家相位dropout5%/router2%，相同AdamW超参；SAM组rho=.03，第一轮后生效。
不把新数据/低清测试当原协议成绩，不挑测试样本降低Qwen。两组都记录干净train/test；TEST选模有偏差。只best/last。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# 从已测试、已推GitHub的干净工作树执行。先nvidia-smi选空闲卡，再设置CUDA_VISIBLE_DEVICES。
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.broad_transfer \
  --mode adapt --profile recovery_phase05_adam \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --target /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --checkpoint "$T07/runs/simulation/readout_subspace_20260912/best.pt" \
  --output "$T07/runs/simulation/recovery_phase05_adam_20260913" \
  --adapt-epochs 15 --steps 64 --batch-size 4
# 配对只改profile为recovery_phase05_sam、output为recovery_phase05_sam_20260913。
```

SHAPE作者数据：https://figshare.com/articles/dataset/24100704 ，CC BY4.0；匿名category/SKU均作为标签，不输入文本。
官方文件training_set.zip/test_set.zip放data/shape_source；准备时强制作者MD5/size匹配、安全解压、跨train/test重复SHA检查。
初筛8类按sha256(shape-category42:<category>)固定，不根据成绩选择；全选中TRAIN图库/TEST查询，不是未见SKU或端到端货架识别。

```bash
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retail_sources prepare-shape \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/shape_source \
  --output "$T07/runs/simulation/shape8_protocol_20260913"
# 下列GPU命令需逐个确认空闲卡，再设置CUDA_VISIBLE_DEVICES；本轮最多同时4卡。
# 目录已经存在时不会覆盖。复现请换新的output，不删除现有结果。
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen qwen64 \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/shape_source \
  --manifest "$T07/runs/simulation/shape8_protocol_20260913/protocol.json" \
  --model /DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --output "$T07/runs/simulation/shape8_qwen64_20260913" --batch-size 4
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/shape_source \
  --manifest "$T07/runs/simulation/shape8_protocol_20260913/protocol.json" \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/readout_subspace_20260912/best.pt" \
  --expected-checkpoint-sha256 50a8607eec392c00cf3675533490cb8ef953245af6f5d9cfbc9d616bf7d22701 \
  --output "$T07/runs/simulation/shape8_adapt_20260913" \
  --epochs 20 --steps 100 --eval-every 5 --classes-per-batch 8 --batch-size 4
# 仅多TRAIN视图SKU参与配对训练，评估图库和查询完整保留；不是过滤难测试SKU。
# 固定最佳权重独立复评（本轮已完成，不重新训练）：
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen optical \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/shape_source \
  --manifest "$T07/runs/simulation/shape8_protocol_20260913/protocol.json" \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/shape8_adapt_20260913/best.pt" \
  --expected-checkpoint-sha256 05d2b8d5738c2ec5e9ca6febea8cd145c9e276a0169aae7d5b57d7f3b4a1cf07 \
  --output "$T07/runs/simulation/shape8_adapt_20260913/verification" --batch-size 4
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retail_sources audit-off \
  --output "$T07/runs/smoke/off_feasibility_20260913"
```

OFF官方API只查一页100条，记录原始JSON/SHA；这是热门商品可行性检查，不是代表性采样/正式benchmark。
同一imgid的front_en/front_fr不算两张，numeric原图可能是营养表/条码；未经照片内容和重复上传审计不启动训练。
图片CC BY-SA、数据库ODbL分别遵守；不能把许可证当所有包装/肖像权保证。大量图片按官方建议使用AWS，不并发轰炸主站。

OFF小样本可视审计（最多10张400像素原照片，按ID哈希选5商品，串行下载，不生成检索成绩）：

```bash
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retail_sources preview-off \
  --data "$T07/runs/smoke/off_feasibility_20260913" \
  --output "$T07/runs/smoke/off_photo_review_20260913"
```

ABO登记照片预算CPU审计：1/3/12张均按sample_id哈希固定，保留所有120图库商品/480查询，报告所有档位，不把新协议替代原12视角成绩。使用原生冻结Qwen64，不拟合降维器，不按测试表现选择视角。

```bash
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.catalog_view_audit \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --optical-cache "$T07/runs/simulation/readout_subspace_20260912/evaluation/retrieval_features.pt" \
  --qwen-cache "$T07/runs/simulation/frozen_qwen_20260912/features.pt" \
  --output "$T07/runs/simulation/catalog_protocols_20260913"
```

该CPU审计还单独报告同SKU实例检索：保留全部40个TEST商品，每个商品12图按sha256(abo-instance42:<sample_id>)固定前4图库、后8查询，共160图库/320查询。
这是新任务定义（找同一SKU，不是找同类别不同商品），不作原ABO精度提升；未重新训练，也不根据特征/成绩选图。

## 64. 已登记ABO重新训练、SHAPE全视角与OFF扩展审计

从GitHub已同步commit的干净工作树执行，使用xml环境。以下输出是本轮固定run ID，重跑必须换新ID，不覆盖结果。
最多4卡是用户授权上限，不自动占满；本轮GPU0 ABO、GPU1 SHAPE、GPU3冻结Qwen，启动前必须检查该卡确实空闲。
CUDA数字枚举可能不等于nvidia-smi物理序号，正式命令使用GPU UUID（下面对应0/1/3）避免选错卡。
ABO所有200商品每个8张训练、4张查询，同SKU判分；原val40商品也纳入新协议，无验证集。所有视图保留，不挑容易商品。
周期TEST选best/live/EMA属于test-selected结果；原数据多数来自同一spin序列，因此不得声称独立拍摄场景泛化。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.enrolled_abo \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --output "$T07/runs/simulation/abo200_enrolled_protocol_20260913"
nvidia-smi
# 以下UUID只在确认空闲后使用；不是允许挤占其他人的任务。
CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$T07/runs/simulation/abo200_enrolled_protocol_20260913/protocol.json" \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/readout_subspace_20260912/best.pt" \
  --expected-checkpoint-sha256 50a8607eec392c00cf3675533490cb8ef953245af6f5d9cfbc9d616bf7d22701 \
  --fresh-trainable --multi-view --epochs 40 --steps 100 --eval-every 5 --batch-size 4 \
  --output "$T07/runs/simulation/abo200_enrolled_fresh_20260913"
# checkpoint只提供明确的结构配置；--fresh-trainable不载入其任何参数。
# 仅从经manifest校验的assets/best.pt加载冻结Qwen前端，其余构造初始化。所有相位从raw0开始。
CUDA_VISIBLE_DEVICES=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/shape_source \
  --manifest "$T07/runs/simulation/shape8_protocol_20260913/protocol.json" \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/shape8_adapt_20260913/best.pt" \
  --expected-checkpoint-sha256 05d2b8d5738c2ec5e9ca6febea8cd145c9e276a0169aae7d5b57d7f3b4a1cf07 \
  --multi-view --lr-scale .5 --epochs 20 --steps 100 --eval-every 5 --batch-size 4 \
  --output "$T07/runs/simulation/shape8_multiview_20260913"
CUDA_VISIBLE_DEVICES=GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen qwen64 \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$T07/runs/simulation/abo200_enrolled_protocol_20260913/protocol.json" \
  --model /DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --output "$T07/runs/simulation/abo200_enrolled_qwen64_20260913" --batch-size 4
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retail_sources review-off-batch \
  --data "$T07/runs/smoke/off_feasibility_20260913" \
  --output "$T07/runs/smoke/off_review20_20260913"
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retail_sources render-off-review \
  --output "$T07/runs/smoke/off_review20_20260913"
```

先做CUDA冒烟：相同参数改`--epochs 1 --steps 2`、output改`runs/smoke/abo200_enrolled_20260913`或`shape8_multiview_20260913`，确认完成释放后再正式训练。
OFF最多60张400px，仅用官方AWS、串行，不因AWS缺图回退轰炸主站；report记录所有下载失败，不静默删商品或配对。
训练只存best/last、执行身份、phase_update、正常/去光、路由和train/test历史。终止时只针对自己的精确PID，结束后检查CUDA占用。

## 65. ABO五个百分点差距主线：路由修复/蒸馏配对，SHAPE备选

结构说明见`reports/reproduction/ABO_ENROLLED_ARCHITECTURE.md`。不改光路、ROI、Top2、64维、电子参数量。
ABO从本协议35 EMA继续，不使用旧类别检索权重；程序检查checkpoint的manifest身份，禁止旧协议权重泄漏。
40轮配对只差TRAIN-only关系蒸馏，前3轮router-only后联合训练；SHAPE20轮无独立预热。
最多4卡的用户预算仍有效，本轮最多3卡（物理0/4/3），按UUID检查空闲，不占他人卡；GPU1已有其他任务，蒸馏组改用GPU4。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi
CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$T07/runs/simulation/abo200_enrolled_protocol_20260913/protocol.json" \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/abo200_enrolled_fresh_20260913/best.pt" \
  --expected-checkpoint-sha256 918956321fe865d74408611716a420330a53cb760e827636d787f6e9a3abfade \
  --multi-view --refine-profile route_repair --lr-scale .5 --epochs 40 --steps 100 --eval-every 5 --batch-size 4 \
  --output "$T07/runs/simulation/abo200_route_repair_20260913"
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$T07/runs/simulation/abo200_enrolled_protocol_20260913/protocol.json" \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/abo200_enrolled_fresh_20260913/best.pt" \
  --expected-checkpoint-sha256 918956321fe865d74408611716a420330a53cb760e827636d787f6e9a3abfade \
  --multi-view --refine-profile route_distill --lr-scale .5 --epochs 40 --steps 100 --eval-every 5 --batch-size 4 \
  --teacher-features "$T07/runs/simulation/abo200_enrolled_qwen64_20260913/normal_features.pt" \
  --expected-teacher-sha256 c6eb631c268d2446a2f783854c86d8493cdbcaa04c0669148b16d9785016d8d7 \
  --output "$T07/runs/simulation/abo200_route_distill_20260913"
CUDA_VISIBLE_DEVICES=GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/shape_source \
  --manifest "$T07/runs/simulation/shape8_protocol_20260913/protocol.json" \
  --assets "$T07/runs/simulation/standalone_assets_20260910" \
  --checkpoint "$T07/runs/simulation/shape8_multiview_20260913/best.pt" \
  --expected-checkpoint-sha256 077e58ef3bd650b3ff9bd834fb2489bac0cf6c40dc680d8ffadaba5c3fd1189d \
  --multi-view --refine-profile shape_views --lr-scale .5 --epochs 20 --steps 100 --eval-every 5 --batch-size 4 \
  --output "$T07/runs/simulation/shape8_view_refine_20260913"
```

CUDA冒烟：ABO蒸馏组同参数改`--epochs 3 --steps 2 --router-warmup-epochs 1 --eval-every 3`，output改`runs/smoke/abo200_route_distill_checked_20260913`；覆盖router-only及非零蒸馏权重的联合反向。旧冒烟因文档教师SHA多抄一个字符而被身份校验拦截，未训练；不得绕过校验。
SHAPE冒烟改`--epochs 1 --steps 2`及`runs/smoke/shape8_view_refine_20260913`，保留所有正式测试图，不拿冒烟分数当新成绩。
读取normal_features.pt后按manifest匹配，只保留1600 TRAIN向量供损失使用；800 QUERY不参与蒸馏，不加载完整Qwen/TF到学生进程。
新增最优选模先检查router资格（每专家≥5%、最高Top2组合≤80%、至少3组合），再比R@1/mAP；目标还需R@1≥80.125%。不合格会明确标router_eligible=false。

## 66. 已登记ABO：实例大池预训练后微调，对照SAM抗过拟合

本轮保持200商品/1600训练图库/800查询、冻结Qwen64=85.125%不变。两组都从同一78.125%权重继续。
使用既有`domain_pool250_views4_20260912`的1986商品/7944照片，按同SKU而非同category作正例；不导入旧类别任务训练权重。
加载时核验pool SHA、原200商品排除合同、每张照片SHA、全目标商品ID及文件哈希零交集。原近重复筛查为启发式，不保证所有语义近似商品独立。
这是listing图片（可能含局部/包装），不能把全部称为同商品纯转台角度。目标测试仍原封不动。
SAM两次forward使用相同随机噪声/dropout状态；只增加训练成本，没有额外推理层、TF、attention。3%phase dropout只用于专家/global，router关闭，部署关闭。
先运行CUDA冒烟：B命令的`--external-pretrain-epochs 12 --epochs 30 --steps 100`改成`--external-pretrain-epochs 1 --epochs 1 --steps 2`，output改`runs/smoke/abo_external_sam_checked_20260913`；完整池/完整TEST仍保留，覆盖两个阶段及正常/去光评估。
GPU0/1仅为本服务器本轮分配，运行前检查空闲；不终止其他人任务。最多两张，本轮不追加SHAPE作业。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi
# A: 目标集抗过拟合对照
CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" \
  --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --multi-view --refine-profile sku_regularized --lr-scale .5 --epochs 30 --steps 100 --eval-every 5 --batch-size 4 --bank-batch-size 16 \
  --output "$R/abo200_sam_control_20260913"
# B: 外部12轮 -> 目标30轮；命令串行完成两个阶段，无需手工换权重
CUDA_VISIBLE_DEVICES=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" \
  --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --external-pool "$R/domain_pool250_views4_20260912" \
  --external-root /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --expected-external-sha256 e6cf6b923ccdfb7c4c04dcf6b033455d1d49db737d2a9ebca941849d6c28e315 \
  --external-pretrain-epochs 12 --multi-view --refine-profile sku_regularized \
  --lr-scale .5 --epochs 30 --steps 100 --eval-every 5 --batch-size 4 --bank-batch-size 16 \
  --output "$R/abo200_external_sam_20260913"
```

正式TEST和train_clean继续batch4，与原结果一致；仅TRAIN bank编码batch16，训练为8个SKU各2张，共16图。微调阶段不混外部图库；测试候选始终1600图。
初始CUDA冒烟使用eval/bank16，原权重评出78.000%而非batch4的78.125%（1/800数值差异）；冒烟只验流程，不报作新精度。正式命令拆分batch参数消除这个口径变化。
history的epoch为全程编号；phase_epoch为阶段内轮数。外部阶段只记录TRAIN batch命中/loss，不测试选模；目标阶段记录完整train_clean和TEST。
只写best/last，初始best也参与比较，若最终仍选epoch0则无提升。目标阶段对已均衡router继续施加温和均衡损失，最后报告同权重正常/去光与mask变化。

## 67. 强正则退化后的温和配对：仅SAM开关不同

保持原200商品协议、冻结前端、64维、光路/alpha/Top2；不采用缓存白化或线性度量诊断中的变换。
两组均从78.125%原best开始，不从退化的last继续。各20轮×100步，TEST batch4、bank16。
没有几何增强或额外phase dropout；只做亮度/对比度.95..1.05。原硬件噪声配置不变，仅训练启用概率从25%降至10%，router噪声关闭。
LR为原基础值的.1，电子weight decay=.01、route辅助系数乘.25；两个profile唯一差别为SAM rho=0/.002。
先对弱SAM做1轮×2步CUDA冒烟（output换到runs/smoke/abo_mild_sam_20260913）；确认后运行以下正式任务，不覆盖旧run。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi
# 物理GPU0，仅确认旧对照PID1665320退出且卡空闲后使用
CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --multi-view --refine-profile sku_mild_adamw --lr-scale .1 --epochs 20 --steps 100 --eval-every 5 --batch-size 4 --bank-batch-size 16 \
  --output "$R/abo200_mild_adamw_20260913"
# 物理GPU1：先确认外部组PID1665328正常结束、卡空闲，再运行；GPU3已有其他任务，不使用
CUDA_VISIBLE_DEVICES=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --multi-view --refine-profile sku_mild_sam --lr-scale .1 --epochs 20 --steps 100 --eval-every 5 --batch-size 4 --bank-batch-size 16 \
  --output "$R/abo200_mild_sam_20260913"
```

本轮串接复用物理GPU0/1两张RTX4090，不抢占其他任务；结束逐PID核验显存释放。

## 68. 外部软关系预训练，再恢复目标SKU监督

**最新指令：本节正式训练暂不启动，先执行第69节纯相位优化。代码/流程验证保留，不代表已训练取得新结果。**

只改训练，不改光路/输入尺寸/描述子/电子网络。使用同一78.125%起点，不继承旧类别任务学生权重。
外部12轮只做冻结Qwen64关系KL及原光学/路由约束；不做外部SKU NLL、SupCon或全正例聚合。
目标20轮恢复原GT检索损失，教师完全关闭。轻微色彩增强、无SAM/额外相位dropout。
旧缓存包含旧目标TRAIN图片，程序只保留外部清单的7944行并逐图核对SHA；所有目标商品/图像行拒绝进入外部损失。
目标800 QUERY/1600图库保持固定，外部阶段不按TEST选模。只保存best/last，含初始最佳回退。

以下命令的GPU为示例，必须先确认空闲；当前0/1有温和配对任务，不中断它们。
先CUDA冒烟：改`--external-pretrain-epochs 1 --epochs 1 --steps 1 --eval-every 1`，output为`runs/smoke/abo_external_relations_20260913`；完成后释放并检查PID，正式启动另行记录。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi
CUDA_VISIBLE_DEVICES=GPU-d53ce4c8-272d-c2fb-dc09-f182d586c4eb python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --external-pool "$R/domain_pool250_views4_20260912" --external-root /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --expected-external-sha256 e6cf6b923ccdfb7c4c04dcf6b033455d1d49db737d2a9ebca941849d6c28e315 \
  --external-teacher-cache "$R/domain_teacher_first_views4_20260912_gpu1/build_teacher_cache/artifacts/cache.pt" \
  --expected-external-teacher-sha256 7e17b59c36b64ec0a0f2a0ca52399c4b3dfc00dfab0dbd138db617ba79c6a87e \
  --external-pretrain-epochs 12 --multi-view --refine-profile sku_external_relations \
  --lr-scale .1 --epochs 20 --steps 100 --eval-every 5 --batch-size 4 --bank-batch-size 16 \
  --output "$R/abo200_external_relations_20260913"
```

## 69. 先固定电子，验证光学相位的实际贡献

两组从同一78.125%权重各训练12轮，不加载任何教师。电子投影/残差、alpha、读出、冻结Qwen前端全部逐参数保持不变，只有12份光相位更新。
`phase_only`专家/global基准LR=.002；`phase_only_hot`=.006；router均=.00003，再乘相同预热/余弦调度。不是改变传播距离、像素或相位精度。
继续10%batch的既有光学噪声/DC，关闭router人工噪声及电子dropout。每轮SHA核验电子不变；EMA不得更新冻结参数。
只存best/last；每3轮评估完整TRAIN/TEST、检查专家分散性，最后同权重去光。若电子真的不变，去光结果应与起点一致；不把相位变化或alpha当准确率贡献。
先用热相位组做1轮×2步CUDA冒烟（output改`runs/smoke/abo_phase_only_20260913`），验证有限梯度、12相位变化、电子SHA及正常/去光流程；然后才正式启动。
复用当前温和配对结束后释放的GPU0/1，不能在旧PID1986961/2008216仍运行时叠加或杀掉其他任务。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi
CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --multi-view --refine-profile phase_only --lr-scale 1 --epochs 12 --steps 100 --eval-every 3 --batch-size 4 --bank-batch-size 16 \
  --output "$R/abo200_phase_only_20260913"
CUDA_VISIBLE_DEVICES=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --multi-view --refine-profile phase_only_hot --lr-scale 1 --epochs 12 --steps 100 --eval-every 3 --batch-size 4 --bank-batch-size 16 \
  --output "$R/abo200_phase_only_hot_20260913"
```

## 70. 相位阶段之后：适度扩容卷积教师与轻量对照

先确认第69节两个任务结束，并记录TRAIN/TEST、相位变化、电子SHA和同权重去光。
不要中断它们抢卡。教师只有原电子残差内部V核7×7/L因果核7、MLP隐藏768；
六次光路及相位形状、alpha>0.4、64维输出不变，无新增分支/attention，冻结前端不解冻。
教师增加607488可训练参数（总3389973），不是已经压缩的部署学生。当前先实现教师/对照，后续蒸馏另做验证。
仅从当前协议最佳继续，绝不加载旧类别协议的训练权重。若纯相位未产生更好候选，保留下面78.125%起点。
两组各20轮×100步，每5轮完整TEST选模（有选择偏差），LR-scale=.2，其余相同；
仍只存best/last，最终同权重去光。原最佳不覆盖。

先把下面教师命令的`--epochs 20 --steps 100 --eval-every 5`改为`--epochs 1 --steps 2 --eval-every 1`，
output改为`$T07/runs/smoke/abo_conv_teacher_20260913`，完成CUDA初始化/反向/正常去光验证并检查PID退出后才正式运行。
必须先检查所指定GPU空闲；命令假定已激活xml并进入GitHub已同步源码工作树，不能原地改运行中的源码。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi
CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --multi-view --refine-profile sku_conv_teacher --lr-scale .2 --epochs 20 --steps 100 --eval-every 5 --batch-size 4 --bank-batch-size 16 \
  --output "$R/abo200_conv_teacher_20260913"
CUDA_VISIBLE_DEVICES=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --multi-view --refine-profile sku_capacity_control --lr-scale .2 --epochs 20 --steps 100 --eval-every 5 --batch-size 4 --bank-batch-size 16 \
  --output "$R/abo200_capacity_control_20260913"
```

## 71. 外部转台视角池：先准备数据，不占GPU、不改测试集

原外部池是listing展示图；本节使用官方真实spin视角。原目标200商品（含全部未用于训练的视角所在spin）
都排除，不能给目标商品追加训练照片后仍声称原8/4划分未变。
共享spin的不同SKU也排除；各商品均匀选12角度，按稳定SKU哈希次序，每个固定product_type最多50商品。
VASE原候选只有30，因此未筛重前上限约480商品/5760图。实际数量以ready报告为准，不能填计划数量。
逐图SHA、目标dHash近重复检查只是保守防重，不证明所有近似款式不存在。

原图及官方元数据缓存在`data/abo/spins`；run只有清单、状态和报告。默认网络总下载上限1GiB、4线程，
不下载40GB整包，不覆盖已有文件。超限/缺少最低类型覆盖会失败，不生成ready报告；原始缓存保留供审计/重用。
官网/实际spins README写CC BY4.0，AWS登记页写CC BY-NC4.0，已记录差异，对外使用前需核实，不给期刊许可保证。

先用`--products-per-type 1 --minimum-products 1 --views 4 --max-download-mib 64`，
output改为`$T07/runs/smoke/abo_spin_pool_20260913`做真实下载/身份排除/加载验证。数据准备不使用CUDA。
正式命令如下；输出目录不得已存在，失败时不直接覆盖旧run。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.prepare_spin_abo \
  --abo-root /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --target-root /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --target-manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" \
  --expected-target-sha256 f1749d5fc22d2dfee6a1333ce2b35e9fa600a070f949eba8420b4def41906dde \
  --products-per-type 50 --minimum-products 10 --views 12 --workers 4 --max-download-mib 1024 \
  --output "$R/abo_spin_pool50_views12_20260913"
```

确认`report.json`的status=ready及`manifest_sha256`，再用`enrolled_regularization.load_external_pool`重新加载验证。
之后才安排外部→目标训练：沿用`retrieval_adapt`的`--external-pool`、`--external-root data/abo`及报告SHA参数，
从当前协议最佳权重开始，禁止加载旧类别协议权重。新增数据训练尚未启动；等待教师配对结果后确定profile和正式命令，
不额外抢占其GPU。目标1600图库/800查询及冻结Qwen64=85.125%全程保持不变。

已ready的40图数据池可先验证CUDA外部→目标切换。以下仅1+1轮、各1步；不是正式性能训练。
先确认GPU0空闲。本次命令从原78.125%轻量模型开始，没有电子扩容或教师缓存，source为5b06f747。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --external-pool "$T07/runs/smoke/abo_spin_pool_20260913" --external-root /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --expected-external-sha256 22a36f45f65795ca6081fbe6390c85a2815efec2517f707bc06c29dc748afdfb \
  --external-pretrain-epochs 1 --multi-view --refine-profile sku_capacity_control \
  --lr-scale .2 --epochs 1 --steps 1 --eval-every 1 --batch-size 4 --bank-batch-size 16 \
  --output "$T07/runs/smoke/abo_spin_curriculum_20260913"
```

## 72. 转台预训练与目标微调：原轻量网络，不增加教师/电子参数

本轮已完成：source `b9a1bb4b850dd328dfd7e400c8fe7f2ee5c3e91e`，原物理GPU3 RTX4090，PID2271973已退出。
最终目标epoch20 EMA正常78.75%、TRAIN97.125%、去光78.125%，仅下降.625pp，路由合格。
准确率小幅提高但光学贡献变弱，不直接替换原78.125%部署参照。后续光优先阶段训练见第77节。
数据已ready并重载检查：453商品/5436图，SHA=`68fd35b6a2308f13e01963eb1233545c44eb07f5caa48ff655dc6fc302b1ed8f`。
不要在已有output上重复执行；以下保留复现命令。

先完成第71节正式数据准备：`report.json`必须ready，`status.json`必须complete；
训练加载器还会逐图重验SHA、目标协议和SKU/spin隔离。失败或尚未生成报告时不要启动。
只使用当前200-SKU协议的78.125%起点，不加载旧类别检索权重，不改1600 TRAIN/gallery和800 QUERY。
外部12轮→目标20轮，各100步；外部不用TEST选模，末状态进入目标阶段时重置Adam动量/EMA。
目标每5轮评估live/EMA，保持原best回退、同权重去光、路由资格约束；TEST参与选模的偏差照常披露。
与已完成的`abo200_capacity_control_20260913`目标20轮相比多12轮外部计算，**不是等计算预算比较**。

`sku_capacity_control`：原电子结构/64维头，光学Top2、六次10cm、alpha>0.4不变。
无教师loss，温和亮度/对比度增强、EMA、电子decay=.01，原噪声/DC作用于10%训练batch。
训练batch=8商品×2张不同视图=16；下面batch-size4仅控制完整评估，bank16仅TRAIN编码。
每轮`fitting_bank_epoch_start`是已有干净bank的排除自身检索，不是训练batch命中率，也不是轮末权重。
`phase_rms_change_from_run_start_rad`保留12份相位逐轮变化；仍只存best/last两份模型。

先在已ready的40图冒烟池使用1外部轮+1目标轮、各1步、eval-every1，
output设`runs/smoke/abo_spin_diagnostics_20260913`，确认新增诊断和完整评估正常后再正式执行。
数据预备工作树可能仍在运行；训练使用另一个固定GitHub commit的干净工作树，不在运行源码上更新。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
# 激活xml，进入已同步GitHub的干净源码工作树；先检查指定GPU确实空闲。
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SPIN_POOL_SHA=68fd35b6a2308f13e01963eb1233545c44eb07f5caa48ff655dc6fc302b1ed8f
nvidia-smi
CUDA_VISIBLE_DEVICES=GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --external-pool "$R/abo_spin_pool50_views12_20260913" --external-root /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --expected-external-sha256 "$SPIN_POOL_SHA" \
  --external-pretrain-epochs 12 --multi-view --refine-profile sku_capacity_control \
  --lr-scale .2 --epochs 20 --steps 100 --eval-every 5 --batch-size 4 --bank-batch-size 16 \
  --output "$R/abo200_spin_pretrain_20260913"
```

上面对应已启动命令，不代表达标；实际完成状态与结果以README及run记录为准。

## 73. 小幅保留空间布局的末端读出对照（不改光路）

`sku_spatial_readout`与已完成的`sku_capacity_control`只有读出头不同：
旧全token mean/max（384维、原LayerNorm）+ L最终输出的49个图像位置固定2×2均值（768维）
→ 拼成1152维 → **同一个Linear64** → L2。每格192通道作固定无仿射LayerNorm。
7×7到2×2使用adaptive average pooling，中心行/列会被相邻池化格共享；不是四块互不重叠的CCD探测器。
它处理的是完整L光电计算之后的特征，不能把这些位置称为未经混合的原始像素/CCD。
文本仍通过L路径和全token池化参与输出；无新的原图分支、TF或attention。
原Linear权重放前384列，后768列初始化0；新增49152参数，训练后真实推理也需要，不冒充零成本。
原光相位、六次10cm、478有效场、Top2、CCD读出、alpha>0.4和电子残差均不变。
本轮没有外部预训练/教师损失，单独隔离读出变化；不把它与转台方案混作一个对照。

先将下面命令改为epochs1、steps2、eval-every1，output改为`$T07/runs/smoke/abo_spatial_readout_20260913`。
CPU测试及完整CUDA初始化、反向、正常/去光评估通过后，才启动下面20轮正式命令。
训练批16、评估批4、TRAIN bank批16；只存best/last，测试选模偏差仍需披露。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# 先确认物理GPU1没有其他任务；不得抢占。不修改任何运行中的源码工作树。
nvidia-smi
CUDA_VISIBLE_DEVICES=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --multi-view --refine-profile sku_spatial_readout --lr-scale .2 \
  --epochs 20 --steps 100 --eval-every 5 --batch-size 4 --bank-batch-size 16 \
  --output "$R/abo200_spatial_readout_20260913"
```

## 74. 语言侧完整CCD读出：零新增参数的当前SKU协议对照

正式已完成：source `b9a1bb4b850dd328dfd7e400c8fe7f2ee5c3e91e`，原GPU1 RTX4090，PID2276631已退出。
最终epoch20 live正常76.75%/去光76.625%、TRAIN94.8125%，路由合格；不采用此候选。
冒烟已完成并释放PID2269583；新模式初始67.375%，两步后67.5%，不是与原模式的等价迁移。
以下为本轮复现命令，不要覆盖已有输出重复运行。

只把L expert/global的`readout_mode`从`prefix_rows`改为`fullfield_rows`。
旧：原强度处理→pool224×224→前77行；新：相同强度处理→整场pool77×224。
后续row LayerNorm/ReLU/Linear192不变，V端不动；router探测器读出及478光场/传播/相位/Top2完全不动。
这会改变数值函数；新run仅在新读出合同内选best，禁止回退旧模式并继续引用旧指标。
旧任务的fullfield结果只作历史证据，这次固定当前1600 TRAIN/gallery+800 QUERY和原d11f3428权重。
原linear64读出和全部电子残差保留，不叠加第73节空间头/卷积教师，也不叠加外部预训练。
同一20轮×100步、每5轮live/EMA、lr-scale=.2；原光学噪声/DC、专家资格约束、alpha>0.4保留。
强度汇聚仍有信息损失；不把未取的行数比例当作损失光能的比例，不声称该修改一定提高精度。

先epochs1、steps2、eval-every1，output=`$T07/runs/smoke/abo_fullfield_language_20260914`验证完整CUDA流程。
确认结束、相位/梯度有限和GPU释放后，才执行正式20轮。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# 先确认GPU1空闲；第72节正在使用GPU3，不抢占他人的GPU0。
nvidia-smi
CUDA_VISIBLE_DEVICES=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --multi-view --refine-profile sku_fullfield_language --lr-scale .2 \
  --epochs 20 --steps 100 --eval-every 5 --batch-size 4 --bank-batch-size 16 \
  --output "$R/abo200_fullfield_language_20260914"
```

## 75. 固定200商品协议的视角错误审计（CPU，不重新训练）

仅重读两个既有64维缓存，核对相同manifest SHA和全部2400个样本ID，按协议图库顺序稳定排序。
保留全部1600图库/800查询；按类别、原始spin索引和最近同SKU训练视角间隔统计，不拟合变换、不改标签。
索引周期72，不是角度数值；不同商品的相同索引也不保证语义上的正面/侧面一致。
分析使用TEST结果，不能当独立验证或据此删困难查询。输出只含JSON指标和逐查询预测，不产生权重。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
# 使用包含本审计代码的GitHub commit；不要切换运行中训练所用的工作树。
CUDA_VISIBLE_DEVICES='' python -m LightGenV2.tasks.t07_abo_image_retrieval.analysis.enrolled_views \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" \
  --optical-cache "$R/abo200_route_distill_20260913/features.pt" \
  --qwen-cache "$R/abo200_enrolled_qwen64_20260913/normal_features.pt" \
  --output "$R/abo200_view_audit_20260914"
```

已有输出时命令拒绝覆盖；新的模型缓存应使用新的有意义run ID，不能覆盖这次基准审计。

## 76. 只保留全图库检索数据目标的训练对照

完整CUDA冒烟已完成并释放PID2301761：两步后选epoch1 EMA，正常78.125%、去光74.875%，
路由合格，梯度6.05466有限，12份相位均更新；这不是新正式成绩。
正式`abo200_retrieval_only_20260914`已完成20轮×100步，PID2307838退出、GPU1释放。
最终epoch20 EMA正常78.375%、TRAIN96.5625%、去光76.375%，下降2pp、路由合格；
12相位均更新，best SHA=`869dd91c3e1b2ef1c578e88977e7c0e3ae6e40a6fb41492a7c65fe5273824585`。
较原参照只多2张查询，光学去除降幅也更小，不当作明显突破，不覆盖原权重。
源码`686a585f6467b6b4f93f9e86530fb7ec0bd8c4e0`，本地/服务器360测试通过，已推GitHub；
训练时工作树`.worktrees/t07_view_audit_20260914`保持固定。以下是复现命令，不要覆盖已有run重复启动。

`sku_retrieval_only`与`sku_capacity_control`相比仅关闭两个把同SKU不同视角拉近的辅助项：
live SupCon权重从.5到0，all-view log-probability权重从.1到0。
保留温度.1的全TRAIN图库多正例NLL，正例仍是相同SKU且排除查询自身，负例仍是全部其他SKU。
它鼓励分配概率给正确SKU的图库图，但不要求每个同SKU视角获得相近概率；不改变检索标签。
原光正则、Top2均衡、alpha>0.4、六次10cm/ROI/电子结构/linear64、EMA及温和增强均不变。
不加教师，不结合fullfield或分区头，仍从原78.125%权重出发、20轮×100步，与已完成轻量对照比较。
批次构成仍8查询+8独立参考图；新数据损失只反传8查询，另8参考图不再贡献SupCon，
但仍经过模型参与同批路由辅助项。记录此差别，不冒充16个检索查询。

复现时先确认已有任务退出/显存释放，再将下方改为epochs1、steps2、eval-every1，
output=`$T07/runs/smoke/abo_retrieval_only_20260914`做完整初始化/反向/正常去光验证。
通过且进程已结束后才执行正式命令。不得因为准备了命令就声称已启动。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# GPU1只是候选；先检查无其他进程。与现有任务合计保持至多两卡。
nvidia-smi
CUDA_VISIBLE_DEVICES=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --multi-view --refine-profile sku_retrieval_only --lr-scale .2 \
  --epochs 20 --steps 100 --eval-every 5 --batch-size 4 --bank-batch-size 16 \
  --output "$R/abo200_retrieval_only_20260914"
```

## 77. 外部先只训练光，目标集再联合微调

source `cf32e92939f23927dc717255c4849ab38549e819`已推GitHub，本地/服务器367测试通过。
1外部轮+1目标轮、各1步的完整CUDA冒烟已完成并退出PID2317638，
输出`runs/smoke/abo_optical_pretrain_20260914`：外部非光参数SHA未变，目标阶段恢复91份非光张量的实际更新，
frontend始终未变、103个Adam状态step均为1；12相位更新。最终选回初始78.125%/去光74.75%、路由合格，非新成绩。
正式36+20轮已经完成，PID2324757退出，`runs/simulation/abo200_optical_pretrain_20260914`。
最终epoch41=目标第5轮EMA正常81.25%、TRAIN93.875%、同权重去光75.875%；第80节固定best复评一致。
best SHA=`dcf768878abddd91558533757e404d9ee788cffbc5e0f0162a6f07d8a197eb1d`，GPU已释放。
工作树`.worktrees/t07_spin_training_20260913`运行时固定上述commit。以下只供复现，不要重复启动或覆盖已有run。

第72节的外部12轮是12×100步，不是12遍全数据。每步16张输入含8查询和8不同照片参考，
总呈现量19200张，约等于5436图的3.53遍；随机采样，并非每张图恰好出现相同次数。
联合预训练最终78.75%但去光78.125%，光学下降仅.625个百分点，因此**没有启动延长原联合预训练**。
新的`sku_optical_pretrain`先在外部只更新12份光学相位，电子残差、投影、读出、alpha全部冻结，
每轮核对非光参数SHA严格不变，电子dropout处于eval；原光噪声/DC仍按10%批次启用。
外部36×100步，总输入呈现量57600（约10.60遍）；expert/global相位基础LR=.002，router基础LR=.000006，
均再乘原warmup/余弦系数。这同时改变外部阶段参数范围、相位LR和预算，不宣称单因素归因。
进入目标集后恢复全部原可训练参数、清空Adam动量并重置EMA，仍20×100步；
expert/global基础LR恢复.0004，普通电子.00002、原读出.00006，router仍.000006。
冻结的Qwen紧凑前端始终不解冻。原轻量推理结构不变，不叠加第76节纯检索损失，不加教师或电子参数。
同一原78.125%起点，同一453-SKU外部池及1600/800目标协议；不是拿目标最后权重再次预训练。
这是光优先的阶段训练实验，余弦调度按36外部轮展开，不能声称等计算量。
外部不执行TEST选模；目标阶段重置Adam/EMA、每5轮择优并最终正常/去光复评。
GPU3为候选，必须确认第72节全部结束、进程退出和显存释放后再用，不抢占。
先使用第71节40图冒烟池及其SHA、外部1轮+目标1轮、各1步、eval-every1，
output=`$T07/runs/smoke/abo_optical_pretrain_20260914`验证：外部仅958728可训练参数、
非光SHA不变、目标恢复2782485参数、完整正常/去光与路由评估。通过并退出后才启动正式命令。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi
CUDA_VISIBLE_DEVICES=GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --external-pool "$R/abo_spin_pool50_views12_20260913" --external-root /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --expected-external-sha256 68fd35b6a2308f13e01963eb1233545c44eb07f5caa48ff655dc6fc302b1ed8f \
  --external-pretrain-epochs 36 --multi-view --refine-profile sku_optical_pretrain \
  --lr-scale .2 --epochs 20 --steps 100 --eval-every 5 --batch-size 4 --bank-batch-size 16 \
  --output "$R/abo200_optical_pretrain_20260914"
```

## 78. 分开验证数据增强和相位dropout（不叠加SAM）

源码`4185ee926fdfe030ecac4979c02145a7995eb843`已推GitHub，本地/服务器369测试通过。
增强组完整CUDA冒烟424.37秒完成，PID2344061退出，12相位更新且梯度有限；
最终选回初始78.125%/去光74.75%、路由合格。正式增强20轮也已完成，PID2350345退出、GPU1释放，
run=`abo200_augmentation_only_20260914`新候选最高77.375%，最终仍选初始78.125%/去光74.75%，不采用。
best SHA=`e945bff953c284b7a82a77408c1d88607841107cf1df4762684f146d56949c53`，不是新训练成绩。
独立Dropout已完成20轮×100步，PID2377609退出、GPU1释放，run=`abo200_dropout_only_20260914`。
新训练最高epoch20 EMA77.125%、TRAIN95.5625%；最终选回初始78.125%/去光74.75%，不采用。
best SHA=`1bcfa1784c75e1b5856e5ee3925f64ea0fc306d2f56866a61af87d7743726694`为初始状态重存。
工作树`.worktrees/t07_view_audit_20260914`运行时固定上述4185ee92源码，没有自动排队进程。
本次省略重复的全图库dropout冒烟：已核对`abo200_sam_control_20260913/final_report.json`，
同一个受保护光学实现SHA=`6490c6ee0ccc7501572fbae722aafbd7d4a016425d21af454019e7b60625433d`、
同样3%相位dropout已经在RTX4090完成CUDA训练；本次只拆分profile，不新增dropout算子，369回归通过。
这不是声称旧组合试验已证明dropout有效；新独立20轮及同权重去光也已完成，结果见上方。

两组均从原d11f3428权重开始，分别与已完成`abo200_capacity_control_20260913`比较。
固定全部1600训练图库/800查询、20轮×100步、每5轮评估live/EMA；TEST参与选模，存在选择偏差。
`sku_augmentation_only`只改变增强：完整物体缩小至85%～100%并在白色画布内平移，
亮度/对比度85%～115%，15%概率轻微高斯模糊；无裁剪、旋转或翻转，不丢物体部件。
`sku_phase_dropout_only`保持原95%～105%亮度/对比度，只增加expert/global训练相位3%的8×8块随机绕过；
不丢整专家、不置零振幅、不使用inverted-dropout增益、不对router施加dropout；eval关闭。
二者均不改六次10cm传播/ROI/Top2/alpha/电子容量/64维头/损失/AdamW；不额外加SAM。
从新代码所在的**空闲工作树**执行，不更新第77节运行中的工作树。只使用一张额外空闲卡，
先增强，结束并检查PID/显存释放后再运行dropout，合计至多两张我们的卡，不抢占他人任务。
新环境或首次修改算子时，先用对应profile执行epochs1、steps2、eval-every1、独立smoke输出，
核对正常/去光/有限梯度后再正式训练。本次dropout沿用已验证算子，采用上面的减少重复测试安排。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi
# 第二组只把下面PROFILE改为sku_phase_dropout_only，RUN改为abo200_dropout_only_20260914。
# 不要覆盖已存在的run；这里准备命令不等于实验已启动或已完成。
PROFILE=sku_augmentation_only
RUN=abo200_augmentation_only_20260914
CUDA_VISIBLE_DEVICES=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_route_distill_20260913/best.pt" \
  --expected-checkpoint-sha256 d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a \
  --multi-view --refine-profile "$PROFILE" --lr-scale .2 \
  --epochs 20 --steps 100 --eval-every 5 --batch-size 4 --bank-batch-size 16 \
  --output "$R/$RUN"
```

## 79. 单模型权重平均候选（不是预测集成，不占新训练卡）

**当前暂缓执行**：光优先预训练目标第5轮EMA已达到81.25%，优先等待最终正常/去光核验。
下面仅保留备选操作，尚未生成这个候选的best.pt或运行评估；不要把准备命令当作已完成试验。

只准备一个预先固定的50:50候选：外部联合预训练best（78.75%）与纯检索目标best（78.375%）。
二者均由原d11f3428继续训练，已在CPU核对metadata、状态键和冻结frontend完全一致，目标manifest相同。
全部可训练权重（包括原始相位参数）平均成一个普通模型；不拼接描述子，不平均两个预测，
不加推理分支/参数/拍摄次数，冻结前端保持逐位相同。原alpha边界、光学实现和ROI不改。
父权重已经用TEST选模，因此派生候选仍记录训练与选模历史；不冒充未训练迁移或独立验证。
平均权重没有继承任一父模型的准确率，必须重新完整正常/去光测试及路由检查。
先CPU构建；等现有GPU任务结束并确认释放后再运行复评，不额外占第三张卡。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
AVG="$R/abo200_average_spin_retrieval_20260914"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES='' python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.weight_average \
  --left "$R/abo200_spin_pretrain_20260913/best.pt" \
  --left-sha256 76ad086eba5114068b01e20b7926da0fcd2416ca54e99f64b273f289deee1d9b \
  --right "$R/abo200_retrieval_only_20260914/best.pt" \
  --right-sha256 869dd91c3e1b2ef1c578e88977e7c0e3ae6e40a6fb41492a7c65fe5273824585 \
  --right-weight .5 --output "$AVG"

# 从构建报告读取预先记录的SHA，不手抄；只在GPU1已空闲时执行，不覆盖已有evaluation。
SHA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' "$AVG/average_report.json")
nvidia-smi
CUDA_VISIBLE_DEVICES=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen optical \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$AVG/best.pt" --expected-checkpoint-sha256 "$SHA" \
  --batch-size 4 --output "$AVG/evaluation"
```

源码对平均时的数据来源增加核对：两父权重manifest不一致或仅一份缺失时拒绝；
保留父训练epoch/variant/来源commit及TEST选模标记，不继承分数、优化器或辅助头。
相位仍为`2*pi*sigmoid(raw)`；只保存这个候选的best.pt，未重新训练所以没有last.pt。

## 80. 光优先最终best独立复评（已完成）

实际执行源码`2ee704e1cffbb54c6052afe83c0f131fdce5a47e`，本地/服务器377项回归通过。
PID2399537已退出、GPU3释放；`verification/final_report.json`状态complete，正常81.25%、去光75.875%，
800 TEST和1600 gallery分别通过路由检查，与主训练报告精确一致。best SHA为第77节的dcf76887全文。
报告SHA=`f28c7f71240079e2bb7479a6603f5102cd54b1076e9f1bf82a4cb6cba0bdf65c`。

训练阶段`normal.router`统计包含1600图库+800查询；最终核验不能仅凭合并统计判断测试侧均衡。
本复评从原图分别重建正常/同权重去光特征，正常捕获每张图的离散Top2，按gallery/query分别汇总。
`final_report.json`的`routing.normal.splits.query`是800张TEST自身统计；`gallery`是1600图库。
每侧至少3种组合、最大组合份额≤80%、每专家选择份额≥5%，沿用现有阈值但分开核查。
份额分母是两次选择×样本数，合计1；不是每样本激活概率（后者合计2）。
`routing.remove_optical.executed=false`，绝不拿移除前的router缓存冒充新的观测。
`normal_features.pt`额外存`router_selected_mask`，可逐样本复核；无额外前向、无随机数或模型参数改变。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
RUN="$R/abo200_optical_pretrain_20260914"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# 先确认原训练PID退出、GPU3空闲，不能边改best文件边复评。
nvidia-smi
SHA=$(python -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="complete"; print(r["best_sha256"])' "$RUN/final_report.json")
CUDA_VISIBLE_DEVICES=GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen optical \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$RUN/best.pt" --expected-checkpoint-sha256 "$SHA" \
  --batch-size 4 --output "$RUN/verification"
```

已有verification时拒绝覆盖；这是固定权重复评，不是重新训练，也不是新的独立测试集。
完整parent训练/TEST选模偏差仍需披露。重新执行请使用新的空output，不能覆盖这次已完成的证据。

## 81. 给老师/同学的两个冻结Qwen baseline复现

说明及解压后可独立运行的命令见[baseline_reproduction/README.md](baseline_reproduction/README.md)。
旧版是480查询/120商品中心的同类别检索；新版是800查询/1600单图的同SKU检索，不能把分数下降称为微调过拟合。
两个模式均零训练参数、无光学/旧工程依赖，先用CPU核算固定缓存，再在一张空闲GPU上顺序从原图复评。
实际本轮run为`baseline_reproduction_20260914_legacy_category`和`baseline_reproduction_20260914_enrolled_sku`，
再次执行必须另选output。源码与说明通过Git同步，不SCP覆盖工程源码。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
B=LightGenV2/tasks/t07_abo_image_retrieval/baseline_reproduction
DATA=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
MODEL=/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python "$B/baseline.py" audit --protocol legacy_category --manifest "$DATA/data/abo_similarity10_manifest.csv" --features "$R/frozen_qwen_20260912/features.pt" --expected-features-sha256 38f77637e48cf28fb7b1e077119bcf4c1ea37dd3480cbd1b6ce5707f05b38324 --output "$T07/runs/smoke/baseline_legacy_cache_20260914"
python "$B/baseline.py" audit --protocol enrolled_sku --manifest "$DATA/data/abo_similarity10_manifest.csv" --enrolled-manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --features "$R/abo200_enrolled_qwen64_20260913/normal_features.pt" --expected-features-sha256 c6eb631c268d2446a2f783854c86d8493cdbcaa04c0669148b16d9785016d8d7 --output "$T07/runs/smoke/baseline_enrolled_cache_20260914"
nvidia-smi
# 仅当GPU0空闲才使用；否则选另一张真正空闲的GPU。
CUDA_VISIBLE_DEVICES=0 python "$B/baseline.py" infer --protocol legacy_category --manifest "$DATA/data/abo_similarity10_manifest.csv" --data "$DATA" --model "$MODEL" --output "$R/baseline_reproduction_20260914_legacy_category"
CUDA_VISIBLE_DEVICES=0 python "$B/baseline.py" infer --protocol enrolled_sku --manifest "$DATA/data/abo_similarity10_manifest.csv" --enrolled-manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --data "$DATA" --model "$MODEL" --output "$R/baseline_reproduction_20260914_enrolled_sku"
python "$B/package.py" --runs "$R" --data "$DATA" --output "$T07/releases/abo_qwen_baseline_reproduction_20260914.zip"
```

此小ZIP只含独立源码、说明、原始特征/名单/报告及可用的本次独立复评结果，不含图像、4.26GB Qwen模型或硬件功能。
打包器白名单读取Git已提交源码；`PACKAGE_MANIFEST.json`记录逐文件SHA。原图和模型按说明另行提供。

本次已从原图顺序重跑完成（推理源码91b2d02d，GPU0 RTX4090）：
旧native64=94.375%、2048=95.2083%；新native64=85.125%、2048=85.625%。
新64维同一排名按同类别判定为98.75%，仅为诊断。PID3428835/3438359已退出，禁止用此诊断替换同SKU主指标。

## 82. Vision外层跳连和V/L光贡献：固定权重诊断

已执行完成，源码025f3223，PID1059479退出。正常81.25%、去外层skip53.50%、仅skip67.125%、
去V光74.875%、去L光77.625%、全去光75.875%。报告SHA
`104540e3b6d4cbf002adccc6ea192edb1848e3ab9e530971b7b05eaf8c178e40`；再次执行须换一个不存在的output。

从已提交/推送的本任务代码所在工作树执行，先激活xml并确认GPU0空闲；下方命令不是运行成功声明。
正常81.25%必须首先重现，否则脚本停止，不解释后续消融。不会生成或修改模型权重。
保留原始相位/光路/ROI，所有干预仅在诊断进程临时hook中生效，结束自动清除。
`skip_only`只去掉Vision光电更新，Language光电仍在；该诊断不是单独训练的纯电子baseline。
同时重新计算图库和查询，报告TRAIN图库/TEST查询各自的特征RMS统计；不把特征幅值当光能。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi
CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.outer_skip_audit \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_optical_pretrain_20260914/best.pt" \
  --expected-checkpoint-sha256 dcf768878abddd91558533757e404d9ee788cffbc5e0f0162a6f07d8a197eb1d \
  --expected-hit-at-1 .8125 --batch-size 4 --output "$R/abo200_outer_skip_audit_20260914"
```

## 83. 更频繁刷新训练图库：ABO主线＋SHAPE备选

本轮已启动（025f3223；本地/服务器385测试）：ABO GPU0 PID1100080，SHAPE GPU1 PID1100081。
两组各15×100，尚无新性能；正常最佳仍分别81.25%/78.8732%。先前CUDA冒烟已complete，
PID1060863退出并释放GPU1。不要重复执行下方命令覆盖正在运行的目录；它们用于身份追溯或新目录复现。

先做CUDA冒烟：把ABO下方输出改为`$T07/runs/smoke/abo_bank_refresh_20260914`，
设置`--epochs 1 --steps 2 --eval-every 1 --bank-refresh-steps 1`，其余参数不变。
确认`history.json`记录一次requires_grad=false的刷新、梯度有限、完整正常/去光评估完成。
正式运行重新从原best开始，不从冒烟best开始。禁止覆盖已有输出。

`--batch-size 4`是评估batch；训练为8商品×2张不同TRAIN照片＝16张。
`--bank-batch-size 16`只影响训练图库编码；`--bank-refresh-steps 25`不表示加入新图或测试图。
两条命令可分别在已确认空闲的GPU0/1运行，最多两张；没有空闲卡则等待，不终止他人进程。
使用短程低学习率续训；不声称与历史run构成单一图库刷新因素的严格对照。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi
CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_optical_pretrain_20260914/best.pt" \
  --expected-checkpoint-sha256 dcf768878abddd91558533757e404d9ee788cffbc5e0f0162a6f07d8a197eb1d \
  --multi-view --refine-profile sku_capacity_control --lr-scale .1 \
  --epochs 15 --steps 100 --eval-every 5 --batch-size 4 --bank-batch-size 16 --bank-refresh-steps 25 \
  --output "$R/abo200_bank_refresh_20260914"
CUDA_VISIBLE_DEVICES=GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/shape_source \
  --manifest "$R/shape8_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/shape8_view_refine_20260913/best.pt" \
  --expected-checkpoint-sha256 45fcfae4a970e2a14d518b9ecebaf4d5e927c5f48f653181abf87c398487c324 \
  --multi-view --refine-profile shape_views --lr-scale .25 \
  --epochs 15 --steps 100 --eval-every 5 --batch-size 4 --bank-batch-size 16 --bank-refresh-steps 25 \
  --output "$R/shape8_bank_refresh_20260914"
```

完成后检查`status.json`、`final_report.json`的正常/同权重去光、TRAIN和专家分布；
若best选回epoch0，则没有获得新训练提升。检查自身PID退出并确认GPU释放，不以历史启动PID当运行状态。

## 84. ABO固定协议冲刺83%：阶段再训练与双向图库监督

2026-09-14：第83节两组已结束。ABO仍81.25%，SHAPE80.2817%。
下面两组都从原ABO81.25%开始，不采用未改善的last。最多两卡；启动前检查空闲，
示例GPU4 RTX4090及GPU5 RTX3090。不得抢占别人的GPU0–3。
源码必须先测试、GitHub同步。正常推理仍原六次传播/光Top2/alpha>=.4001/64维，
无新增电子参数。仅best/last，不周期保存mask。目标83%不是保证。

先对B运行CUDA冒烟：改为`--epochs 1 --steps 2 --eval-every 1 --bank-refresh-steps 1`，
output改`$T07/runs/smoke/abo_symmetric_bank_20260914`。确认两方向自排除、有限梯度和
完整正常/去光评估，然后确认进程退出再启动正式B，不另占第三卡。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi
# A：同一已审计外部池，24轮只训光；接着目标10轮联合微调，每轮评估live/EMA。
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_optical_pretrain_20260914/best.pt" \
  --expected-checkpoint-sha256 dcf768878abddd91558533757e404d9ee788cffbc5e0f0162a6f07d8a197eb1d \
  --external-pool "$R/abo_spin_pool50_views12_20260913" --external-root /DATA/DATA1/guest3/2026OpticsMoE/data/abo \
  --expected-external-sha256 68fd35b6a2308f13e01963eb1233545c44eb07f5caa48ff655dc6fc302b1ed8f \
  --external-pretrain-epochs 24 --multi-view --refine-profile sku_optical_pretrain \
  --lr-scale .2 --epochs 10 --steps 100 --eval-every 1 --batch-size 4 --bank-batch-size 16 \
  --output "$R/abo200_optical_reheat_20260914"
# B：两个TRAIN视角都作为图库查询，保留mean损失尺度；目标20轮，每轮评估。
CUDA_VISIBLE_DEVICES=GPU-d53ce4c8-272d-c2fb-dc09-f182d586c4eb python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_optical_pretrain_20260914/best.pt" \
  --expected-checkpoint-sha256 dcf768878abddd91558533757e404d9ee788cffbc5e0f0162a6f07d8a197eb1d \
  --multi-view --refine-profile sku_symmetric_bank --lr-scale .1 \
  --epochs 20 --steps 100 --eval-every 1 --batch-size 4 --bank-batch-size 16 --bank-refresh-steps 25 \
  --output "$R/abo200_symmetric_bank_20260914"
```

A沿用原逐轮图库，不叠加B的双向损失；B与第83节同起点/学习率/刷新间隔，
但预算20而非15轮、评估频率1而非5轮，所以最终最佳差异不能完全归因于损失。
`batch_natural_hit_at_1`在B为两方向均值（训练诊断，不是TEST）。
TEST定期选模按用户约定，须明确不是独立无偏测试。失败保留81.25%，不覆盖原正式权重。

## 85. CPU校准现有64维读出（必须完整GPU复核后才能报告提升）

不更新正在训练的worktree。采用另一个干净、已同步GitHub的源码worktree运行；
仅CPU4线程，不额外占卡。输入为第80节完成独立复评的原81.25%模型及其normal/remove缓存。
校验原图清单SHA、缓存ID顺序和source checkpoint；800查询从不进入优化函数。
训练一个64x64临时矩阵A：同SKU多正例NLL(temp=.1)+1.0*||A-I||F²/64，
batch128、800步，lr=.001余弦衰减，每50步按缓存TEST Hit@1/mAP择优。
这是用户接受的TEST择优实验，不是独立测试。gallery与query仍按原余弦逐图排序，不改SKU相关性定义。
最终A合并进既有384→64权重和bias，不引入新的推理层。best.pt/last.pt均为合并后的完整模型。
光学参数不改，因此此项不能宣称增加了光学学习；同权重去光必须用同一个A，不能另拟合。

```bash
T07=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval
R="$T07/runs/simulation"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout \
  --source-run "$R/abo200_optical_pretrain_20260914" \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --expected-checkpoint-sha256 dcf768878abddd91558533757e404d9ee788cffbc5e0f0162a6f07d8a197eb1d \
  --expected-hit .8125 --steps 800 --eval-every 50 --batch-size 128 --lr .001 --anchor 1 \
  --output "$R/abo200_metric_readout_20260914"
```

`final_report.json`只记录cached_candidate，不是部署精度；只有候选优于原值才进一步用
第80节完整原图评估入口验证（换checkpoint、实际SHA和新的output，不覆盖原verification）。
即使缓存达到83%，也不得直接宣布目标完成。核对合并前后除readout.projection两张量外全部相同。

已完成补充对照：同一命令仅改`--anchor .1`及
`--output "$R/abo200_metric_weak_anchor_20260914"`，缓存最佳81.875%，原anchor1为81.75%。
两组均complete、CPU进程已退出，没有新占GPU。弱锚定best SHA
`a579413726de92c22000f3379c185df83e604860990e57010a5bb39259aeb482`，后续完整复评见下方。
此工具PT的`stage=readout_metric_fit`、`epoch`记录优化step，不能解释为350轮原图训练；
phase/alpha/前端全冻结，确实只改变原head的weight/bias；初筛时尚未晋升正式结果。

弱锚定候选随后已完成原图4090复评：81.875%、去光75.50%，以下为实际成功命令。
GPU4当时只有自有训练占约2.1GB/24GB，故添加只读batch4复评共享同卡，不占第三卡；
不把并发运行用时作为基准。复评期间不写训练run，不改其源码/权重。
第一次手抄SHA长度有误而失败于加载前，`verification/`保留；成功目录为`verification_4090/`。
下例从完成报告读取SHA并核对实际文件，不需要人工复制长串，也不跳过校验。

```bash
RUN="$R/abo200_metric_weak_anchor_20260914"
METRIC_SHA=$(python -c 'import json,sys,pathlib,hashlib; p=pathlib.Path(sys.argv[1]); s=json.loads((p/"final_report.json").read_text())["best_sha256"]; assert len(s)==64 and hashlib.sha256((p/"best.pt").read_bytes()).hexdigest()==s; print(s)' "$RUN")
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen optical \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$RUN/best.pt" --expected-checkpoint-sha256 "$METRIC_SHA" \
  --batch-size 4 --output "$RUN/verification_4090"
```

该output已经存在，再现时请用空的新目录，禁止覆盖已完成报告。完整指标/原始向量/路由都在成功目录。

## 86. 冻结光电主干，直接校准原384→64读出（零新增推理参数）

第85节的A只在已投影的64维子空间内调整。本对照直接拟合已有Linear的weight/bias，
可重新选择384维中的方向；LayerNorm、前端、全部光/电残差、alpha、Top2不变。
不是新增384维检索输出：最终仍为L2归一化64维、逐图余弦排序。仍不使用query拟合。
先从81.875%的固定权重原图前向，通过只读hook记录原head归一化之后、Linear之前的384维输入。
同一缓存目录同时包含正常/去光输入、64维输出、图像ID、checkpoint/manifest SHA与路由。
缓存hook不改变推理；正式提升仍需从原图独立复评，不以FP32缓存分数代替BF16原图分数。

在GitHub已同步的干净源码worktree运行，不能更新正在训练的worktree。先核验第85节METRIC_SHA，
并设置T07/R。GPU UUID是本次自有GPU4；其他机器需要先查看空闲卡再替换，禁止占用别人的卡。

```bash
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen optical \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_metric_weak_anchor_20260914/best.pt" --expected-checkpoint-sha256 "$METRIC_SHA" \
  --batch-size 4 --cache-readout-input --output "$R/abo200_readout384_cache_20260914"

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout \
  --source-run "$R/abo200_metric_weak_anchor_20260914" --verification-dir "$R/abo200_readout384_cache_20260914" \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --expected-checkpoint-sha256 "$METRIC_SHA" --expected-hit .81875 \
  --fit-space projection384 --steps 800 --eval-every 50 --batch-size 128 --lr .0001 --anchor 1 \
  --output "$R/abo200_direct_readout_20260914"
```

CPU只拟合1600 TRAIN、self-excluded同SKU多正例NLL(temp=.1)，另加相对起点权重的平方偏离约束：
`(||W-W0||²+||b-b0||²)/(||W0||²+||b0||²)`，系数1。每50步按TEST Hit@1/mAP选择，有选择偏差。
仅保存best.pt/last.pt两份完整模型；记录TRAIN命中率。部署时不用缓存、A或其他额外计算层。
成功候选按第85节原图复评入口核验正常/去光、TRAIN、专家均衡和alpha之后才能晋升。

2026-09-14实际执行：源码`77f6aaeba05d57fd90d164a23c49d0acfa616f21`，本地/服务器各399测试通过。
缓存run完成并保持原图正常81.875%/去光75.50%完全一致；PID1664187已退出、释放CUDA。
随后CPU读出run完成800步，PID1666689退出：缓存最高仍81.875%，末步81.375%，
TRAIN由94.75%升至95.5625%。没有改善主指标，保留原81.875%正式权重，不把该候选晋升，
不追加无意义的原图复评。best/last与历史均保留用于判断本方向，未增加推理容量。

训练正则对照：复用同一缓存与起点，将第86节第二条命令添加`--input-dropout .1`，
output改为`abo200_direct_readout_dropout_20260914`，其余800步/lr.0001/anchor1保持一致。
query与gallery各自随机屏蔽10%输入维度，只作用于TRAIN loss；评估、保存模型和原图推理无dropout。
它是电子读出训练正则，不是更改CCD/相位，也不能仅由该项宣称物理鲁棒性增强。

### 第86节实际结果与当前82.50%复评命令

所有对照均同一81.875%起点/同一缓存、800步、每50步TEST选模、batch128、seed42。
源码`3c0ff5455014d7f51c70abf40cbdd806f015829f`，本地/服务器400测试通过。
改变拟合命令中的output及下表参数即可复现；每个run都有完整execution.json、history.json、best/last。

| run后缀（前缀abo200_direct_readout） | dropout | anchor | lr | 缓存最高Hit@1 | 原图正常/去光 |
|---|---:|---:|---:|---:|---|
| _20260914 |0|1|.0001|81.875%|未晋升、不重复复评|
| _dropout_20260914 |.1|1|.0001|82.25%|82.25%/76.25%|
| _dropout20_20260914 |.2|1|.0001|81.875%|未晋升|
| _dropout_weak_20260914 |.1|.1|.0001|82.375%|**82.50%/76.25%**|
| _dropout_lr2_20260914 |.1|.1|.0002|82.25%|未超过现有最佳、不复评|

所有CPU拟合均已完成退出；光学再加热后续完成，结果见第87节，GPU4已释放。
双向bank旧组完成10轮后因连续9轮未改善主动停止，GPU5释放，记录在该run的early_stop_report.json，
不伪称完整20轮训练，不覆盖原训练器的failed_or_interrupted状态。

当前best是weak组第450步：SHA `b1e205c70505de9df77ea52bed9c962bebacd3171e9f7b55b0569ec25f9fafd6`。
自动取SHA并做完整原图复评（output已存在，重现须使用新的空output目录）：

```bash
RUN="$R/abo200_direct_readout_dropout_weak_20260914"
WEAK_SHA=$(python -c 'import json,sys,pathlib,hashlib; p=pathlib.Path(sys.argv[1]); s=json.loads((p/"final_report.json").read_text())["best_sha256"]; assert len(s)==64 and hashlib.sha256((p/"best.pt").read_bytes()).hexdigest()==s; print(s)' "$RUN")
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen optical \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$RUN/best.pt" --expected-checkpoint-sha256 "$WEAK_SHA" \
  --batch-size 4 --cache-readout-input --output "$RUN/verification"
```

完整报告SHA `eb0901a854f6ac9bddefcfeae00ea4f15a52d94a1f7046c645173d191ee0cafc`，
weight_train_audit.json记录只有head weight/bias改变、原图TRAIN94.375%(1510/1600，自图排除)、
alpha与gallery/query分别合格的Top2路由；原图正常660/800、去光610/800，尚差4张达到83%。
400/450是优化step不是原图epoch。全部是TEST择优的单seed结果，不能宣称独立验证或统计显著。

## 87. 完成光学再加热后，固定新主干校准读出（2026-09-15）

`abo200_optical_reheat_20260914`已完整结束24外部+10目标轮，源码efd7dd55；
最佳总epoch29（目标第5轮EMA），正常82.125%/去光76.00%、TRAIN94.4375%，
external_frozen_electronics_verified=true；PID1361629已退出、CUDA释放。
该组不是当前82.50%的替换版；下面验证新相位是否能与第86节有效的读出正则结合。
仍是同一200SKU/1600图库/800查询、6次10cm、光Top2、alpha>.4；不增加推理层。
原GPU4后来被别人占用，以下实际改用空闲GPU0的RTX4090，不干扰其他进程；仅短暂原图缓存占GPU。

```bash
REHEAT_SHA=$(python -c 'import pathlib,json,hashlib,sys; p=pathlib.Path(sys.argv[1]); r=json.loads((p/"final_report.json").read_text()); s=r["best_sha256"]; assert r["status"]=="complete" and len(s)==64 and hashlib.sha256((p/"best.pt").read_bytes()).hexdigest()==s; print(s)' "$R/abo200_optical_reheat_20260914")
CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen optical \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_optical_reheat_20260914/best.pt" --expected-checkpoint-sha256 "$REHEAT_SHA" \
  --batch-size 4 --cache-readout-input --output "$R/abo200_reheat_readout_cache_20260915"

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout \
  --source-run "$R/abo200_optical_reheat_20260914" --verification-dir "$R/abo200_reheat_readout_cache_20260915" \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --expected-checkpoint-sha256 "$REHEAT_SHA" --expected-hit .82125 \
  --fit-space projection384 --steps 800 --eval-every 50 --batch-size 128 --lr .0001 --anchor .1 --input-dropout .1 \
  --output "$R/abo200_reheat_readout_dropout_20260915"
```

起点SHA `ae0d8c9214c5f8d70070ea9fe94202266f583fe05611b26e9b2c760941f55525`；
新GPU0已从原图复现正常82.125%/去光76.00%，缓存PID1737263退出。
读出拟合使用GitHub已有源码3c0ff545（400测试），只使用TRAIN；候选必须重新读取原图正常/去光复核。

实际完成：800步缓存最佳82.50%，在同一GPU0重新读取原图后仍82.50%/去光76.25%。
候选SHA `a7fae6bfe41f3d27e3e47bf0fb6d63f5ab5d46e0841d41e2f3c58c27f55adcb9`；
原图报告在`abo200_reheat_readout_dropout_20260915/verification/`，GPU复评PID1744107退出。
与旧相位的82.50%并列，不宣称已经达到83%。补充旧相位dropout=.05/anchor=.1对照缓存82.375%，未晋升。

## 88. 两个近邻checkpoint离线平均成单个模型（不是推理集成）

两个父模型共享同一光优先训练起点、冻结前端、网络metadata和原200SKU协议；
一个为旧相位+读出dropout弱锚定，另一个为再加热相位+同类读出校准，原图均82.50%。
先固定50/50，所有可训练浮点参数（含raw phase和alpha参数）离线平均；
冻结前端要求逐值相同，不做平均。相位仍由2*pi*sigmoid(raw)产生，同一6次10cm光路。
最终只加载一份best.pt，推理计算量/Top2/64维输出不变；这是已有模型权重的派生，非未训练baseline。
仍有TEST选模偏差。不得将父模型各跑一遍再融合预测而冒称该方案。

```bash
REHEAT_HEAD_SHA=$(python -c 'import pathlib,json,hashlib,sys; p=pathlib.Path(sys.argv[1]); s=json.loads((p/"final_report.json").read_text())["best_sha256"]; assert len(s)==64 and hashlib.sha256((p/"best.pt").read_bytes()).hexdigest()==s; print(s)' "$R/abo200_reheat_readout_dropout_20260915")
# WEAK_SHA是第86节旧相位82.50%模型的SHA，不是81.875%起点的METRIC_SHA。
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.weight_average \
  --left "$R/abo200_direct_readout_dropout_weak_20260914/best.pt" --left-sha256 "$WEAK_SHA" \
  --right "$R/abo200_reheat_readout_dropout_20260915/best.pt" --right-sha256 "$REHEAT_HEAD_SHA" \
  --right-weight .5 --output "$R/abo200_reheat_readout_average_20260915"

AVERAGE_SHA=$(python -c 'import pathlib,json,hashlib,sys; p=pathlib.Path(sys.argv[1]); s=json.loads((p/"average_report.json").read_text())["checkpoint_sha256"]; assert len(s)==64 and hashlib.sha256((p/"best.pt").read_bytes()).hexdigest()==s; print(s)' "$R/abo200_reheat_readout_average_20260915")
CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen optical \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_reheat_readout_average_20260915/best.pt" --expected-checkpoint-sha256 "$AVERAGE_SHA" \
  --batch-size 4 --cache-readout-input --output "$R/abo200_reheat_readout_average_20260915/verification"
```

构建源码3c0ff545，average_report.json记录两个父权重SHA、派生口径和结构审计；
原图normal/remove、TRAIN自图排除、gallery/query分开路由和alpha审计全部通过后才能晋升。

50/50实际原图81.875%/去光76.25%，未超过两个82.50%父模型，不采用。
补充只读诊断：旧82.50%模型TRAIN均值向量范数.0988，平均向量范数3.839；
按TRAIN均值进行部分/全部去中心化均未改善缓存82.375%，未产生新模型；这不是CCD暗背景校正。

## 89. 只训相位+原线性读出，冻结其他电子（403项本地测试）

新profile `sku_phase_head`从第86节原图82.50%权重继续，绝不从冒烟last继续正式训练。
仅12份相位与readout.projection.weight/bias共14张量、983368参数参与优化；
frontend、所有电子残差、alpha、head LayerNorm全部冻结，并每轮/最终SHA断言。
固定电子保持eval模式，不添加电子dropout；只在Linear输入上采用训练期10%dropout，推理关闭且无新增层。
完整forward仍经过原光学router和专家/global，光正则、路由均衡、TRAIN-bank NLL/SupCon/all-view loss不变。
沿用10%训练batch的原metadata噪声/DC（router不加噪声）、微弱亮度/对比度增强，不增加像素扰动。
本方案是相位+读出联合训练，不是外部预训练，也不是完全冻结相位的缓存拟合。

先在确认空闲的GPU上做两步完整冒烟（启动前查看nvidia-smi，下面GPU0仅为本次机器）：

```bash
CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_direct_readout_dropout_weak_20260914/best.pt" --expected-checkpoint-sha256 "$WEAK_SHA" \
  --multi-view --refine-profile sku_phase_head --lr-scale .1 \
  --epochs 1 --steps 2 --eval-every 1 --batch-size 4 --bank-batch-size 16 \
  --output "$T07/runs/smoke/abo_phase_head_20260915"
```

必须确认冒烟complete、frozen_except_phase_projection初末SHA相同、14张量真实更新、正常/去光/路由评估通过，
且冒烟进程已退出释放显存，再用同一命令将epochs改12、steps改100、添加`--bank-refresh-steps 25`，
output改`$R/abo200_phase_head_20260915`启动正式对照。其他设置与起点保持一致。
最大学习率：expert/global .0002、router .000003、readout .00009；仍2轮warm-in与余弦衰减、EMA .99。
每轮评估TRAIN/TEST并按既定TEST+路由门槛保存best/last，只有普通单模型，目标83%未自动保证。

实际GPU冒烟已complete（389.22秒），scope_audit.json核验14张量更新、其余参数逐值不变、
冻结初始/每轮/最终SHA一致；梯度范数4.6033，12份相位均有非零变化。
两步后的best选回epoch0（82.50%/去光76.25%），不是新的训练收益。
PID1763945已退出释放CUDA后，正式12轮从原b1e205c7权重启动，源码固定ff5b44e3；
仅使用GPU0 RTX4090。完整正式命令（目录已使用，复现实验请换新的空output）：

```bash
CUDA_VISIBLE_DEVICES=GPU-afc19890-6209-ee4d-622d-e619da5bd5b2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_direct_readout_dropout_weak_20260914/best.pt" --expected-checkpoint-sha256 "$WEAK_SHA" \
  --multi-view --refine-profile sku_phase_head --lr-scale .1 \
  --epochs 12 --steps 100 --eval-every 1 --batch-size 4 --bank-batch-size 16 --bank-refresh-steps 25 \
  --output "$R/abo200_phase_head_20260915"
```

## 90. TRAIN最近正确/错误商品排序损失（CPU读出对照）

保持原384→64 Linear、六次光传播、Top2、alpha和全部主干。只改变TRAIN拟合目标：
`mean(softplus((max_wrong_cosine - max_correct_cosine + .02)/.1))`。
正确集为同SKU的其他7张TRAIN图；错误集为不同SKU的TRAIN图；自身从两者排除。
优化最近正确图排在最近错误图之前，仍加原权重相对平方偏离约束.1和10%独立TRAIN特征dropout。
这不是测试时按标签重排，也不是增加检索头层数；QUERY仅每50步评估选模（有选择偏差）。
从已验证82.50%权重开始，先只做一组800步、lr .00005、batch128、seed42。
必须在独立、已经同步GitHub的新worktree运行，不能更新第89节正在训练的ff5b44e3 worktree。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout \
  --source-run "$R/abo200_direct_readout_dropout_weak_20260914" \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --expected-checkpoint-sha256 "$WEAK_SHA" --expected-hit .825 \
  --fit-space projection384 --ranking-loss top1_softplus --input-dropout .1 \
  --steps 800 --eval-every 50 --batch-size 128 --lr .00005 --anchor .1 \
  --output "$R/abo200_readout_top1_20260915"
```

入口先核验2400个缓存ID/固定manifest/源权重SHA/已完成原图报告；仅前1600 TRAIN张量进loss。
缓存候选不算正式结果；只有重新原图编码并通过正常/去光/路由/alpha/权重变更审计后才可晋升。
本对照不影响第89节单GPU相位训练，结束后不保留CPU后台任务。

实际已完成：源码737f9303，本地/服务器各406测试。缓存best第450步82.625%，原图复评只有82.50%，
仍未达到83%；mAP=.7304591394、NDCG=.7873025540，Hit同分按mAP优于第86节候选。
TRAIN自图排除95.00%，去光75.875%，下降6.625pp；12份相位/alpha/电子主干均不改变，
只head两张量更新，gallery/query路由分别合格。CPU PID1789346和GPU复评PID1791190均退出释放。
完整原图复评命令（先确认GPU空闲；以下output已用，重现请换新output）：

```bash
TOP1_SHA=$(python -c 'import pathlib,json,hashlib,sys; p=pathlib.Path(sys.argv[1]); s=json.loads((p/"final_report.json").read_text())["best_sha256"]; assert hashlib.sha256((p/"best.pt").read_bytes()).hexdigest()==s; print(s)' "$R/abo200_readout_top1_20260915")
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen optical \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_readout_top1_20260915/best.pt" --expected-checkpoint-sha256 "$TOP1_SHA" \
  --batch-size 4 --cache-readout-input --output "$R/abo200_readout_top1_20260915/verification"
```

### 第90节固定种子复核（不是集成）

预先固定补充seed17/73，保留原seed42结果；均从第86节相同b1e205c7权重/缓存开始，
不是从本节新10925d29权重递进。其他800步、lr.00005、anchor.1、dropout.1、batch128设置不变。
在第90节CPU拟合命令加`--seed 17`或`--seed 73`，output分别改为
`abo200_readout_top1_seed17_20260915`和`abo200_readout_top1_seed73_20260915`，按顺序运行，不占GPU。
每组均只TRAIN参与梯度；保留best/last/完整history，不能隐去不理想的种子。
先看完整三组缓存结果，再对候选原图复评；不得将三个模型预测合并或平均成推理集成。
这些种子共享已择优的预训练起点，不能当成从头独立三次训练，也不能用其中最好值冒充均值。

两组已完整结束并原图复评：seed17缓存82.75%/原图82.625%/去光76.625%/TRAIN94.6875%，
seed73缓存82.75%/原图82.75%/去光76.50%/TRAIN94.75%。seed42原图82.50%，三组算术均值82.625%。
seed73 best第150步，SHA b14a34ea12aad01a305d27596e4e47edade5610fae2e24190f452c7034b68cbe，
原图报告SHA b632cd9e4b32ca2af1994082ffea20085b7ce82f452761233a7604b01deb34b2c。
两组weight_train_audit.json均确认只有原head两张量变化，所有其他参数/metadata相同；分split路由合格。
CPU PID1799262/1800500和复评GPU4 PID1802144/1804390均退出，不再占GPU；83%尚差2张。

后续仅一组低步长抛光：仍用第90节CPU命令，source-run改为`abo200_readout_top1_seed73_20260915`，
SHA自动核验上述b14a34ea权重，expected-hit改`.8275`、lr改`.00001`、anchor改`1`、seed固定42，
output为`abo200_readout_top1_polish_20260915`；其他配置保持800步/dropout.1/排序损失/batch128。
此为从新的已验证起点继续训练，不计入前述三种子重复实验；未原图复评前仍不是正式新成绩。

抛光组已完成并复评：原图82.75%/去光76.50%，mAP=.7295324281、NDCG=.7867065295，
未超过seed73的同Hit与更高mAP，故不替换。CPU PID1808391与复评GPU4 PID1811287均已退出。
原图证据在该run的verification/final_report.json；不把缓存mAP的提升当成完整模型提升。

## 91. 固定混合TRAIN损失，不增加读出参数

从当前原图82.75%的seed73权重继续，原head的TRAIN目标改为固定
`0.5 * 多正样本NLL + 0.5 * 最近正确/错误SKU softplus`，不搜索混合系数。
两项都使用同一独立dropout后的TRAIN query/gallery，温度.1，Top1余弦margin .02，自图排除。
冻结原光学/电子主干，原384→64头尺寸不变。只做一组800步/seed42/lr.000025/anchor.1；
此为后续目标微调，不算前一节的固定种子复核。

```bash
S73_SHA=$(python -c 'import pathlib,json,hashlib,sys; p=pathlib.Path(sys.argv[1]); s=json.loads((p/"final_report.json").read_text())["best_sha256"]; assert hashlib.sha256((p/"best.pt").read_bytes()).hexdigest()==s; print(s)' "$R/abo200_readout_top1_seed73_20260915")
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout \
  --source-run "$R/abo200_readout_top1_seed73_20260915" \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --expected-checkpoint-sha256 "$S73_SHA" --expected-hit .8275 \
  --fit-space projection384 --ranking-loss hybrid_nll_top1 --input-dropout .1 \
  --steps 800 --eval-every 50 --batch-size 128 --lr .000025 --anchor .1 --seed 42 \
  --output "$R/abo200_readout_hybrid_20260915"
```

仍只有TRAIN参与loss，周期TEST/跨run选模存在偏差；缓存候选必须按90节原图入口复评后才可晋升。
不能更新正在进行12轮相位训练的ff5b44e3工作树；CPU目录无活跃进程后才能切换到已推GitHub的新源码。

实际混合损失已完成800步，选回step0原权重，末期缓存82.00%；未改善，不做重复GPU复评。
CPU PID1819040已退出。保持82.75%的seed73正式候选，不把新PT文件名视为性能提升。

## 92. 原图相位+读出接受Top1排序监督

第89节仍按原配置完成，不能在线修改。新`sku_phase_head_top1`从已验证82.75%的b14a34ea权重开始，
只更新12份相位+原Linear两张量；电子残差、frontend、alpha、head LayerNorm全冻结，逐轮/最终SHA检查。
两张TRAIN视角分别以完整1600张detached TRAIN bank为参考，排除自身；两个方向的Top1 softplus取平均。
温度.1、余弦margin .02，关闭该组SupCon和全正样本项（权重均0）；原光学正则与路由均衡仍保留。
原10% batch噪声/DC、10%训练读出dropout、轻微亮度/对比度增强不变；推理无新增层/分支/集成。
这是一组训练配方对照，不将效果归因于某个单独因素。公用损失函数位于retrieval_refine，
CPU和原图训练共用同一实现，默认旧NLL行为保持不变。

先在确认空闲GPU上执行两步冒烟（以下GPU4是本服务器空闲卡；禁止占用他人GPU）：

```bash
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$R/abo200_readout_top1_seed73_20260915/best.pt" --expected-checkpoint-sha256 "$S73_SHA" \
  --multi-view --refine-profile sku_phase_head_top1 --lr-scale .05 \
  --epochs 1 --steps 2 --eval-every 1 --batch-size 4 --bank-batch-size 16 \
  --output "$T07/runs/smoke/abo_phase_head_top1_20260915"
```

确认complete、冻结SHA不变、12份相位及head均更新、正常/去光/路由检查通过后，且冒烟进程已退出，
再从同一原b14a34ea权重启动8轮×100步，添加`--bank-refresh-steps 25`，
output改`$R/abo200_phase_head_top1_20260915`；不得从冒烟last续训。
最大lr：phase .0001、router .0000015、head .000045；2轮warm-in+cosine，EMA.99；只best/last。
共享GPU预算最多两张，旧12轮组完成后及时核验进程/显存；未验证的新候选不覆盖正式82.75%。

### 第92节实际结果与83%权重复现

实际两步训练已选第1轮EMA，训练器最终原图83.00%/去光76.375%，独立原图重跑一致。
故原计划8轮没有启动；不能把本结果写成8轮训练收益。固定起点为第90节seed73，不是随机初始化。
best SHA `538477e0168c90cb7e0952d7c32ef5f3c4391568839a2865f86a21febf0b27b2`；源码d111dd91，410测试通过。
scope_audit确认best/last只有12份相位+原Linear两张量变化，其他张量及metadata完全不变；
EMA相位变化非常小（约2e-6 rad），不能夸大成重新学到大幅不同的光场。
TRAIN自图排除94.875%；mAP=.7297231523、NDCG=.7873187667，gallery/query路由各自合格。
周期TEST/EMA/跨run选模有偏差，比82.75%仅多2张，不能称统计显著或实验室准确率。
原12轮NLL对照也完成，最佳第1轮live82.50%/去光76.50%，未采用。
训练PID1772832/1827319、复评PID1836668均已退出，GPU已释放。

为便于查找，完整候选已按逐文件SHA复制到
`$R/abo200_phase_head_top1_verified_20260915`，源短程run保留不移动；
`promotion_manifest.json`记录来历、所有文件校验及不是8轮训练这一事实。
里面的execution/command原路径保留，是实验溯源记录，不要误当成复制失败。
复现训练：checkout源码d111dd91，按本节两步命令、同起点和配置运行到一个新空目录。
复现固定权重：设置T07/R（第89节），使用下面命令；不需要重新训练，output必须是新的空目录。

```bash
P="$R/abo200_phase_head_top1_verified_20260915"
NEW_SHA=$(python -c 'import pathlib,json,hashlib,sys; p=pathlib.Path(sys.argv[1]); s=json.loads((p/"final_report.json").read_text())["best_sha256"]; assert hashlib.sha256((p/"best.pt").read_bytes()).hexdigest()==s; print(s)' "$P")
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen optical \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$P/best.pt" --expected-checkpoint-sha256 "$NEW_SHA" \
  --batch-size 4 --cache-readout-input --output "$P/verification_repeat"
```

已完成的独立复评目录为`verification`；上面`verification_repeat`供下一位同学重新验证。
GPU编号仅示例，必须先确认空闲。phase_masks.png是相位可视化；best.pt是推理/续训权重，last.pt保留末步训练状态。

## 93. 在83%基础上继续，固定目标至少665/800

每SKU仍8张TRAIN/gallery+4张QUERY，200SKU共2400张；训练/图库是同一1600张，不是额外1600张。
800个查询每多对1张增加.125个百分点。新目标83.125%=665/800，不更换划分、相关性定义或排序规则。
保留第92节83.00%原权重，不用针对某张QUERY的规则凑分。
源码58815677（计算实现与d111dd91相同），先仅对原384→64读出做TRAIN Top1拟合：

```bash
START83="$R/abo200_phase_head_top1_verified_20260915"
START83_SHA=$(python -c 'import pathlib,json,hashlib,sys; p=pathlib.Path(sys.argv[1]); s=json.loads((p/"final_report.json").read_text())["best_sha256"]; assert hashlib.sha256((p/"best.pt").read_bytes()).hexdigest()==s; print(s)' "$START83")
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout \
  --source-run "$START83" --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --expected-checkpoint-sha256 "$START83_SHA" --expected-hit .83 \
  --fit-space projection384 --ranking-loss top1_softplus --input-dropout .1 \
  --steps 800 --eval-every 50 --batch-size 128 --lr .000025 --anchor 1 --seed 42 \
  --output "$R/abo200_readout_after83_20260915"
```

每50步TEST择优，明确有选择偏差。原图复评沿用第92节retrieval_screen，
checkpoint换成新run/best.pt，SHA必须从新报告核对，output为新run/verification。
如果没有达标，再从原83%（不是失败读出）执行`sku_phase_head_top1`：lr-scale .025，
3轮×20步、seed42、每轮评估、eval batch4、bank batch16，output为`abo200_phase_head_after83_20260915`。
仍只训练12份相位和原Linear，冻结其余电子/alpha。先原图验收再晋升；默认只用一张空闲4090。

仅读出组实际完成：缓存82.875%，best第550步；原图复评82.375%/去光76.25%，未采用。
CPU PID2340402与复评PID2342002已退出。不能拿缓存结果覆盖83%原图候选。
随后从原538477e0权重启动上述3轮短程相位训练；不在线更新训练worktree源码。

```bash
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$START83/best.pt" --expected-checkpoint-sha256 "$START83_SHA" \
  --multi-view --refine-profile sku_phase_head_top1 --lr-scale .025 \
  --epochs 3 --steps 20 --eval-every 1 --batch-size 4 --bank-batch-size 16 --seed 42 \
  --output "$R/abo200_phase_head_after83_20260915"
```

相位组进行期间，补充一个CPU收尾对照：仅将上面读出命令的`--input-dropout .1`改`0`，
output改`abo200_readout_after83_nodrop_20260915`，其他设置及原83%起点不变。
实际800步完成，选回step0；TRAIN缓存约94.75%→95.75%，TEST缓存82.875%→81.875%，不采用。
没有把训练集改善解释成泛化改善；PID2349563退出，不为相同起点重复占GPU评估。

再补充训练读出dropout=.2对照：同一CPU命令只改`--input-dropout .2`，
output为`abo200_readout_after83_dropout20_20260915`。800步完成，缓存最高仍82.875%、末期82.50%；
没有明显收益，保留完整history，不把更大正则化默认认定为有效。PID2351547已退出。

原dropout=.1配方另做预先固定seed17/73（非集成），output分别为
`abo200_readout_after83_seed17_20260915`和`abo200_readout_after83_seed73_20260915`。
两组800步均未超过step0缓存82.875%，停止该配方种子重复；不把较低结果隐去。

3轮相位组已完整结束：第1轮live82.375%/EMA82.75%，第2轮均82.50%，
最终best选回epoch0原83%（同权重去光76.375%）；冻结SHA通过，PID2344101退出释放GPU。
这不是新增83%训练收益，不替换第92节的538477e0原权重。

## 94. 原读出SAM平坦化对照（无新增推理层）

第93节的关闭dropout、增加dropout及两种子复核没有提升。这里保持83%起点、原Linear和TRAIN-only数据，
只将CPU优化改成已有SAM+Adam：第一次反传后在原head weight/bias上沿梯度做L2半径.01扰动，
第二次反传使用相同TRAIN batch、相同dropout；恢复权重后裁剪梯度并Adam更新。
异常时也恢复。默认`--sam-rho 0`逐值保持旧Adam更新及随机数状态，单元测试覆盖。
冻结全部前端/电子残差/光学相位/alpha，不向checkpoint增加任何推理模块；原8-bit导出/光路不变。
每个记录点附SAM loss gap/gradient norm，仅训练诊断，不是测试指标。先运行同一个800步、固定seed42对照。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout \
  --source-run "$START83" --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --expected-checkpoint-sha256 "$START83_SHA" --expected-hit .83 \
  --fit-space projection384 --ranking-loss top1_softplus --input-dropout .1 \
  --steps 800 --eval-every 50 --batch-size 128 --lr .000025 --anchor 1 --seed 42 --sam-rho .01 \
  --output "$R/abo200_readout_sam_after83_20260915"
```

必须先通过本地/服务器回归并推GitHub，再在无活跃任务的worktree切换新commit执行。
新候选仍需原图正常/同权重去光复评、逐张量保护与TRAIN/路由审计；缓存分数不代表达标。

该SAM组已完成800步，选回step0，缓存新训练最高82.75%，未超过起点82.875%。
源ccdc87f2，本地/服务器413测试通过；CPU PID2361120退出，没有晋升。

## 95. 对齐选模精度，仍以完整原图验收

只读诊断`abo200_phase_head_top1_verified_20260915/verification/head_precision_audit.json`发现：
同一538477e0权重及冻结head输入，CPU FP32读出为82.875%，而CUDA BF16 autocast、batch4读出为83.00%，
后者与完整原图缓存**逐值相等，max error=0**。因此旧CPU选模有可能选错临界权重。
不能把该差别直接计为新训练收益，也不能更换原图评估口径。

新可选`--selection-precision cuda_bf16`仅在每50步评估时重放原有Linear及L2归一化，
使用相同batch4与GPU autocast；起点须与源原图缓存逐值一致，否则拒绝运行。
梯度训练仍只使用CPU FP32 TRAIN1600张；QUERY评估装饰no_grad，不参与梯度。
冻结的光电主干不变，部署代码不变；所选best最终仍完整原图重跑normal/remove并核验。
先与93节首组严格匹配：相同83%起点、seed42、lr.000025、anchor1、dropout.1、800步，不加SAM。

```bash
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout \
  --source-run "$START83" --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --expected-checkpoint-sha256 "$START83_SHA" --expected-hit .83 \
  --fit-space projection384 --ranking-loss top1_softplus --input-dropout .1 \
  --steps 800 --eval-every 50 --batch-size 128 --lr .000025 --anchor 1 --seed 42 \
  --selection-precision cuda_bf16 --output "$R/abo200_readout_cuda_select_20260915"
```

这组虽在CPU训练，也会因选模占用一张GPU，须计入同一预算，结束核验PID/显存释放。

实际第一组完成800步：起点逐值重放83%通过，最终选回step0；末期82.50%，不晋升。
源码1fa478ce，本地/服务器415测试通过，PID2368299退出释放GPU。
下一次只试小步长SAM收尾：同第95节命令，lr改`.0000025`、steps改`200`、eval-every改`10`，
加`--sam-rho .01`，output为`abo200_readout_sam_micro_20260915`；仍原83%起点/seed42，
每10步TEST择优的选择偏差须披露，不用频密选模结果冒充独立测试。

小步长SAM seed42已完成，所有记录点82.875%～83.00%，未达到83.125%，PID2370178已退出。
固定补充seed17/73，其他设置完全相同（仍各自从原83%开始），output分别为
`abo200_readout_sam_micro_seed17_20260915`和`abo200_readout_sam_micro_seed73_20260915`。
按顺序单GPU执行，不集成，不隐去较低结果；这不是三次从头独立训练。

固定seed17/73两组均完成200步，最高仍83%，没有达到665/800。
PID2371881/2372427和父队列2371880均已退出，GPU上下文释放；不替换原83%权重。

## 96. 同商品两张TRAIN照片的排序约束

不继续重复第95节的小学习率/dropout组合。训练损失增加可选`two_view_softplus`：
对每张TRAIN query，在其余7张同SKU训练图里选择相似度最高的两张不同照片，
取二者余弦均值，与最相近的错误SKU照片作softplus margin比较，温度.1、余弦margin .02。
两个正例都有梯度；排除query自身。该损失约束的是二者均值，并不保证两张分别都超过负例，
也不按相机角度筛图、不声称两张必定是大角度差。实际检索仍只需Top1同SKU正确。
它与光学router的Top2是两回事：推理阶段没有新投票、原型平均或标签筛选。

从固定83%权重开始，只训练原384→64 Linear，冻结光学、电子残差、alpha与输入前端。
其余配置严格匹配95节首组：800步、lr.000025、dropout.1、anchor1、seed42，不加SAM。
选模按CUDA BF16原head重放，每50步TEST择优；起点逐值检查、最后原图复评要求不变。

```bash
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout \
  --source-run "$START83" --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --expected-checkpoint-sha256 "$START83_SHA" --expected-hit .83 \
  --fit-space projection384 --ranking-loss two_view_softplus --input-dropout .1 \
  --steps 800 --eval-every 50 --batch-size 128 --lr .000025 --anchor 1 --seed 42 \
  --selection-precision cuda_bf16 --output "$R/abo200_readout_two_view_20260915"
```

仅TRAIN进入梯度，QUERY参与周期选模的偏差仍存在；不能将该对照当成独立测试或新数据划分。

第96节仅读出对照已完成800步，最高仍起点83%，末期82.875%，未晋升；PID2380918退出。

## 97. 原光电主干联合学习两正视图目标

只优化现有读出多轮未提升，接下来让同一目标同时更新原光学和原电子残差。
`sku_two_view_joint`继承原`sku_capacity_control`，只换TRAIN目标为第96节的两正照片softplus，
启用两张已有TRAIN视角分别查询完整detached训练图库；SupCon/全正例附加loss权重均0。
没有扩核、扩宽、额外头或分支；六次10cm、光Top2、478ROI、64D检索输出保持原样。
Qwen紧凑前端冻结；alpha可在原[.4001,.8]内更新，不把它误写成固定alpha实验。
原电子dropout、轻微亮度/对比度增强、10% batch噪声/DC及router均衡保留；像素位移/8-bit相位STE不新增。

从原83%权重开始，6轮×100步，lr-scale .05、每25步刷新TRAIN图库，每轮评估live/EMA与TRAIN。
峰值lr为expert/global .0001、router .0000015、原head .000015、其余原电子 .000005，另有原warm-in/cosine。
这是联合微调，活动参数约278万，不是此前约98万的相位+头冻结电子对照。
只保留best/last；沿用初始83%保底。必须监控首轮梯度、相位变化、alpha界限、专家分布及最终去光。

```bash
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$START83/best.pt" --expected-checkpoint-sha256 "$START83_SHA" \
  --multi-view --refine-profile sku_two_view_joint --lr-scale .05 \
  --epochs 6 --steps 100 --eval-every 1 --batch-size 4 --bank-batch-size 16 --bank-refresh-steps 25 --seed 42 \
  --output "$R/abo200_two_view_joint_20260915"
```

不得在本任务运行期间在线切换其worktree源码。默认只使用一张空闲4090，运行结束确认自己PID释放。

第97节于3轮结束后提前停止，3轮EMA为82.50%、82.00%、81.375%；
`early_stop_report.json`记录原因、best/last SHA和原83%权重逐值相同验证。PID2388019已退出，未晋升。

## 98. TRAIN间隔满足后停止排序梯度

起点仍是独立原图确认的83%权重，不用第97节低于起点的中间权重。
依据第97节run的`train_margin_diagnostic.json`：原83%模型在1600张TRAIN自图排除检索中，
1518张正确，1493张最近正确/错误余弦间隔已大于.02；82张错误中77张错到同大类其他SKU。
这里只检查TRAIN，不按QUERY错例选样或改变标签。

新增`top1_squared_hinge`目标为`.5*relu((s_wrong-s_correct+.02)/.1)^2`。
满足间隔的TRAIN query排序梯度为0；其他样本及参数共享、锚定约束仍可能改变其预测，
不能保证旧正确图片一张不掉。它和softplus的区别是有限间隔后停止继续推开简单样本。
只训练原384→64 Linear，所有光学、前端、电子残差和alpha冻结；无新推理参数。
先用无dropout的干净TRAIN间隔检验该假设，800步、lr.000025、anchor1、seed42、每50步评估。
仍按CUDA BF16 batch4重放选模，不能直接把CPU FP32缓存分数当成正式原图成绩。

**第97节仍运行时不得切换其worktree或占用第二张GPU。**确认其PID已退出且GPU空闲，
再切到包含本节代码、已通过测试且已推GitHub的commit，运行：

```bash
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout \
  --source-run "$START83" --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --expected-checkpoint-sha256 "$START83_SHA" --expected-hit .83 \
  --fit-space projection384 --ranking-loss top1_squared_hinge --input-dropout 0 \
  --steps 800 --eval-every 50 --batch-size 128 --lr .000025 --anchor 1 --seed 42 \
  --selection-precision cuda_bf16 --output "$R/abo200_readout_active_margin_20260915"
```

无dropout对照已完成800步：最高仍起点83%，新训练最高82.25%，最后81.50%；PID2407635已退出。
源码a817887c，本地/服务器422项测试通过。原图权重不替换，不对原83%重复宣布新提升。
匹配对照仅将上述命令改为`--input-dropout .1`，输出改为
`abo200_readout_active_margin_dropout10_20260915`；其他参数、起点、种子相同。
检验干净TRAIN间隔优化是否缺少增强，不同时改学习率/锚定，不增加推理dropout。
任何新候选均须正常/去光原图复评、TRAIN与路由审计，保留原83%权重。

第98节10% dropout匹配组800步已完成，新训练最高82.625%、末期81.875%，未超过83%起点。
短程收尾`abo200_readout_active_margin_micro_20260915`保持10% dropout，只改lr.0000025、
50步/每5步评估，最高83%、末期82.875%。PID2410499/2412167均已退出，不晋升；
源码08f5309c（只更新文档，训练实现与a817887c相同）。这些都是TEST择优，不是无偏独立实验。

## 99. 从已验证83%权重极短相位/原头续训

第93节每20步评估的相位+头续训未提升；这里只检验更早的2步状态，仍不改原电子残差和alpha。
该设置对应第92节短程训练配方，但起点为83%而非82.75%；不把历史结果当成本次结果。
保留best/last，按原live/EMA和路由条件选择，之后还须独立原图复评。
光学相位的微小更新不能自动解释为宏观的光贡献提高；性能结论只引用真实全量结果。

```bash
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$START83/best.pt" --expected-checkpoint-sha256 "$START83_SHA" \
  --multi-view --refine-profile sku_phase_head_top1 --lr-scale .05 \
  --epochs 1 --steps 2 --eval-every 1 --batch-size 4 --bank-batch-size 16 --seed 42 \
  --output "$R/abo200_phase_head_micro_after83_20260915"
```

seed42已完成，live82.50%/EMA83%，最终选择epoch0原权重：正常83%、去光76.375%，
冻结SHA和路由检查通过，PID2413377已退出。正式成绩未提高，不晋升。
固定补充seed17/73的同配方控制，分别将`--seed`改为17/73、output改为
`abo200_phase_head_micro_seed17_20260915`/`abo200_phase_head_micro_seed73_20260915`。
各自从原83%开始、一次只跑一组；若某组达到目标则先独立原图复评，不为凑组数继续占GPU。
它们是同一训练起点的短程种子对照，不是从头三次训练；TEST/live/EMA跨组择优偏差必须披露。

## 100. 仅TRAIN统计量的原读出bias中心化（待运行）

第99节seed17 run的`train_readout_geometry.json`只检查原83%起点的1600张TRAIN：
pre-L2均值范数.23505、单图范数中位数3.77127，归一化特征均值范数.05741；
线性权重条件数5.04，不据此声称存在严重退化或中心化一定有效。
可检验训练数据中的公共偏置是否影响检索，且不加新分支、层、推理算子：
令`mu=mean_TRAIN(Wx+b)`，直接将现有bias改为`b-mu`。
统计量只来自TRAIN，QUERY/标签不参与估计；去光时也使用同一个校准后的bias，不能另估计一份。
W、光学相位、前端、电子残差、alpha和metadata都不改。推理仍原Linear64+L2+cosine。
此项是一次解析的bias校准，不是训练了更多epoch，也不宣称光学mask发生新学习。
先固定完整中心化strength=1，不扫描测试图片或设定按商品的偏置。

```bash
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout \
  --source-run "$START83" --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --expected-checkpoint-sha256 "$START83_SHA" --expected-hit .83 \
  --fit-space projection384 --train-center 1 --steps 0 --selection-precision cuda_bf16 \
  --output "$R/abo200_train_centered_bias_20260915"
```

开始前原83%权重的CUDA重放必须逐值一致；中心化后正常/去光采用同精度head重放。
只有候选达到目标后，再从完整原图独立确认；不能用缓存分数直接替代正式结果。
默认train-center=0严格保持原数值路径；仅显式正strength允许steps=0，避免误把空训练当作成功。

实际结果（源码f9d25bff，本地/服务器424项测试通过）：完整校准82.75%、
`--train-center .5`的`abo200_train_half_centered_bias_20260915`为83%；
`--train-center .25`的`abo200_train_quarter_centered_bias_20260915`为82.875%，均没有新提升。
另外`abo200_train_half_centered_refine_20260915`从原83%先半量校准再训练：
`--train-center .5 --steps 200 --eval-every 25 --ranking-loss top1_softplus --input-dropout .1 --lr .000005 --anchor 1 --seed 42`，
其他参数同本节，最高仍83%，未晋升。PID2428479/2429193/2430031/2430936均退出。
未扫描更多校准强度，也未将微小mAP提升当作Hit@1达到目标。
第99节seed17也已正常结束，最终原83%/去光76.375%；PID2422424退出，seed73按原约定单独继续。

## 101. 只校准原四个alpha，冻结全部相位和电子权重

`sku_alpha_only`从原83%开始，只训练V/L各两层现有fusion logit，共4个标量；
保留原sigmoid映射区间[.4001,.8]，不引入新的缩放/分支或更低alpha。
光相位、router相位、紧凑前端、电子残差、读出头、归一化参数全部冻结；每轮和选出的best
都对这些参数做SHA核对。仍为六次10cm/Top2/64维单图检索；alpha变化可能改变下游L路由，
不能直接继承旧路由统计，必须重新评估资格及去光。

模型eval模式下保留梯度，只拟合TRAIN双向图库最近正/负排序loss，TRAIN自身排除；
无教师/额外SupCon、沿用原router均衡约束。此校准关闭训练随机CCD噪声及电子dropout，
仍有轻微亮度/对比度增强，不删除checkpoint中的硬件噪声配置；不是新噪声鲁棒性证据。
峰值raw-logit学习率.01（lr-scale1、alpha专属倍率100），沿用warm-in/cosine和EMA；
实际alpha更新还经过sigmoid导数，并不等于每步直接变化.01。

先做1轮2步真实原图scope检查（输出位于simulation，标注短程检查，不宣称完整训练）：

```bash
CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt \
  --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data \
  --manifest "$R/abo200_enrolled_protocol_20260913/protocol.json" --assets "$R/standalone_assets_20260910" \
  --checkpoint "$START83/best.pt" --expected-checkpoint-sha256 "$START83_SHA" \
  --multi-view --refine-profile sku_alpha_only --lr-scale 1 \
  --epochs 1 --steps 2 --eval-every 1 --batch-size 4 --bank-batch-size 16 --seed 42 \
  --output "$R/abo200_alpha_only_scope_20260915"
```

scope/原图/冻结SHA检查全部通过后，若未达目标，再独立从原83%执行3轮×20步、
`--bank-refresh-steps 10`，输出`abo200_alpha_only_20260915`。每轮完整TRAIN/TEST/live/EMA，
保留best/last与起点保底。不能把准备好profile或有限梯度当作已经提升性能。

第99节最后的seed73已完成（source f9d25bff）：live/EMA82.875%，最终原83%/去光76.375%，
PID2431884已退出。第101节源码c546791f已推GitHub，本地/服务器427项测试通过，
真实原图scope检查已在GPU4启动；不得在线切换活动worktree源码。
