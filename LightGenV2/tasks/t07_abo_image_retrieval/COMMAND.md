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

## 47. 更强的逐图特征蒸馏（运行中；不要重复启动）

从第40节固定79.375%权重开始，只把教师64维余弦损失权重2改成8，其余同joint_restart。
没有新增推理网络或改变光路，也不叠加第45/46节。先确认GPU2空闲，不挤占他人任务；最多3张GPU。
源码必须通过测试并同步GitHub。新高必须独立4090复评正常/去光，不能把蒸馏强度当作光贡献占比。
已用18c4e400启动，监督3308242/学生3308245；两端181项测试通过，完整初始评估79.375%，只改变教师余弦权重。

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

## 48. Router按物理相位弧度做Adam更新（运行中，不重复启动）

训练前向、光路、ROI、Top2、alpha及保存格式不变；仅两个Router采用弧度坐标Adam及角度EMA。
只在backward和optimizer.step之间临时转换，不能在该上下文内前向或保存。
不叠加第45节初始化平移、第46节相位优先或第47节强教师。固定原79.375%权重，初始评估需重算。
先完成测试并同步GitHub、真实输入梯度检查，再按下列命令接续第45节；依赖未完成时不分配CUDA。
已用a28a8438启动监督3357893等待依赖；两端187项测试及真实4图CPU一步检查通过，不要重复排队。
第45节已正常完成并释放GPU1，当前学生3378938已接续，execution确认radians配置；不占第四张卡。
完整初始原协议评估79.375%；等待训练后成绩，不把该初始值报成训练提升。

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

## 49. 仅扩宽现有电子残差MLP（运行中，不重复启动）

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

## 50. 原结构联合续训的seed123对照（已排队，不重复启动）

只改变第40节joint_restart的训练seed，保持原384宽MLP、原raw Router优化器、固定79.375%起点及原协议。
监督3488475使用已发布源码abb36acd，等待第47节结束后接续GPU2；等待不占CUDA，不启动第四张卡。
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
