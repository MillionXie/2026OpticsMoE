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

按 `preserve_adam → preserve_sam → preserve_fullfield_sam` 顺序执行，参数在 `standalone/generalization.json`，
继承`high_alpha.json`和`retrieval_training.json`。学习率、增强、采样、数据、源权重都匹配，SAM增量rho0.03。
每组30轮，SAM同step需两次反传，**不是等GPU时间对照**。状态统一看output下status.json；各组console.log、
artifacts/history.json、final_report.json保留完整数值，失败队列停止，不会一直启动失败任务。

快速实跑检查使用另一个 `runs/smoke/generalization_20260911` 输出，并加：
`--profiles preserve_fullfield_sam --epochs 1 --steps 2`。
此检查会全量编码/评估，但只更新两步；不把冒烟的准确率当最终优化成绩。
每组进程结束才启动下一组；若卡被其他人占用，队列停止并记录原因，不杀别人的进程。
