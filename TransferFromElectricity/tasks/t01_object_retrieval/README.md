# Caltech101：固定专家库生成

用户已批准首轮固定专家库实验。实现与结果只在本任务维护，不写回 LightGenV2 的正式对照表。

## 首轮协议

- 光路直接复用 LightGenV2 T01 main_dc20；vision/language 各四专家 Top-2，Router/global 正常直接优化。
- A direct；B small_hyper；C qwen_frozen；D qwen_lora。生成器只接收固定任务/模态/expert 条件。
- D 使用独立 Qwen3-VL-2B-Instruct 的语言 Transformer，所有 q_proj/v_proj 挂 rank=8、alpha=16 LoRA。
  Qwen 基座冻结，LoRA 与 float32 patch 解码器训练；不改变原任务前端。
- 使用 `anchor + decoder(context) - initial_reference` 生成 raw phase。anchor 是同 seed 的固定随机相位，
  initial_reference 是训练前生成值，二者均为不可训练 buffer。因此四组初始光学相位完全一致；
  后续没有独立可训练的 expert 像素参数（A 除外）。这替代草案中的 B/C 初始化拟合。
- 经 torch parametrization 将生成 tensor 接入原 PhaseLayer，不改变 FFT、相位转换、DC 正则和噪声实现。
- 复用原 2,625 train / 30 gallery / 200 test 划分，再仅从 train 固定抽每类 30 张，合计 300 张。
  PK=10×3；5 epoch，默认每 epoch 10 optimizer steps。此轮是 pilot，不能和全量正式结果混表。
- 任务 loss 与 backend 一致：supervised contrastive + episodic prototype，附原路由/CCD/DC 正则。
- 只在最后一轮评估 EMA，不按 test 选择 checkpoint；`best_checkpoint.pt` 明确表示最终 EMA。
  `last_checkpoint.pt` 为 live 可恢复状态；保留 `expert_bank.pt` 独立部署相位。
- 记录 task loss 到 expert / LoRA 的非零梯度、显存、训练日志，以及移除生成器后的预测一致性。

## 运行

从仓库根目录，在已有 xml 环境运行；GPU 由外部环境选择。

```bash
python -m unittest discover -s TransferFromElectricity/tasks/t01_object_retrieval/tests -v
CUDA_VISIBLE_DEVICES=3 python -m TransferFromElectricity.tasks.t01_object_retrieval.run \
  --method qwen_lora \
  --run-dir TransferFromElectricity/tasks/t01_object_retrieval/runs/simulation/YYYYMMDD_static_qwen_lora_pilot_s42
```

相同命令替换 method 为 direct / small_hyper / qwen_frozen，使用独立 run-dir。
`--generator-source` 可指定本机缓存 snapshot；默认通过已有模型缓存解析。
`--epochs 1 --steps-per-epoch 2` 是实际光路短检查，用独立 runs/smoke 目录。
中断后同配置加 `--resume`，只读取 last；恢复要求相同代码 commit。

## 证据与限制

每 run 保存完整 backend config、pilot 配置、样本清单、Git SHA、初始化 checkpoint SHA、
环境、原始命令、实际参数组、LoRA 模块名、loss/梯度、最终指标与导出误差。
固定 anchor 与初始生成值仅用于相同起点，不是从已训练 baseline 蒸馏。
小生成器与 Qwen 的总参数/训练成本不相同；本轮仅按相同样本与 optimizer steps 比较。
CUDA 仿真耗时不是物理光路延时；生成器仅在训练和导出使用。
真实 SLM/CCD 闭环不属于本轮。

## 2026-09-07 首轮结果

四组均已完成：300 train、30 gallery、200 test，seed=42，5 epoch / 50 updates，最终 EMA。
训练 commit：`56d7e6b66b9d820b64ad476ceccb5bff56f55947`；运行在服务器 GPU 3（RTX 4090）。
源码、数据划分和预算一致；本地轻量文件从服务器经 SFTP 同步并逐文件校验 SHA256。

| 方法 | Top-1 | Top-3 | MRR | 峰值显存 GiB |
|---|---:|---:|---:|---:|
| direct | 82.0% | 93.0% | 0.8860 | 7.43 |
| small_hyper | 82.0% | 93.0% | 0.8868 | 7.44 |
| qwen_frozen | 81.5% | 93.0% | 0.8843 | 10.64 |
| qwen_lora | 81.5% | 93.0% | 0.8835 | 11.25 |

机制验证通过：D 的任务 loss→expert 梯度范数 0.05483，任务 loss→LoRA B 梯度范数
0.0008387；最终 EMA expert 相位相对起点的 RMS 变化 0.01331 rad。
四组固化 expert 后输出误差均为 0，样本逆序误差为 0；单样本与 batch 推理最大 embedding
绝对差为 0.00223–0.00298（BF16 路径，低于预设 0.005 容差），不能表述为逐位一致。

结论仅为链路可训练、可导出；本轮没有观察到大模型生成优于直接优化。
D 与 A 的 Top-1 差距仅 1/200 个查询，单 seed 短训练不能据此判断方法优劣。
Language Router 四组均只使用前两个专家（训练选择次数 1500/1500/0/0），尚未解决路由集中。
相位变化较小且四组 loss 曲线非常接近，不能将 loss 下降全部归因于生成器。
小生成器首 batch task loss 与 A 相差约 0.00154，虽初始无梯度相位校验误差为 0，
该端到端微小差异尚未逐算子定位；B 保留为辅助对照。

完整数值、run ID 和证据哈希见 [summary.json](reports/pilot_20260907/summary.json)、
[evidence_manifest.json](reports/pilot_20260907/evidence_manifest.json)；
[对照图](reports/pilot_20260907/pilot_comparison.png) 给出每 epoch 平均 task loss 与最终 Top-1。
原始日志、配置和 sample manifest 在任务 runs 下；checkpoint 留在服务器同名 run。
后续优先诊断 Language Router 集中及生成相位的实际贡献，再考虑延长训练和多 seed。

## 当前实际计算图

本任务复用的是光电混合 student。原任务 Qwen3-VL-Embedding 的图像 patch/位置嵌入、
视觉 merger、文本 token embedding 等冻结组件保留；原 Vision/Language Transformer 主干
在 student 模式被替换或旁路。不能将其描述为完整冻结大模型先提取语义特征、后接纯光学网络。
生成 mask 的 Qwen3-VL-2B-Instruct 是另一套独立模型，仅使用其语言 Transformer。

```mermaid
flowchart TD
  I[图像] --> P[冻结的 patch 与位置嵌入]
  P --> V[Vision 光电混合模块]
  V --> M[冻结的视觉 merger]
  T[固定检索提示词的 token embedding] --> L[Language 光电混合模块]
  M --> L
  L --> R[池化与可训练读出：64 维归一化向量]
  R --> LOSS[检索损失]
  C[固定任务、模态、专家编号] --> G[独立 Qwen：冻结基座与可训练 LoRA]
  G --> H[可训练的连续相位解码器]
  H -. 4 张固定专家 mask .-> V
  H -. 4 张固定专家 mask .-> L
  LOSS -. 通过光学仿真反传 .-> H
```

Vision 和 Language 顺序连接；每个模块内部有两个阶段，且都有可训练电子通路：

1. 第一阶段：电子 block 与“幅度编码→光学 Router→Top-2/4 expert→传播与 CCD→读出”并行，再融合。
2. 第二阶段：上一步融合特征进入电子 block；同时按已有路由重新编码到 global 光路，经过
   global phase、传播与 CCD、读出，再融合。因此 expert→global 之间存在探测和电子重载。

两阶段均先匹配光电特征尺度，再按约 `(1-alpha)*E + alpha*O` 融合并归一化，外围还有适配器和残差。
当前四个 alpha 从 0.055 起步，50 步后仍在 0.054957–0.055038 左右；这是特征融合系数，
不能解释为光学分支贡献了 5.5% 的准确率或物理能量。
Vision 的电子 block 使用二维局部混合，Language 使用因果一维局部混合；内部宽度 192。

| 参数 | direct | qwen_lora |
|---|---|---|
| 每模态 4 张 224×224 expert raw phase | 直接梯度下降 | 由生成器产生，任务梯度回传解码器与 LoRA |
| 光学 Router、global phase | 直接梯度下降 | 直接梯度下降 |
| 电子 block、适配器、融合门、检索读出 | 直接梯度下降 | 直接梯度下降 |
| 任务前端的预训练嵌入/merger | 冻结 | 冻结 |
| 独立生成器的 Qwen 基座 | 不使用 | 冻结；训练 q_proj/v_proj 的 LoRA |

固定专家库指所有样本共用同一套参数；训练时随优化更新，导出后固定，推理时无需生成器。
生成器输入没有当前图片、样本标签或 batch。连续解码器将条件向量映射为 14×14 个 16×16 patch，
形成每张 224×224 raw phase，再由原 PhaseLayer 转成 `2*pi*sigmoid(raw)`。
这实现了可微的专家参数生成；是否利用了大模型预训练知识带来收益，需要后续对照，不能由链路通畅推出。

## 2026-09-07 checkpoint 诊断

仅重新评估上述四个 run 的最终 live/EMA checkpoint，没有继续训练或按 test 重新选择权重。
诊断源码 commit：`f44979cede66f252e3bfa29d6826709b862482fc`；
诊断 run：`runs/smoke/20260907_checkpoint_diagnosis`。
原四组 EMA Top-1 全部复现；原 checkpoint 的 SHA256 已记录。

| 方法 | 最终 live Top-1 | 原报告 EMA Top-1 | live 相位变化 RMS / rad | EMA 相位变化 RMS / rad |
|---|---:|---:|---:|---:|
| direct | 85.5% | 82.0% | 0.046407 | 0.008393 |
| small_hyper | 85.0% | 82.0% | 0.044263 | 0.007774 |
| qwen_frozen | 85.5% | 81.5% | 0.071835 | 0.014128 |
| qwen_lora | 85.5% | 81.5% | 0.059505 | 0.013309 |

相位变化相对共同初始 expert bank，使用圆周差统计全部八张相位。D 与 A 的 live 相位
RMS 差为 0.075171 rad；去除各 mask 的常数相位偏移后仍为 0.075170 rad，说明不是仅有整体相移。
两者 query embedding RMS 差为 0.0031523，200 个 query 中有 2 个 Top-1 类别预测不同，
但总准确率相同。相同准确率不能表述为相同 mask 或相同输出。

EMA 衰减率 0.995，50 步后初始参数权重仍有 `0.995**50 = 0.778313`。
这是参数平均的权重，不能按此线性分解生成器输出相位或准确率。
初始 warmstart 模型、随机专家、重置 Router/融合门在本次重评估中已有 81.0% Top-1。
因此首轮仅展示 EMA 隐藏了短训练 live 权重的变化；两种权重都应报告，不能按 test 挑较好者。

### 保持其他训练后参数不变的干预

以下均针对 D 的最终 live checkpoint，gallery 与 query 都在对应干预下重新编码：

| 干预 | Top-1 | 相对正常模型改变的 Top-1 类别预测数 / 200 |
|---|---:|---:|
| 正常训练后模型 | 85.5% | 0 |
| 仅将 expert 换回初始随机相位 | 86.0% | 1 |
| 仅将 expert 换为统一 pi 相位 | 85.5% | 0 |
| 仅将 LoRA B 置零，保留训练后的相位解码器 | 85.5% | 0 |
| 关闭所有光学特征贡献 | 84.5% | 11 |

关闭所有光学分支是推理干预，不等于另行训练的纯电子 baseline。
A 的 live 模型换回初始 expert 也维持 85.5%，关闭光学为 84.0%。
当前证据表明光学分支能影响结果，但这 50 步的 expert 学习尚未显示必要的检索收益。
这不证明继续训练无效，也不证明整个光学分支无用。

LoRA B 从零更新到范数 0.192866，结合先前任务梯度记录，排除了完全未更新。
在保持 D 的训练后解码器不变时，关闭 LoRA 只引起 0.000483 rad 的相位 RMS 差，
远小于其相对初始的总相位变化 0.059505 rad；LoRA 在最终生成函数中的直接影响较弱。
这一干预不能隔离 LoRA 训练过程对解码器轨迹的间接影响，也不能将 RMS 比值视作严格贡献比例。
当前 D 的 expert mask DC 指标 `abs(mean(exp(i*phase)))**2` 为 0.99527–0.99558，
相位仍接近常数屏；该指标不是整个光路的实测零级能量占比。

完整证据见 [diagnosis.json](reports/diagnosis_20260907/diagnosis.json)、
[传输校验](reports/diagnosis_20260907/transfer_manifest.json)。
[live/EMA 相位变化图](reports/diagnosis_20260907/phase_changes.png) 使用共同色标，显示每模态 expert 0；
[D 的全部八张相位变化图](reports/diagnosis_20260907/qwen_all_expert_changes.png) 同样按共同色标展示。

### 下一阶段建议（尚未执行）

先验证“只靠 expert 更新，能否学到任务”，再扩大方法比较。保持固定专家库与本任务数据身份：

1. 从 train 单独固定诊断训练/验证子集，统一起点，先做 direct 与 qwen_lora 两组。
   暂时冻结电子 block、适配器、读出、Router、global 和融合门，仅更新 expert 或其生成器。
   记录任务 loss 本身、梯度、物理相位变化、路由覆盖以及换回初始 expert 的损失变化。
   先证明训练小集能被 expert 学习，不能仅凭总 loss 中正则项下降验收。
2. 若冻结后的 0.055 融合系数限制学习，再用单独 profile 做双方一致的固定 alpha 对照；
   同时检查近常数相位初始化、DC 约束与 Language Router 仅选两专家的问题。
   每次改变一个因素，不在四种生成器上同时大规模扫参。
3. 机制通过后恢复 Router/global 的正常优化，再逐步恢复电子部分的联合训练，
   保留冻结专家对照；分别报告 live 与 EMA，然后扩展训练步数和 seed。

上述冻结 Router/global 是定位问题的临时干预，不改变主方案让它们直接梯度下降的设计。
本轮 test 已用于机制诊断，后续调参只使用固定验证集，正式泛化评估另行预先约定，
不把本次干预中较高的 test 数字作为新方法成绩。

## 第二轮：光学系数至少 0.4 的分阶段对照

用户已批准执行。配置为 [staged_alpha40.yaml](configs/staged_alpha40.yaml)，
入口为 `python -m TransferFromElectricity.tasks.t01_object_retrieval.train_staged`。
这是独立的新协议，不能将其与 50-step pilot 的差异单独归因于某一个改动。

| 命令 method | 含义 | expert 相关的可训练参数 | 对照目的 |
|---|---|---|---|
| fixed | 专家存在，但始终保持共同初始相位 | 无 | 其他部件能否在不学习 expert 时完成任务 |
| direct | 直接优化每个 expert 像素 | expert raw phase | 传统方法主基线 |
| qwen_frozen | Qwen 输出固定特征，经解码器产生 expert | 相位解码器 | 只训练生成器末端能达到什么效果 |
| qwen_lora | 微调 Qwen 内部 LoRA，并训练解码器 | LoRA 与相位解码器 | 用户提出的大模型参与学习方案 |

第 3/4 组使用同一 Qwen3-VL-2B-Instruct 语言模型、相同固定描述和完全配对的解码器初始化。
Qwen 原始权重均冻结；第 4 组的 q_proj/v_proj LoRA rank=8，训练参数改为 FP32，
冻结基座仍是 BF16。解码器末层初始化标准差由 0.002 改为 0.02，增强上下文到相位的初始梯度；
固定 reference 抵消初始生成值，因此不改变四组相同的初始专家库。
不再将上一轮 small_hyper 纳入这次主表。

师姐的 `d2nn_pack` 使用离线 CLIP 特征（提取脚本默认 RN50），后接从头初始化的四层
Transformer/MLP；优化器只更新后面的生成器，不微调 CLIP。其 mask 在 batch 内求均值，
与本任务输入样本无关的固定专家库合同不同。此说明基于实际代码，不依据注释中的 sigmoid 描述。

### 共同设置与阶段

- Vision/Language 的 expert/global 四个融合位置全部采用 alpha 下限 0.4、初值 0.5、上限 0.95。
  前两阶段冻结门值 0.5；最后阶段允许在该区间内学习，逐轮断言不低于 0.4。
- 每类从原 train 留出 10 张用于本轮适配验证，其余全部参与训练；保持原 gallery/test。
  原 warmstart 曾使用原 train，所以该验证集仅对本轮更新留出，不是整个训练历史的未见样本。
  原 test 也曾用于上一轮诊断；这轮仍是机制对照，正式泛化结论需要新的类别/数据评估。
- 专家物理相位从 `pi + Uniform(-a,a)` 初始化，其中 `(sin(a)/a)^2=0.25`，再逆 sigmoid
  映射回 raw。四组共用 seed 对应的同一张随机库，远离 sigmoid 饱和与常数屏。
  0.25 是初始化的 mask DC 统计目标，不是将光路中 20–30% 零级噪声参数改成相位初始化。
- 光学几何、传播、Top-2/4、CCD、任务损失继续复用原 backend。阶段 1 用确定性光学；
  阶段 2/3 恢复 backend 的训练扰动。每组都保存具体 resolved config。

| 阶段 | 轮数 | 更新的部件 | 冻结的部件 |
|---|---:|---|---|
| experts | 4 | expert 或其生成器；fixed 组不更新 | Router、global、电子模块、适配器、读出、融合门 |
| optics | 8 | 上述部件加 Router/global | 电子模块、适配器、读出、融合门 |
| joint | 8 | 上述部件加低学习率电子部分和融合门 | 原预训练任务嵌入/merger、生成器 Qwen 基座 |

PK=10×3；原 train 2,625 张中留出 100 张，训练 2,525 张，每轮 85 batch，20 轮共 1,700 batch。
fixed 组前 340 batch 无可训练部件，实际 optimizer update 为 1,360；其余三组为 1,700。
同一 seed 的四组共享样本顺序、划分、相位与任务模型起点，固定专家组保留为干预对照。
各方法的参数量、显存和训练成本仍不相等，不声称算力预算严格匹配。

学习率：expert 0.02、Router 0.01、global 0.006、LoRA 0.0005、解码器 0.001；
电子部分 0.00001、读出 0.00002、融合门 0.0001。每阶段单独 warmup 20 step，再 cosine
下降至该阶段峰值的 20%；按参数组分别 clip norm=1，避免所有部件共享一次梯度裁剪。
EMA decay=0.95；每轮报告 live/EMA 验证指标，主 checkpoint 固定按 live validation Top-1、
再按 MRR 选择。最终统一评估 validation-selected live 和 final EMA，不用 test 挑权重。

逐轮记录任务 loss、相位 RMS/DC/饱和比例、task→expert/LoRA 梯度、路由使用次数、融合系数，
并验证冻结参数没有变化。每阶段结束，在验证集换回初始 expert 检查专家学习贡献。
最终对所选模型做初始 expert、均匀相位、关闭 LoRA 的干预，并校验固化 mask 后输出一致。
只保存 best/last checkpoint 和独立部署 expert_bank；不生成周期权重文件。

```bash
python -m unittest discover -s TransferFromElectricity/tasks/t01_object_retrieval/tests -v
CUDA_VISIBLE_DEVICES=0 python -m TransferFromElectricity.tasks.t01_object_retrieval.train_staged \
  --method direct --run-dir TransferFromElectricity/tasks/t01_object_retrieval/runs/simulation/YYYYMMDD_alpha40_direct_s42
```

先用独立 `runs/smoke` 加 `--smoke` 检查全部阶段，再运行上面的完整协议。
`--resume` 要求相同 config 与代码 SHA，恢复 last 的优化器、EMA、阶段、日志和 RNG。
扩展顺序：先完成共同协议下的四组，再复查主比较的随机种子和新类别；类别扩大与新任务
必须使用独立配置及数据 manifest，不混入本表。新增类别是否被 warmstart 见过要明确记录。

### 类别规模与随机种子扩展

用户进一步授权扩大任务检查偶然性。新增 [staged_alpha40_caltech30.yaml](configs/staged_alpha40_caltech30.yaml)
继承相同阶段/光路配置，只将检索类别扩大到 30，并将生成器的固定任务描述改为 thirty-category。
完整数据有 101 类、8,677 张前景图像；原十类之外，按名称排序后从至少 60 张图像的类别中，
用 random.Random(42) 抽取二十类。排除 BACKGROUND_Google 和 Faces_easy，避免引入 Faces 的明显重复版本。
类别由数据数量和固定随机规则选择，没有参考模型成绩。

二十个新增类共 1,602 张图像，与原十类合计 4,457 张；每类 gallery=3、test=20、
本轮 validation=10，预计 train=3,467、validation=300、gallery=90、test=600。
每轮仍按 PK=10×3，从三十类中抽取十类，116 batch/epoch，20 轮共 2,320 batch。
除改任务描述外，生成器容量、专家数量、光学系数与训练学习率保持不变。
在三十类共同 gallery 上分别统计原十类查询和新增二十类查询；新增类在本轮有训练样本，
因此属于监督适配，不是零样本识别。

主比较 direct / qwen_lora 计划增加 seed=43、44，与 seed=42 一起报告。
`--seed` 改优化初始化、训练采样与扰动，validation 划分使用固定 data_seed=42，原 gallery/test
仍使用 backend 固定划分；不能将每个种子的测试样本换一批再混算。
报告工具同时支持四组完整表和 direct/qwen_lora 配对表。

这些属于当前工作的扩展安排；具体已完成和运行状态以每个 run 的 status.json 为准。
