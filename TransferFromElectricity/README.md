# TransferFromElectricity：大模型生成光学专家相位

状态：**用户已批准固定专家库与 Caltech101 首轮尝试，正在实施**（2026-09-07）。
当前实现和实验事实以 [任务 README](tasks/t01_object_retrieval/README.md) 为准；下文保留原设计草案。
`d2nn_pack/` 是用户提供的参考包，保留原样。
审阅基于本地 commit `b1285f0e0ecbfa21b3acbf797d643f5d52aaf36e`。
本次 `git fetch origin` 因 GitHub HTTPS 连接失败而未成功，不能据此确认远端最新版本。

## 1. 研究问题与边界

原训练：任务 loss → 光学传播 → 直接更新 expert mask。
拟议训练：任务 loss → 光学传播 → 生成的 expert mask → 相位解码头 → 大模型内的可训练参数。
Router、global phase、原有电子适配器/融合门/任务读出按现有协议直接优化。
生成器参数仍通过梯度下降更新；改变的是 expert 相位的参数化及梯度经过的路径。

这种“一个网络生成另一个网络参数”的方法属于 [HyperNetworks](https://arxiv.org/abs/1609.09106)。
首版拟用 [LoRA](https://arxiv.org/abs/2106.09685) 更新大模型内部的低秩适配参数，
同时训练连续相位解码头；这与只训练一个接在冻结特征后的 MLP 有明确区别。
预训练模型是否比小生成器更有效，需要实验证明，不能由结构本身推出。

### 推荐先做固定专家库，再研究动态专家

- `static`：输入固定任务描述、模态标识与 expert ID；每次参数更新后重新生成八张专家相位。
  同一步中所有样本使用同一专家库，光 Router 仍根据样本选择 Top-2。
  评估/部署时生成一次并固化，移除生成器后结果应一致。
  这是对“更换训练方式”的首轮主实验。
- `dynamic`：额外输入当前图像/文本，逐样本生成 `[B, 2, 4, 224, 224]` 专家相位。
  不跨 batch 平均，不读标签或其他测试样本；部署需要执行大模型并刷新相位。
  这是后续单独的研究问题，须另计电子推理、传输与 SLM 刷新成本。

固定专家方案不是跨任务零样本生成：首轮每个任务独立训练一个生成器。
多任务联合训练或新任务快速适配留到首轮机制成立以后。

## 2. 参考包审阅

实际目录为 `TransferFromElectricity/d2nn_pack/`，与消息中的 `d2nn/_pack` 略有不同。
以下为静态代码检查，尚未运行梯度或性能验证。

| 观察 | 代码位置 | 含义 |
|---|---|---|
| 生成相位经过 `exp(1j * phase)`、FFT 传播和探测能量进入 loss，再调用 backward | `model_adapt.py`、`optics.py`、`train_d2nn_adapt_grid.py` | 主计算链支持对生成器反传，可借鉴外部相位注入方式 |
| CLIP 特征提前在 no_grad 下提取，训练时读取 NumPy 缓存；optimizer 仅接收 mask_gen 参数 | `extract_clip_features.py`、`data.py`、`train_d2nn_adapt_grid.py` | 梯度不回到 CLIP；不能称为大模型端到端微调 |
| `masks.mean(dim=0)` 在训练和测试中均使用 | `mask_generator.py` | 同一样本的预测可能随 batch 组成变化；不是稳定固定 mask，也不是独立逐样本 mask |
| Transformer 输入为 `[B, 1, D]` | `mask_generator.py` | 单 token 注意力没有跨 token 信息交互；Transformer 名称不等于利用了预训练大模型 |
| 文档写 sigmoid，但实际返回无约束 logits；默认再做 4-bit STE | `mask_generator.py`、`config.yaml` | 无约束相位本身可行，但当前量化函数对范围的假设不成立，未保证只有 16 个输出等级 |
| YAML 重复定义 optimizer/training；生成器脚本硬编码 lr=1e-3 和 CUDA_VISIBLE_DEVICES=6 | `config.yaml`、`train_d2nn_adapt_grid.py` | 实际设置可能与记录不符，且会覆盖外部 GPU 选择 |
| direct 脚本从 YAML 读学习率，生成器脚本硬编码；direct 相位没有相同的默认量化路径 | 两份 train 脚本与 `model.py` | 不能直接把现有两条命令的结果当成仅训练方式不同的公平对照 |

另外，默认单层的输出 Linear 权重就有 `3072 × 65536 = 201,326,592` 个参数。
新方案用共享 patch 解码头输出完整相位，避免随专家数量堆大 dense 输出层。
MNIST 特征与原图目前按下标配对；新任务必须校验 sample ID、预处理和数据 manifest。

## 3. 光路与生成器设计

两条分支均沿用 LightGenV2 的实际 O/E/O 边界和融合位置：

```text
Vision 输入   → 光 Router → Top-2 expert → CCD/电子处理与重载 → global phase → CCD
Language 输入 → 光 Router → Top-2 expert → CCD/电子处理与重载 → global phase → CCD
                         ↑ 两路各四张生成的 expert 相位
任务描述 + 模态/专家标识 → Qwen（LoRA）→ 共享 patch 相位解码头
```

原有电子支路、RMS 同尺度融合和任务头继续存在；上图只突出光学阶段。
语言任务前端使用的冻结 Qwen 与“生成 mask 的 Qwen”在参数与用途上分离，
防止生成器微调偷偷改变任务特征，从而引入第二个改动因素。

生成器优先复用仓库已有的 `Qwen3-VL-2B-Instruct` 权重，首轮固定任务文本走语言部分，
使用连续 hidden states，不自回归输出数字字符串，不通过 argmax/token 采样生成相位。
建议 LoRA rank=8 起步，具体挂载层与模型 revision 在实现前的环境审计中固化。
模态和 expert 描述产生不同条件，解码头再结合二维位置编码输出 `14×14` 个 `16×16` patch，
拼成每张 `224×224` 的 raw phase；这不是先生成低分辨率 mask 再插值。
两个模态共享生成主干和解码器，用模态/expert 条件区分八张 mask。
沿用目标 backend 的 `2π·sigmoid(raw_phase)` 参数化和物理相位处理。

| 参数/合同 | 首轮参考 |
|---|---|
| 光波长 | 532 nm |
| 逻辑采样间距 | 17 μm；不与相位 SLM 原生 8 μm 混淆 |
| 单段传播距离 | 0.10 m |
| 专家 | 每模态 4 个，2×2 版面，每个 224×224，Top-2 |
| Router | 光学 CCD 能量，power_l2，straight-through Top-2 |
| CCD ROI | 固定 478×478；Router 四个 59×59 积分区域 |
| 传播画布/global phase | 复用实际 backend 的完整几何，不能将其误替换为 224×224 expert 画布 |
| 鲁棒训练 | DC20 profile：20%–30% 相干未调制强度、±16 pixel 位移、0.65° k 空间约束、CCD 噪声及 phase dropout |
| 融合 | 原 RMS 同尺度凸融合，门范围 [0.01, 0.95]，初值 0.055 |
| 正式相位精度 | 首轮双方连续相位；量化另开两组一致的 profile |

实施时导出完全展开的配置和 geometry report，以代码解析值为准；不从历史 README 拼装参数。
T01/T04 的 loss、路由辅助监督和已有权重初始化各有差异，在任务配置中分别保留。
生成相位的 DC 等正则必须作用于实际生成 tensor，不能继续只遍历名为 raw_phase 的 Parameter。

梯度路径：

```text
L_task + L_router + L_CCD + L_phase
  → optical_forward(x, expert_phase=Gθ(c), router=ρ, global=γ)
  → ∂L/∂expert_phase → 相位解码头 → Qwen LoRA
  → 同步直接更新 ρ、γ 与原电子部分
```

expert 输出保持为计算图内 tensor，不写回 `nn.Parameter`、`.data` 或 load_state_dict。
训练期间不缓存已经脱离计算图的生成器 hidden states；可缓存的是未训练的原任务前端。
同一 optimizer step 的各 microbatch 可重新生成相位并累计梯度，更新后重新前向，
不跨 step 重用旧图。首轮关闭生成器 dropout，以便初始化、导出与 batch 一致性审计。
大模型部分拟用 BF16，光学复数传播保持后端的 float32/complex64；显存与 batch 待实测。
这里首先做可微光学仿真；真实 SLM/CCD 读数不会自动提供这条 autograd 链，硬件闭环另立方案。

## 4. 必要对照

| ID | expert 相位来源 | 生成侧训练范围 | 回答的问题 |
|---|---|---|---|
| A `direct` | 直接优化完整 expert raw phase | 无生成器 | 主要基线；Router/MoE/global 全保留 |
| B `small_hyper` | 小型 Transformer + 同类 patch 解码器 | 小生成器与解码器 | 收益是否只来自生成式参数化 |
| C `qwen_frozen` | 冻结 Qwen hidden + patch 解码器 | 仅解码器 | 冻结预训练特征是否已足够 |
| D `qwen_lora` | Qwen + patch 解码器 | Qwen LoRA + 解码器 | 光学 loss 回传到大模型适配参数是否有增益 |

四组均保留相同光路，原电子模块、Router/global、损失和数据协议一致。
D2NN 是结构对照，可在后续补充；不能替代 A，否则会同时改训练方式和 MoE 结构。
不宣称上述四组电子参数量或算力相等；分别报告生成器总/可训练参数、光学储存/激活参数、
训练显存、optimizer steps、样本数和 GPU 时间。先按等 steps 比较，再给性能随时间曲线。
各参数组可用不同学习率，但候选数量与选择预算一致，不能强行共用一个 lr 后宣布优劣。

初始化采用配对控制：相同 seed 下先生成 D 的初始八张 raw phase，A 复制成独立可训练相位，
从完全一致的光学输出和原电子权重起跑。B/C 使用同一初始化目标进行无标签校准，
记录误差与额外成本；达不到预设相位/输出容差时，单列其初始化差异，不称严格同起点。
原有 LightGenV2 初始化的 A 可作为补充稳健性对照，不能只选择有利于 D 的起点。
不以已经训练好的 A 的最终 mask 监督 D 作为主实验；那属于 mask 蒸馏的另一个问题。

## 5. 任务与分阶段运行

### 第一任务：Caltech101 十类检索

复用 T01 的 2,625 train、30 gallery、200 test manifest，64 维读出和检索方式，
主指标 Top-1，附 Top-3/MRR。任务规模适中、已有 vision+language 双分支，适合先验证训练链。
语言 prompt 基本固定，因此不能单凭该任务宣称获得复杂语言条件能力。

### 第二任务：OpenMoji 指令编辑

复用 T04 的 5,000 train、1,000 test；add/replace/move/remove，输出 6×6 类别和编辑网格。
主指标 changed-cell accuracy，附 edit IoU、scene exact match 和分操作指标。
用于检验指令与视觉协同。已有结果接近饱和，除最终性能还应看收敛和样本效率；
若增加组合泛化划分，必须作为新协议单列，不能与原测试集数字直接混合。

第一轮不加入 LGVQ、多视频并行、SALICON 大规模训练或 MNIST 正式性能表。
MNIST 最多用于局部梯度调试，不能代替上述目标结构。

| 阶段 | 具体范围 | 继续条件 |
|---|---|---|
| P0 | 小尺寸无噪声算子检查 + 实际几何一次前反向；32–64 样本过拟合诊断 | 相位、解码头、LoRA、Router/global 梯度与参数更新可追踪；loss 可下降 |
| P1 | T01 A–D，各 1 seed，固定 train 子集、5 epoch pilot | 无断梯度/NaN；无 batch 依赖；记录未使用专家及显存/耗时 |
| P2 | T01 A–D，完整数据，各 3 seeds（42/43/44），40 epoch | 使用锁定配置；12 个正式 run，报告 mean±sample std |
| P3 | T04 A–D，各 3 seeds（73/74/75），40 epoch | 复用已验证的生成模块；12 个正式 run |
| P4 | 按 P2/P3 结果选择 vision-only、language-only、动态生成或量化消融 | 每个实验先写明新增变量与预算，不全面组合扫参 |

默认首批执行只覆盖 P0/P1；跑完给出四组曲线与资源预算，再安排完整实验。
服务器 GPU 型号、空闲卡、模型 revision、依赖、数据与缓存 SHA 尚未在线审计，不预估具体小时数。
首轮从单 GPU 开始，多 GPU 仅在语义和资源需求明确后启用。

### 选模与评价

兼容表遵守 LightGenV2 已有“每 5 epoch 按 test 选 best、无 validation”协议，必须标记
`selection_biased=true / test_used_for_selection=true`。该结果不是无偏泛化估计。
调试/学习率探索只用训练内固定子集；如需证明泛化增益，另建 train 内 val 选模的独立 profile，
并重跑所有对照，测试集仅最终使用。两种协议不得混表。
DC20 默认评估与原任务一致为 deterministic ideal optics；噪声/位移鲁棒性另做固定扰动集评估。

必须附带：每模态每物理专家的硬选择计数、能量分布、专家相位差异、梯度范数和更新幅度、
光电融合 alpha、冻结/打乱专家相位后的性能变化，排查电子分支绕过光学部分。
导出相位后验证生成器可移除且预测一致；eval 模式改变 batch 大小、顺序或陪同样本不能改变单样本预测。
数值梯度验证只针对无噪声连续光学子图；Top-k STE 和量化 STE 不声称是真实离散操作导数。

## 6. 目录与代码边界

以下是评审通过后逐步创建的结构，当前不创建空目录或训练脚手架：

```text
TransferFromElectricity/
  README.md                         # 当前唯一总方案与进度入口
  AI_RULES.md                       # 实施时落地本节目录/同步规则
  paths.example.yaml
  paths.local.yaml                  # 忽略 Git，不含凭证
  d2nn_pack/                        # 原始参考包，不作为新 trainer 后端
  tasks/
    t01_object_retrieval/
      README.md
      configs/                      # direct/small_hyper/qwen_frozen/qwen_lora
      models/                       # expert provider、生成器、后端桥接
      dataset/                      # manifest 引用/哈希，不复制原数据
      tests/
      run.py
      runs/smoke|simulation|hardware/<run_id>/
      reports/                      # 结论、轻量指标、run ID 引用
    t02_semantic_interaction/        # P3 开始时建立，映射 LightGenV2 T04
  common/                           # 第二任务实际复用且接口稳定后才建立
```

优先调用 LightGenV2 及其既有兼容后端，不复制整套网络或继续向 experiments 散落文件。
设计统一 `ExpertPhaseProvider`：direct 和 generated 走相同相位消费/光学传播路径。
若现有后端缺少外部相位接口，先用本任务内适配；确需修改公共接口时只加可选参数，
默认行为不变，并做旧模型输出与梯度回归检查。
普通实验差异放配置，不创建 new/new2 或日期版模型目录。

## 7. 同步、记录与存储

遵循 [LightGenV2/AI_RULES.md](../LightGenV2/AI_RULES.md) 的目录、产物和 Git 原则。
用户已批准固定专家库与 Caltech101 的 P0/P1；后续任务仍按阶段安排。

1. 实施前先在本地与服务器读取 status、HEAD、分支和依赖清单；fetch 成功后确定源码起点。
2. 如有未提交改动或分叉，使用独立分支/必要时隔离 worktree，不覆盖或清空用户改动。
3. 源码测试后 commit、push GitHub；服务器仅 checkout 已发布 SHA 或在干净工作树 pull --ff-only。
   不用 SCP、OneDrive 文件覆盖来同步源码；禁止 force push/reset --hard。
4. 缓存/权重按 manifest+SHA256 传输；原数据、Qwen 权重和可复用缓存优先引用服务器已有副本。
5. 新项目在首次 add 前配置忽略规则，特别覆盖参考包现有 `.npy` 特征、paths.local、缓存及权重。
   当前参考包整体尚未跟踪，不能直接 `git add TransferFromElectricity` 把特征一并上传。
6. run ID 建议 `YYYYMMDD_<task>_<static|dynamic>_<method>_<profile>_s<seed>`；禁止覆盖旧 run。
7. 每 run 保存实际配置、命令、Git SHA、模型 revision/权重 SHA、数据 manifest、环境、seed、状态、
   指标日志、参数口径和梯度诊断。断点需含生成器/LoRA、Router/global、电子模块、optimizer、
   scheduler、EMA、随机数和训练位置，能够恢复同一训练过程。
8. 权重只保留 best/last；导出固定 expert 相位作为具备 SHA 的部署产物，不周期复制大模型。
   LoRA checkpoint 引用冻结基座 revision，不在每个 run 复制完整 Qwen。
9. 服务器保留权重/完整日志，本地同步轻量报告、manifest 和必要预览，不镜像全盘。
10. SSH 凭证仅用于连接，不写进代码、配置、README、命令清单或 Git。

审批范围建议：先实现固定专家方案，Qwen LoRA 更新生成侧，T01 四组对照完成 P0/P1，
确认机制和资源后扩展正式多 seed 与 T04。若目标本来就是逐输入动态生成，应在实施前将 dynamic
设为主方案，并相应调整公平对照与部署计时边界。
