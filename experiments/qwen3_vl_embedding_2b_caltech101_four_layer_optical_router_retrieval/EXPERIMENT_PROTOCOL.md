# 电子/光学 Router 实验协议

本文规定本工程的实验设计、checkpoint 选择和结果记录方式。它的首要目的不是
“尽可能找到最高的 test 数字”，而是保证下面三个问题不会混在一起：

1. 同一个已训练模型在推理时激活 1、2、4 个专家，会发生什么；
2. 给每个方案相同适配预算后，哪一种专家数更合适；
3. 在相同 Top-2 和功率合同下，电子 Router 能否被光学 Router 替代。

三个问题必须分别报告。禁止从其中一组的测试结果反过来修改另一组配置。

### 当前实现范围（2026-09-02）

本版本已经实现并作为正式主实验支持：

- `F-L2`：旧电子 Router、Top-2、`legacy_l1` 的零训练步兼容性 anchor；
- `A-E1/A-E2/A-E4`：电子 Router 在 Top-1/2/4 下的等预算适配；
- `A-O2`：光学 Router Top-2 与 `A-E2` 的等预算适配对照。

`F-L1/F-L4/F-P1/F-P2/F-P4` 只是可选的后续 fixed-inference 研究设计，当前 release
配置、命令和结果产物并未实现它们。不得把这些预留 ID 写成“已经完成”，也不得用
`A-E1/A-E2/A-E4` 的训练后 checkpoint 冒充 fixed-inference 结果。当前主结论应来自
`A-E1/A-E2/A-E4` 的正式电子适配消融，以及 `A-E2` 对 `A-O2` 的光/电比较。

## 1. 固定来源与数据合同

所有实验以同一个已封存的 warmstart5 模型为来源：

```text
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/
  runs/caltech101_warmstart5_stage2_joint_sealed_test/
  ema_best_train_loss_checkpoint.pt

SHA-256:
6a27f54d8c869cce46150583383a127b0ba47b3d34503f5753aa23974ac1e55d
```

该权重为 epoch 8 的 EMA/train-loss-best checkpoint。它没有使用 test 指标选 epoch，
对应的已封存结果是 Top-1 `0.8100`、Top-3 `0.9300`、MRR `0.876345`。

所有变体必须共享：

- 完全相同的 Caltech101 十类清单、train/gallery/test manifest 和 manifest SHA；
- 30 张 gallery 图和 200 张 test query；
- 同一个 Qwen model ID、processor cache、prompt 和 64 维检索定义；
- 相同的 518×518 传播画布、478×478 CCD ROI、17 μm、532 nm 和 10 cm；
- 相同的四层 feature optics、电子 Mixer、CCD readout、融合门和 retrieval head来源；
- 独立的输出目录，禁止多个任务写入同一个 `train_log.csv` 或 metrics 目录。

如果来源 checkpoint、数据 manifest 或上述物理合同任意一项发生变化，应建立新的
实验批次，不能继续填入本文所定义的同一张结果表。

## 2. 两类实验不能混为一谈

### 2.1 固定 checkpoint 推理消融

这个实验回答：

> 不再学习任何参数，只改变一次前向中的稀疏路由规则，激活专家数会怎样影响现有
> warmstart5 模型？

要求如下：

- optimizer step 必须为 0；
- 除运行时 Router 的 `top_k` 和明确声明的权重归一化外，所有 tensor 逐位相同；
- 不允许根据 test 结果挑 checkpoint、温度、归一化或随机种子；
- Top-2 legacy anchor 必须先复现原 checkpoint 的 `0.8100` Top-1；
- k=1/2/4 必须在执行前一次性预声明，随后每组只运行一次 test；
- 该组结果只能称为“fixed-checkpoint inference ablation”，不能称为训练后的最优架构。

当前代码只把 `F-L2` 做成可执行的零步兼容性 anchor。下面关于 k=1/2/4 固定推理的
要求定义的是后续扩展协议，并不表示本版本已具备 F-L1/F-L4/F-P1/F-P2/F-P4 的配置
或 CLI。正式电子 Top-k 主消融请执行第 2.2 节的等预算适配组。

固定推理建议同时报告两个子口径：

| 子口径 | 目的 | 注意事项 |
|---|---|---|
| `legacy_l1` | 完全沿用旧振幅权重语义 | k 改变时总光功率也会变化 |
| `power_l2` | 令每个样本 `sum(amplitude_scale^2)=1` | Top-2 不再等同于旧模型的原始振幅尺度 |

这两组不能交叉排名。`legacy_l1` 回答部署兼容性，`power_l2` 回答功率匹配后的敏感性。

正式 warmstart5 checkpoint 的 architecture 标签与新变体标签不同，而 shared loader
会严格比较标签。因此固定推理变体必须通过新工程的“只移植权重、零训练步”入口生成
带有完整来源 SHA 的变体 checkpoint；禁止修改旧 checkpoint 文件本身，也禁止通过
关闭 architecture 检查来偷偷加载。

光学 Router 新增了不存在于来源 checkpoint 中的 Router phase，因此它没有严格意义
上的固定 checkpoint 推理对照。可以报告其解析初值的诊断结果，但不能与已经训练过的
电子 Router 固定推理结果作为正式优劣结论。

### 2.2 等预算适配消融

这个实验回答：

> 从同一个已封存模型主体出发，给予相同训练数据和 optimizer step 后，k=1/2/4 或
> 光/电 Router 哪个更适合当前任务？

正式电子 k 消融为：

```text
electronic + power_l2 + STE + k=1
electronic + power_l2 + STE + k=2
electronic + power_l2 + STE + k=4
```

光/电主对照为：

```text
electronic + power_l2 + STE + k=2
optical    + power_l2 + STE + k=2
```

共同预算固定为：

| 项目 | 固定要求 |
|---|---|
| 起点 | 上述同一 checkpoint SHA |
| batch | 30，即 10 类×每类 3 张 |
| epoch | 12 |
| optimizer steps/epoch | 12 |
| 总 optimizer steps | 144 |
| 每轮 model-forward 样本 | 360 |
| test during training | 禁止 |
| checkpoint 选择 | 最小 training total loss 的 EMA 权重 |
| phase-focus 交替 | 关闭，保证每组每一步都能更新 Router |
| 训练噪声与数据增强 | 所有变体完全一致 |
| feature phase LR | 所有组固定为 `0.006` |
| electronic Router LR | `A-E1/A-E2/A-E4` 固定为 `0.0002` |
| optical Router phase LR | `A-O2` 预声明为 `0.01` |

这里的“等预算”指相同数据前向数和 optimizer step，不代表参数量或物理曝光数相同。
光学 Router 有更多 phase 参数并增加两次物理 Router 曝光，这些差异必须作为结果的
一部分报告，不能隐藏。

电子 gate 与 sigmoid 参数化的 phase 不是同一种参数，故允许使用预声明的架构专属 LR；
这些 LR 必须在打开 sealed test 前冻结，不得根据 test 结果再调。

### 2.3 Router 初始化公平性

旧 checkpoint 中的电子 Router 已经训练过，而光学 Router phase 是新增参数。若电子
适配组直接保留旧 gate、光学组从新 phase 开始，则电子组具有额外的 Router 预训练
优势。

因此正式主表推荐采用下面的合同：

- legacy anchor 和固定推理组保留旧电子 gate；
- 等预算适配组只移植共同模型主体；
- 三个电子 k 变体的 gate 用同一 seed 按同一规则重新初始化；
- 光学 Router phase 使用解析、确定性的四束初值；初始化本身不添加随机微扰。训练期的
  shift、phase dropout 等随机扰动由该 run 的固定 `random_seed` 控制；
- 结果中分别记录 `router_pretrained` 和 `router_initialization`。

如果实现阶段决定让电子适配组保留已训练 gate，也可以作为“continued adaptation”结果
报告，但必须把 `router_pretrained=true` 写入结果，且不得把它称为严格公平的光/电
Router 主对照。

## 3. 为什么正式训练必须使用 STE

普通 hard top-k 的计算为：

```text
softmax probabilities
→ 保留 top-k
→ 所选值重新归一化
```

当 k=1 时，唯一所选权重恒等于 1；离散索引本身不可导，所以 retrieval loss 无法训练
Router。正式适配实验统一使用 straight-through estimator：

- 前向仍是严格 hard top-k，未选专家精确为零；
- 反向使用 dense soft probability 的 surrogate gradient；
- k=1/2/4 都使用同一个估计器，避免只给某一组额外的梯度优势。

固定 checkpoint 推理时不发生反向传播，因此 STE 开关不应改变前向数值。

## 4. Shared trainer 兼容合同

新工程复用 shared trainer 时必须满足以下条件。

### 4.1 动态 checkpoint architecture

`save_checkpoint()` 会把 `replacement.checkpoint_architecture` 写入 metadata；
`load_checkpoint()` 会严格比较保存值和当前值。当前实现的字符串为：

```text
vision2_language2_moe4_10cm_router_ablation_
{router_backend}_k{top_k}_{weight_normalization}_{ste|noste}_v1
```

它显式编码 backend、k、权重归一化和 STE。`protocol` 未单独写进 architecture 字符串，
但 `settings.py` 对当前有效组合施加强约束：legacy 只能是 electronic/k2/legacy_l1/
no-STE/no-reset，等预算适配只能是 power_l2/STE/reset。因此当前五份 release 配置不会
发生 architecture 冲突。未来若实现 fixed power_l2 或其他 protocol，必须先升级
architecture 版本并加入 protocol/reset 语义，不能复用现有 `_v1` 标签。

训练、resume 和 evaluate 必须用同一份配置解析出完全相同的字符串。不同 k、不同
Router backend、不同 normalization 或不同 STE 的 checkpoint 禁止互相 resume。

从 warmstart5 导入属于“初始化移植”，不是 resume：

- 必须核验来源 SHA 和 `test_metrics_used_for_selection=false`；
- 只移植同名同形状且属于共同主体的 tensor；
- 电子 gate 是否移植必须遵守第 2.3 节并写入初始化报告；
- optical Router 新 phase 不得伪装成来源 checkpoint 已包含的 tensor；
- 不恢复旧 optimizer state。

### 4.2 Router 参数组

shared optimizer 优先调用 `replacement.router_parameters()`。电子 Router 的
`Linear(196,4)` weight/bias 应进入 `routers` 参数组并使用 router LR。

光学 Router 的参数刻意命名为 `raw_router_phase`，而不是原四层 feature mask 使用的
`raw_phase`。这是参数所有权合同，不只是命名习惯：

- shared trainer 按名称把所有普通 `raw_phase` 放入 `optical_phases` 参数组；
- `raw_router_phase` 不会被这次名称扫描重复收集；
- `replacement.router_parameters()` 返回两张 `raw_router_phase`，令它们进入 `routers`
  参数组并使用独立的 router LR；
- `phase_parameter_groups()` 仍将它们分别登记为 `vision_router`、`language_router`，这只
  用于 phase gradient、运动量和缺失梯度诊断，不改变 optimizer 所有权；
- 如果误把它重新命名为 `raw_phase`，它会同时出现在 router/phase 两组，shared trainer
  应立即以参数组重叠报错；
- phase-focus 必须关闭，否则“等 optimizer step”不等于“等 Router 更新步数”。

因此当前 `A-O2` 的 Router phase 实际学习率等于配置中的 `router_learning_rate`，不是
`phase_learning_rate`；后者只属于原四层 feature phase。结果字段
`router_phase_learning_rate` 必须记录这个实际 optimizer-group 值。

参数组审计报告至少要证明：

```text
所有 requires_grad tensor 恰好进入一个 optimizer group
phase IDs 与 router IDs 无交集
电子 Router gate gradient 非零且有限
光学 Router raw_router_phase gradient 非零且有限
```

### 4.3 Phase artifact 与 capture loss

shared phase snapshot 默认只认识原四张 expert/global phase；本工程已经通过
`RouterAblationReplacement.save_multiplane_phase_snapshot()` 和
`save_multiplane_phase_preview()` 覆盖该入口。光学 Router 运行必须核验 snapshot/preview
中同时包含：

- 原四张 feature expert/global phase；
- Vision/Language 两张 `router_physical_phase_rad`；
- snapshot JSON 中两张 Router phase 的 tensor 统计量；
- preview 的 Vision/Language Router phase 面板。

相对 run 初值的位移和梯度不属于 snapshot JSON：它们由 shared phase diagnostics 根据
`phase_parameter_groups()` 中的 `vision_router/language_router` 计算并写入 train log。
`phase_parameters.pt`、最终导出 BMP 的 SHA 也必须由独立 audit/result manifest 计算，
不能误称为 snapshot 已经自动提供。

若任一 Router phase 缺失，不能把该 run 标为完成，也不能只用原 feature-mask preview
替代 Router 证据。

当前光学 Router 将 capture loss 加入 `balance_loss` 后再交给 trainer。实际进入总损失的
系数为：

```text
effective_capture_weight
= lambda_router_balance × optical.capture_loss_scale
```

结果表必须记录这个有效系数，不能只写内部 `capture_loss_scale`。

### 4.4 Evaluate 输出

shared evaluate 会直接读取 `bundle.test_samples` 和 gallery，并把文件写入当前
`output_dir`。因此：

- 每个变体必须有唯一 output directory；
- evaluate 必须显式指定预声明 checkpoint；
- 禁止使用文件名含 `observed_test` 的 checkpoint；
- 禁止把 checkpoint 改名来绕过 selection-bias 标记；
- evaluate 前核对 checkpoint architecture、来源 SHA、数据 manifest SHA；
- evaluate 完成后记录 checkpoint SHA，之后不得根据结果换 epoch 再测。

shared trainer 当前能可靠保存 train-loss-best/EMA 权重，但没有独立 development split
选模入口。因此本批实验统一使用“最小 training total loss 的 EMA checkpoint”，不能在
文档中误写成 validation-best。将来若增加 development split，必须在查看 test 前冻结
划分并为整批实验建立新协议版本。

## 5. Sealed-test 规则

每个预声明变体都必须遵守：

1. `evaluate_test_each_epoch=false`；
2. 训练日志中的 `test_top1/test_top3/test_mrr` 应为空或 NaN；
3. checkpoint 只按 training total loss 选择，优先使用对应 EMA 版本；
4. 所有超参数、运行种子和待测 checkpoint SHA 在 evaluate 前冻结；
5. test 只用于一次最终报告，不用于提前停止、选 epoch、调温度、改 ROI 或改 loss；
6. 某组 test 较差时，必须原样报告；若据此修改配置，修改后的运行属于下一批探索实验，
   不能覆盖原结果；
7. `best_observed_test_checkpoint.pt` 和 `ema_best_observed_test_checkpoint.pt` 永远不能
   进入正式结果表；
8. test 结果的知晓不会自动消失。若后续进行大量调参，应从 train 固定划出 development
   set，并把当前 200 张 test 继续封存。

建议 evaluate 前生成一个只读的 `evaluation_intent.json`，至少写入 variant ID、config
SHA、checkpoint SHA、manifest SHA、日期和 `test_open_count_before=0`。

## 6. 预声明运行矩阵

### 6.1 当前可执行的固定 checkpoint 组

| ID | Router | k | 权重合同 | 训练步数 | 用途 |
|---|---|---:|---|---:|---|
| `F-L2` | electronic | 2 | legacy_l1 | 0 | 必须复现 81% 的 anchor |

`F-L2` 由 `materialize_initialization` 产生动态 architecture 对应的零步 checkpoint，再用
显式 `--checkpoint` 执行一次 sealed-test evaluate。它不应进入训练循环。

以下 ID 保留给未来另行实现的 fixed-inference 扩展，不属于当前运行矩阵：

| 预留 ID | Router | k | 权重合同 | 状态 |
|---|---|---:|---|---|
| `F-L1` | electronic | 1 | legacy_l1 | 未实现 |
| `F-L4` | electronic | 4 | legacy_l1 | 未实现 |
| `F-P1` | electronic | 1 | power_l2 | 未实现 |
| `F-P2` | electronic | 2 | power_l2 | 未实现 |
| `F-P4` | electronic | 4 | power_l2 | 未实现 |

只有补齐独立配置、动态 architecture、零步移植和显式 evaluate 命令后，这些 ID 才能
进入结果表。当前不得声称已经完成 fixed-inference k=1/4 或 power_l2 推理消融。

### 6.2 等预算适配组

| ID | Router | k | 权重合同 | STE | 总步数 | 主比较 |
|---|---|---:|---|---|---:|---|
| `A-E1` | electronic | 1 | power_l2 | on | 144 | k 消融 |
| `A-E2` | electronic | 2 | power_l2 | on | 144 | k 消融、光/电基线 |
| `A-E4` | electronic | 4 | power_l2 | on | 144 | k 消融 |
| `A-O2` | optical | 2 | power_l2 | on | 144 | 与 A-E2 比较 |

首轮可先运行一个固定 seed。只有在确认所有合同和梯度正确后，才对正式候选运行至少
3 个 seed。多 seed 结果应报告 mean、standard deviation 和每个 seed 原值；不能只挑
最好的一次。

## 7. GPU 并行计划

按当前服务器空闲情况，首轮等预算适配分配为：

| GPU | 任务 | 说明 |
|---:|---|---|
| 0 | `A-E1` | electronic top-1 |
| 1 | `A-E2` | electronic top-2 control |
| 3 | `A-E4` | electronic top-4 |
| 4 | `A-O2` | optical top-2 |
| 5 | 第二 seed 或审计任务 | 3090，先确认显存；当前无 F-L1/F-L4/F-P* 正式任务 |

GPU 2 和 6 在计划制定时有其他负载，不应抢占。启动前仍必须重新执行 `nvidia-smi`；
本表不是永久设备所有权。

并行任务必须满足：

- 独立 config 和 output directory；
- 不共享可写的 checkpoint、train log 或 phase artifact 目录；
- 数据/cache 只读共享；
- 每条命令保存 PID、GPU、启动时间、git commit 和 resolved config SHA；
- 某任务失败不能用其他任务的 last checkpoint 续跑。

## 8. 必须记录的结果字段

### 8.1 来源与选择

```text
experiment_batch_id
variant_id
protocol                    # fixed_inference / equal_budget_adaptation
git_commit
config_path
config_sha256
data_manifest_sha256
source_checkpoint_path
source_checkpoint_sha256
source_checkpoint_epoch
source_test_used_for_selection
router_pretrained
router_initialization
selected_checkpoint_path
selected_checkpoint_sha256
selected_checkpoint_epoch
selection_criterion
test_metrics_used_for_selection
test_open_count
random_seed
```

### 8.2 模型和训练

```text
router_backend
top_k
weight_normalization
straight_through
router_temperature
trainable_parameters_total
trainable_router_parameters
trainable_feature_phase_parameters
trainable_router_phase_parameters
frozen_qwen_parameters
epochs
optimizer_steps_per_epoch
optimizer_steps_total
model_forward_samples_total
best_train_total_loss
router_learning_rate
feature_phase_learning_rate
router_phase_learning_rate
effective_capture_loss_weight
```

### 8.3 检索与 Router

```text
test_query_count
gallery_image_count
top1
top3
mrr
per_class_top1
confusion_matrix_path
mean_active_experts
router_entropy_vision
router_entropy_language
expert_load_vision_0..3
expert_load_language_0..3
top1_margin
topk_boundary_margin          # 第k名减第k+1名；k=4记N/A
route_stability_shift
route_stability_gain
route_stability_noise
```

### 8.4 光学 Router 专属

```text
vision_capture_fraction_mean
vision_capture_fraction_p05
language_capture_fraction_mean
language_capture_fraction_p05
router_detector_intervals
router_phase_std_rad
router_phase_delta_from_init_rad
router_phase_gradient_rms
router_phase_bmp_sha256
electronic_top1_agreement
electronic_topk_jaccard
router_ccd_exposures_per_sample
feature_ccd_exposures_per_sample
end_to_end_latency_ms
```

注意：shared trainer 的默认 train log 不包含上述全部光学 Router 字段。缺失字段必须在
独立 audit/report 中计算，不能留空后假装“未发现问题”。

## 9. 结果表模板

### 9.1 固定 checkpoint 推理消融

| ID | Router | k | weighting | steps | Top-1 | Top-3 | MRR | entropy V/L | power contract passed | checkpoint SHA |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|
| F-L2 | electronic | 2 | legacy_l1 | 0 | | | | | | |

F-L1、F-L4、F-P1、F-P2、F-P4 在当前版本中保持未实现，不填入数值。未来实现时必须
新建协议批次和预声明结果表，不能回填本表。

### 9.2 等预算适配

| ID | Router | k | router pretrained | steps | selected epoch | Top-1 | Top-3 | MRR | mean active | route stability | capture fraction |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A-E1 | electronic | 1 | | 144 | | | | | | | N/A |
| A-E2 | electronic | 2 | | 144 | | | | | | | N/A |
| A-E4 | electronic | 4 | | 144 | | | | | | | N/A |
| A-O2 | optical | 2 | no | 144 | | | | | | | |

## 10. 结果解释边界

- 当前固定 checkpoint 组只有 F-L2 兼容性 anchor，不能据此得出 k 敏感性结论；只有
  未来完成预声明的 F-L1/F-L4 或 F-P1/P2/P4 后，才能讨论固定模型对 k 的敏感性。
- 等预算适配组是 warm-start adaptation，不是从随机初始化开始的全量重训。
- k=4 时所有专家均被选择，selection load 和 balance loss 的解释与 k=1/2 不同；应更
  关注连续权重、entropy、串扰和最终检索指标。
- 电子 Router 只有 1,576 个 gate 参数；光 Router 有两张 224×224 phase，共 100,352
  个 Router phase 参数。参数量不同必须报告。
- 四个专家为空间并行，k 减少不会缩短 10 cm 自由传播时间；它主要改变幅度分配、
  串扰、容错和 SLM 有效照明面积。
- 光学 Router 不增加器件和 ROI，但增加 Vision/Language 各一次 Router SLM/CCD 时序。
  仿真准确率相同不等于硬件吞吐相同。
- 任意 test 结果一旦被用于决定下一次配置，下一次只能标为 exploratory，不得回填到
  本批预声明主表。
