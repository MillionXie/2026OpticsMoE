# T04 语义交互（OpenMoji）

## 2026-09-09：精简共享电子读出头

保守对照为 `routerfill_slim_norm` / `qwen_slim_norm`：删除相同卷积及前置FiLM，
但保留decoder的GroupNorm+GELU，仅比slim多384参数，合计269,272（比旧头少29.51%）。
使用下方命令时替换profile即可；输出目录同profile加`_s73`。两对均100epoch，保留全部结果，
最终必须成对比较相同头，不能从不同头中挑一组主方法和一组baseline拼表。

`routerfill_slim` 与 `qwen_slim` 使用同一个精简头：381,976 → 268,888 参数（减少29.61%）。
只删去两组条件卷积前的重复FiLM和解码器预处理卷积；保留两组条件卷积、位置线性汇总、
坐标映射、6×6平均降采样、类别/编辑输出以及训练用四操作辅助头。光路、光Router Top2、
同尺度融合alpha>0.4和原噪声配置均不变。没有新增TF/attention或其他分支。

两组均从头训练任务网络100epoch，seed73、batch32、相同v2数据和损失/学习率协议，
不从旧best截取部分权重。Qwen主干完整冻结，复用经过校验的原生特征缓存；双方共享头
初始权重相同、训练后独立。只保留best/last，光电组结束自动生成同best去光对照及相位图。

```bash
# 仓库根目录；服务器只运行已推送的源码。此限制用于多进程特征缓存读取。
ulimit -n 65536
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile routerfill_slim --phase all --device cuda
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile qwen_slim --phase all --device cuda
```

输出分别为 `runs/simulation/routerfill_slim_s73`、`runs/simulation/qwen_slim_s73`，新结果待训练完成。
旧共享头正式100轮结果：标准光电修改格87.15%、IoU0.8327、F1 0.9339、整场景69.00%；
强均衡84.85%、0.8993、0.9478、77.00%；冻结Qwen84.20%、0.7312、0.8757、54.20%。
旧标准光电同best去光降至40.55%（46.60个百分点），不解释为物理贡献百分比。
当前精简只针对末端读出头，不代表光电网络全部电子参数只有26.9万。

## 2026-09-08：语言Router铺满孔径 + 公平共享读出头

当前入口为 `routerfill_shared`（两组卷积、光Router）、`routerfill_shared_balance`（同结构、更强均衡正则）和
`qwen_shared`（完整冻结Qwen、相同读出头）。旧lean已经完成100轮，修改格89.40%，但语言Router固定使用专家0/1，
不满足专家均衡要求。新旧run不混用；详见[新修复及公平对照合同](reports/reproduction/ROUTER_SHARED_HEAD.md)。

```bash
# 主模型 / 相同架构的强均衡候选：分别在可用GPU上运行
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile routerfill_shared --phase all --device cuda
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile routerfill_shared_balance --phase all --device cuda
# 完整冻结Qwen baseline：自动提取原生特征，再用同一个训练器训练共享结构的读出头
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile qwen_shared --phase all --device cuda
```

三个profile都是5000train/1000test、同一v2清单、100epoch、seed73。后端完全复用 `SharedGridReadout`，
参数量381976、初始权重逐位一致（训练后各自独立）。主模型仍无TF/attention；**完整Qwen仅作为baseline执行**。
baseline只增加必要的1024/2048→192线性适配和LayerNorm，无LoRA、额外卷积支路或额外任务prompt。
本轮只测性能；不把训练服务器的耗时/功耗充作5090D实测。

## 当前优化入口：2026-09-08 embedding-only / alpha > 0.4

新 profile 为 `embedding_alpha40`（光 Router Top-2）、`embedding_alpha40_lean`（末端卷积由3组减为2组）和
`embedding_d2nn_alpha40`（普通 D2NN 对照）。**下文98%的历史结果不属于这些新模型。**
新模型文本只做冻结 Qwen 词表查表和正弦位置编码，不使用语言 TF 缓存；视觉只用冻结 Qwen patch/位置前端。
两个模态都各有光 Router → expert → global（D2NN 对照没有 Router）；四个光电融合 alpha 均约束在
`[0.4001, 0.95]`，初始0.60。语言 global 后采用固定位置系数的可学习线性汇总，不使用 mean/max 拼接或 attention。
新模型随机初始化任务网络，不继承旧语言TF教师或旧任务头。每个光阶段只保留一条电子残差和一条光支路；
末端仍有指令条件卷积和类别/编辑双输出头。详见[新架构及运行说明](reports/reproduction/EMBEDDING_ALPHA40.md)。

运行（从本仓库根目录，先 prepare 一次，再按可用显卡分别启动）：

```bash
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile embedding_alpha40 --phase prepare --device cpu
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile embedding_alpha40 --phase all
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile embedding_alpha40_lean --phase all
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile embedding_d2nn_alpha40 --phase all
```

下面是保留供核对的旧协议。

输入是 `224×224` OpenMoji 场景和文本指令，指令任务为 `add / replace / move / remove`；输出是 `6×6` 类别网格和编辑网格，再由固定 OpenMoji 合成器得到目标图像。

论文口径、baseline差异和复现命令统一见 [reports/reproduction/README.md](reports/reproduction/README.md)。
本任务是自行构造的封闭词表网格编辑，不是OpenMoji官方benchmark。当前文本缓存经过完整冻结语言Transformer，
最终保留合成使用真实`source_grid`，不能描述为纯embedding前端、完全无Transformer的RGB+文本端到端系统。
2026-09-08审计发现1对train/test完全重复输入；历史数值保留，后续去重必须使用新split/profile。

**2026-09-08三组完整复评已完成**：[全部指标、去光、相位/冗余审计与通俗数据流说明](reports/reproduction/RESULTS_20260908.md)。
本次修改格准确率：MoE0.9800、D2NN0.9890、Qwen在线0.5390；旧数值仍保留于下文，不覆盖历史。
MoE去光后为0.9850，但整场景从0.8950降至0.8760；四个alpha约5%。
旧版语言TF缓存不满足当前禁TF要求，且CCD归一化含clip/log；这些是待修正旧架构成绩，不是合规新版本成绩。

本目录固定比较三组系统：

1. `main_dc20`：语言、视觉各含光学 Router，均为 4 专家 Top-2；随后各有一张 global phase。融合前做 RMS 同尺度归一化，训练含 20%–30% 相干零级分量与硬/软专家均衡。
2. `d2nn_dc20`：没有 Router；语言和视觉各使用两层普通 D2NN。四张 `224×224` dense 相位与主方法两个模态实际激活的四张专家相位参数量相等。
3. `qwen_pending`：历史待测合同入口，不执行大模型评估。实际冻结Qwen加训练任务头baseline由`baseline_structured_5090d.py`运行；历史结果见下文，复现缺口见复现说明。

主指标为 changed-cell accuracy，并同时报告 foreground category accuracy、edit-grid IoU、object F1、scene exact match 和按四种操作分组的指标。

## 数据协议

- train：5,000 个合成场景，每个操作 1,250 个。
- test：1,000 个不同随机种子的场景，每个操作 250 个。
- train/test 同分布、样本种子不相交；validation 为无。
- epoch 1、每 5 epoch、末轮测试；按最高 test changed-cell accuracy 选择 checkpoint。
- 正式目录只保留 `best_checkpoint.pt` 和 `last_checkpoint.pt`。

## 运行

```bash
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile main_dc20 --phase all
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile d2nn_dc20 --phase all
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile qwen_pending --phase all
```

大模型 baseline、零样本生成诊断、计时和功耗口径见
[BASELINE_5090D_TODO.md](BASELINE_5090D_TODO.md)。正式 baseline 冻结 Qwen 原生
Vision/Language Transformer，只训练普通结构化任务读出头；旧的自由生成 JSON 结果只作为
zero-shot diagnostic，不写入论文 baseline 行。

RTX 5090 D 正常 baseline 已完成：输入 224×224 图像和完整指令，完整执行冻结的原生
Vision/Language blocks，只训练 1,212,434 参数的结构化任务头。5000 train 训练 50 epoch，
每 5 epoch 测一次完整 1000 test，并按 changed-cell accuracy 选择 epoch 20。正式结果为
changed-cell **0.5475**、foreground category **0.2168**、edit IoU **0.2909**、object F1
**0.1666**、scene exact **0.0160**；第一个 Vision block 到 `6×6` 两个输出的
mean/median/P95 为 **27.166/26.628/30.280 ms/sample**。四任务 changed-cell 分别为
add 0.224、replace 0.336、move 0.650、remove 0.980。该结果没有 LoRA、没有主干微调、
没有自回归生成，也没有为抬数值加入额外 loss 或增强。

## 正式单次结果（seed 73）

- 光 Router Top-2：selected checkpoint 正式复评 changed-cell accuracy 0.9800、
  foreground category 0.9895、edit IoU 0.9350、object F1 0.9837、scene exact match 0.8950。
- 参数匹配 D2NN：changed-cell accuracy 0.9895、foreground category 0.9944、
  edit IoU 0.9813、object F1 0.9949、scene exact match 0.9650。
- 主方法语言 Router 使用 3/4 专家，选择占比 50.00% / 25.00% / 25.00% / 0%；
  视觉 Router 也使用 3/4，选择占比 20.30% / 45.80% / 33.90% / 0%。

因此该结果满足“物理光 Router、Top-2”的结构要求，但不能宣称四专家完全均衡；D2NN 的
主指标高 0.95 个百分点。机器可读结果和可视化见 `reports/dc20_comparison/`；其Qwen pending行为早期快照，不代表后来的baseline未运行。
