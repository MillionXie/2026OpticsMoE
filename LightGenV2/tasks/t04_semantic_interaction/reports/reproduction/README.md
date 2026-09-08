# OpenMoji：复现入口与论文口径审计

审计日期：2026-09-08。任务状态仍见[任务 README](../../README.md)。本次读取本地代码、
guest3 服务器实际数据及 run 元数据，**没有重训或重新评估模型**。
审计时本地/服务器源码均为 `8207fa995c242e15b7ea0c2353b3950ba857cc0f`。
本目录记录稳定复现协议；权重、日志、逐样本预测仍留在任务 runs，不复制进 Git。

## 1. 论文里的任务及指标

建议名称：**基于 OpenMoji 的封闭词表、文本指令驱动网格编辑**。
这是项目自行生成的受控任务，不是 OpenMoji 官方标准准确率 benchmark，也不是开放词表图像生成。

- 输入：224×224 RGB 场景和一条文本指令；6×6 网格，16 种固定图标及空白类，通常1–4个对象。
- 操作：add、replace、move、remove；每种操作3种语言模板。
- 输出：6×6 类别 logits 与编辑 logits，离散化后由固定图标合成器输出图像。
- 数据清单记录素材为 OpenMoji 17.0.0 官方72×72彩色PNG、CC BY-SA 4.0；发布需保留署名/许可。

**重要输出约定**：主方法和结构化 Qwen baseline 均使用

```text
generated = argmax(category_logits)
predicted_edit = sigmoid(edit_logits) >= 0.5
prediction = where(predicted_edit, generated, source_grid)
```

`source_grid` 是数据集提供的真实源类别网格，不是模型从 RGB 识别出的网格。
它不进入主方法神经网络 forward，却参与最终输出。不是使用 target_grid 生成预测，
但属于额外结构化源信息，不能描述为“完全只依赖 RGB 与文本的端到端场景重建”。
foreground/F1/exact 会受到保留策略帮助；所有比较须使用一致合成口径。

changed-cell accuracy 对每个样本只在**真实修改格子**上计算正确率，再按样本平均。
add/replace/remove 各改1格，move 改2格；不是全图像素准确率，也不是98%的场景全部正确。

| 方法 | Changed-cell | Edit IoU | Scene exact | 证据状态 |
|---|---:|---:|---:|---|
| 光 Router Top-2 | 0.9800 | 0.9350 | 0.8950 | 已归档仿真复评，本次未重跑 |
| 激活专家参数匹配 D2NN | 0.9895 | 0.9813 | 0.9650 | 已归档仿真复评，本次未重跑 |
| 冻结 Qwen + 结构化任务头 | 0.5475 | 0.2909 | 0.0160 | README/5090D文档历史记录，本次未取回原run复评 |

前两行见 [comparison.json](../dc20_comparison/comparison.json)。它的 Qwen pending 行是早期快照，
不能与后来的 baseline 文档当成同一次测量。Qwen 历史 changed-cell 分项：
add 0.224、replace 0.336、move 0.650、remove 0.980。这值得诊断输出头/监督差异，不能直接归因于语义能力不足。
D2NN 比主方法高 **0.95 个百分点**，scene exact 高7.00个百分点，必须如实保留。

## 2. 实际数据核对

服务器仓库：`/DATA/DATA1/guest3/2026OpticsMoE`。
数据：`experiments/qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing/data/pilot_gpu`。

| 项目 | 核对结果 |
|---|---|
| train / test | 5000 / 1000；各操作1250 / 250 |
| validation | 无；周期test用于选模，也用于先前方案选择 |
| train/test seed交集 | 0 |
| test与train相同的 `(source_grid, instruction)` | 1个；源PNG SHA与目标网格也完全相同 |
| train内多余重复输入 | 2条；test内0条 |
| test指令文字曾在train出现 | 637/1000条；同分布允许，不代表未见表达泛化 |
| test修改格子数 | 750个样本改1格，250个样本改2格 |

重复对：`train_001201` 与 `test_000265`，指令 `Replace the bus with a light bulb.`。
种子不同不保证合成内容不同。下一版split需去重，但单个样本在1000样本均值中最多占0.1个百分点，
不足以解释98%。不能删完这条便把历史分数称为新结果。

清单 SHA256（原始文件字节，包括换行/路径）：

```text
train.jsonl  4f7c077223c8d5afc3b364a7e8fdae9c9b8a1bcef5f999833215e0278d1df660
test.jsonl   0015459c5e6accaede9107dbf3e39b125448da6094a6545ed26b028533cf15b2
```

固定图标、同模板同分布、少量对象以及周期test选模，都应披露。高分并不自动错误，
但不可据此宣称开放世界图文理解能力；选模集也不能称为从未使用的最终盲测集。

## 3. 当前实际架构

```text
指令 → tokenizer → 完整冻结Qwen语言Transformer
     → contextual hidden缓存 [L≤64,2048]
     → 两级语言光电核心 → [L,192] → mean/max pooling → 条件向量 [192]

RGB [3,224,224] → 冻结Qwen patch+position（无原生视觉Transformer）
               → [196,1024] + 条件向量投影偏置
               → 两级视觉光电核心 → [196,192] → 14×14空间特征
               → 条件尺度/偏置调制 + 坐标
               → 3个条件卷积残差单元（dilation 1/2/4）
               → 6×6类别头、编辑头 → 真实源网格保留合成 → 输出图像
```

语言/视觉核心各有电子残差与光分支；光 Router 从4专家选Top-2，再经过global相位。
共4次特征传播+2次Router传播；17μm、10cm、激活场478×478。
融合为 RMS 同尺度后的 `(1-alpha)E + alpha O`；alpha初值0.055、可学习范围[0.01,0.95]，
不是固定50%光，alpha也不直接等于性能贡献百分比。训练未调制强度参数范围20%–30%。
后端条件卷积不是Transformer，但属于训练的电子计算，须计入参数和耗时。

Router训练还含辅助code：语言按操作给Top-2组合；视觉按输入四象限能量指定两个目标。
不是推理期直接输入task标签，但属于应披露的训练先验。当前两个Router均有一个未使用专家，
不可称四专家完全均衡。

**缓存边界**：语言缓存实际执行完整语言Transformer，绝不是仅tokenizer+embed_tokens。
若要求整个新指令推理链不执行attention/Transformer，当前版本不满足。
固定指令库可以离线缓存，但必须明确缓存命中假设；未知指令需要重新编码并计入相应开销。
若改成冻结embed_tokens+明确位置编码，应建新profile重训，不能继续引用0.9800。

源码： [任务wrapper](../../modeling.py)、
[完整编辑模型](../../../../../experiments/qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing/modeling.py)、
[文本缓存](../../../../../experiments/qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing/prompt_cache.py)、
[指标](../../../../../experiments/qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing/metrics.py)。

## 4. baseline公平性与建议修订

| 项目 | 主方法 / D2NN | 当前结构化Qwen baseline |
|---|---|---|
| 文本表示 | 最多64个上下文token，光电处理后mean/max pooling | 完整多模态网络最后有效token向量 |
| 图像表示 | merger前196 token，14×14处理后降采样 | 原生visual的merger后token，读出前先插值6×6 |
| 后端训练 | 条件卷积编辑器，继承旧电子权重 | 另一套条件卷积头，从头训练 |
| 类别loss | 全网格加权CE | 只在真实编辑格子CE |
| 编辑loss | BCE + Dice + 保留惩罚 | 动态正样本权重BCE |
| 辅助监督 | 操作分类、光学/Router正则等 | 无对应操作辅助头 |
| 最终合成 | 真实source_grid | 同样使用真实source_grid |

Qwen baseline 是冻结 `Qwen3-VL-2B-Instruct` + 有监督任务头，不是零样本原版Qwen，也不是Embedding型号。
`baseline_5090d.py` 的自由生成JSON只是另一个零样本诊断。冻结主干训练任务头是合理baseline类别，
但当前读出/损失/初始化差异较多，不能将性能差直接归因于光学架构优势。

建议按以下顺序做受控修订（**待确认，本次未改模型**）：

1. 保留v1历史结果，新建去重v2清单，三组方法用完全相同split。
2. Qwen仍冻结、不加LoRA；尽量共享电子读出头、任务loss、选模和训练预算。
   输入维度差异只加明确投影适配，避免不必要地提前压到6×6；核验视觉token二维顺序。
   文本可试简单多token pooling，但改善与否需要测量，不预先保证。
3. 明确source_grid属于输入还是oracle：结构化编辑可保留并公开；若只允许RGB+文本，
   另建不依赖真值源网格的协议。baseline当前类别头只监督编辑区，不能仅改成全格argmax却不改监督。
4. 同一checkpoint做去光复评，报告alpha、Router利用率与性能下降；不另训完全去光模型。
5. 同时报告changed-cell、scene exact与四操作分项；去重同分布为主结果，另加未见模板或组合留出的泛化测试。
   不因分数高而随意加难，不故意削弱baseline，也不以新指标遮蔽旧结果。

参数匹配也须限定：run报告主方法相位958728，D2NN相位200704；匹配的是激活专家预算200704，
不是包含Router/global的总相位或传播次数。可训练总参数分别3882608 / 3124584。

## 5. 固定权重与重新训练分开复现

任务runs根目录为 `LightGenV2/tasks/t04_semantic_interaction/runs/simulation/`。

| 方法 | run ID | run记录的源码commit |
|---|---|---|
| 主方法 | moe_router_scale_dc20_seed73 | 1328c8456f8cd140d9e3a0ccea63612618adc30d |
| D2NN | d2nn_matched_dc20_seed73 | 0af54aa24a6c3b09789d4ffb24b6e855d3b496be |
| Qwen | qwen_structured_native_visual_5090d_20260907 | 待取回原5090D run，不能以审计commit替代 |

前两项best SHA（来自comparison归档，本次未重新计算权重文件）：

```text
MoE    b3e7a005a7b29e8a52a733d79dcf48f85d2fee2f7cb1edc64db151de1d4e33e4
D2NN   3b935e8f7ae68fdab233b662605ba4e988b3780f9b117f5c56c752cb08e33baf
```

初始化报告：二者均继承旧第20轮电子/解码器权重116个tensor，光学重新初始化。
旧权重：`experiments/qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing/runs/pilot_gpu/checkpoints/last.pt`；
报告SHA：`edcb1a6ff022c94ed1693066bc2ad095f70a6937ec320577611f1c29bb83a020`。
Qwen snapshot：`89644892e4d85e24eaac8bacfd4f463576704203`。
模型文件SHA、完整依赖环境、旧权重训练过程仍需补齐发布清单。

### 已有权重复评

在对应源码checkout的仓库根目录运行。先准备并校验上面的数据、图标、同snapshot本地Qwen、文本缓存和best。
`qwen.checkpoint: auto` 需找到对应权重，否则在专用profile配置实际路径。
下面为待执行命令，使用新复评目录，避免覆盖原run的配置记录：

```bash
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile main_dc20 --phase evaluate --device cuda --checkpoint LightGenV2/tasks/t04_semantic_interaction/runs/simulation/moe_router_scale_dc20_seed73/best_checkpoint.pt --run-dir LightGenV2/tasks/t04_semantic_interaction/runs/simulation/reproduce_moe_v1
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile d2nn_dc20 --phase evaluate --device cuda --checkpoint LightGenV2/tasks/t04_semantic_interaction/runs/simulation/d2nn_matched_dc20_seed73/best_checkpoint.pt --run-dir LightGenV2/tasks/t04_semantic_interaction/runs/simulation/reproduce_d2nn_v1
```

### 给定旧电子权重的续训复现（不是从零初始化）

```bash
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile main_dc20 --phase all --device cuda --run-dir LightGenV2/tasks/t04_semantic_interaction/runs/simulation/retrain_moe_v1
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile d2nn_dc20 --phase all --device cuda --run-dir LightGenV2/tasks/t04_semantic_interaction/runs/simulation/retrain_d2nn_v1
```

外部从头重训还需旧权重的生成命令、split、选模历史，或重新建立统一随机初始化协议。
数据入口目前只检查文件存在，不核验split/cache与配置身份，不能拿别的同名缓存混用。

Qwen baseline命令见 [BASELINE_5090D_TODO.md](../../BASELINE_5090D_TODO.md)。完整1000条计时须加
`--timing-samples 1000 --warmup-forwards 50`（CLI计时默认200条）。现有训练头入口没有固定seed，
不能保证重训恰好得到0.5475；还需记录RNG、环境、模型/缓存身份。
在线特征与缓存预测也应做一致性检查，再合并性能与在线计时数据。

### 数据只读审计（Python标准库，无GPU）

在仓库根目录进入Python后执行；不修改原始数据：

```python
import json, hashlib
from pathlib import Path
p = Path('experiments/qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing/data/pilot_gpu')
rows = {s: [json.loads(x) for x in (p / (s + '.jsonl')).read_text(encoding='utf-8').splitlines() if x.strip()] for s in ('train', 'test')}
key = lambda r: json.dumps([r['source_grid'], r['instruction']], sort_keys=True)
train_keys = {key(r) for r in rows['train']}
print({s: hashlib.sha256((p / (s + '.jsonl')).read_bytes()).hexdigest() for s in rows})
print('overlap:', [r['sample_id'] for r in rows['test'] if key(r) in train_keys])
print('train duplicates:', len(rows['train']) - len(train_keys))
```

## 6. 本轮结论

优先完善数据去重、输出信息边界、共享读出/损失、旧权重来源与缓存计时合同，不先大改光路。
baseline正常优化并如实报告，包括高于主方法的情况。本轮只更新复现文档和README口径，
未删除数据、未改训练/相位代码，也没有生成新的性能成绩。
