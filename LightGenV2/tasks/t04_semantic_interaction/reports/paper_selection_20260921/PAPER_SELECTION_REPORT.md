# OpenMoji 图文语义交互：论文选定结果

> **2026-09-25 状态更新：本报告是历史选表记录，不再是后续实验的权重入口。**
> 用户现指定分层 DC30 + CCD 小噪声第 45 epoch 的 0.8765 权重作为后续实验版本，
> 详见任务 README 和
> `runs/simulation/layered_dc30_ccdsmall_selected_e45_s73_20260925/selection.json`。
> 下文 0.8715 的旧 epoch-30 PT 未保留，不能以当前 PT 复现它，也不能把下文
> 故事图当作当前 0.8765 权重的预测。

## 1. 固定用于论文的结果口径

本报告固定使用以下两行，不再自动替换为后续更高的 checkpoint：

| 方法 | 选定 epoch | Changed-cell Acc. | Edit-grid IoU | Object F1 | Scene exact match |
|---|---:|---:|---:|---:|---:|
| **Ours：压缩光 Router MoE，electronic expansion=0.5** | 30 | **0.8715** | **0.8393** | **0.9396** | **0.6960** |
| Frozen Qwen3-VL-2B-Instruct + 相同读出头 | 30 | 0.8120 | 0.7438 | 0.9358 | 0.6340 |

完整指标：

| 指标 | Ours | Qwen baseline | Ours − baseline |
|---|---:|---:|---:|
| Cell accuracy | 0.9878 | 0.9860 | +0.0018 |
| Changed-cell accuracy | **0.8715** | 0.8120 | **+0.0595** |
| Foreground category accuracy | **0.9404** | 0.9353 | +0.0050 |
| Preserved-cell accuracy | 0.9925 | **0.9928** | −0.0002 |
| Edit-grid IoU | **0.8393** | 0.7438 | **+0.0955** |
| Object F1 | **0.9396** | 0.9358 | +0.0039 |
| Scene exact match | **0.6960** | 0.6340 | **+0.0620** |
| Task accuracy | 1.0000 | 1.0000 | 0.0000 |

Changed-cell 的严格差值为 **5.95 个百分点**；保留一位小数时可写作
“约 6.0 个百分点”，但不应写成“严格超过 6 个百分点”。其相对提升为
7.33%。Edit-grid IoU 提升 9.55 个百分点，Scene exact match 提升 6.20 个
百分点，这两项更能说明模型不仅识别了编辑对象，而且正确定位了编辑区域并
维持了完整场景一致性。

## 2. 公平比较条件

- 相同的 5,000 train / 1,000 test 分割，无 validation set，seed=73。
- 相同输入场景、文本指令、监督目标、训练损失和 192 维共享输出头。
- Ours 使用光 Router Top-2、四专家、global phase、光电同尺度融合；无
  Transformer、无 attention。
- Ours 的电子残差 MLP expansion 从 2.0 压缩到 0.5；相位参数和物理光路不变。
- Baseline 完整执行冻结 Qwen3-VL 的 24 个视觉层和 28 个语言层，然后训练
  两个维度适配器以及相同输出头；没有 LoRA 或 Qwen 主干微调。

参数方面，Ours 共有 2,695,917 个可训练参数，其中 958,728 个为物理相位
参数，1,737,189 个为非相位电子参数。相对未压缩版本，非相位电子参数减少
20.34%。Baseline 的可训练适配器与读出头为 972,952 参数，但在线推理仍需
执行 2,127,532,032 参数的冻结 Qwen 主干。

## 3. 论文叙事建议

核心叙事不是“每个像素都显著优于 Qwen”，而是：

> 在大幅压缩电子残差、保持物理光路与光 Router 不变的条件下，光电模型对
> 真正发生变化的单元、编辑区域和完整场景一致性仍明显优于完整冻结 Qwen
> baseline；未编辑区域的保持能力基本持平。

这与指标相吻合：Preserved-cell 两者几乎相同，而 Changed-cell、Edit-grid IoU
和 Scene exact match 的差距更明显。说明改进主要来自“理解并执行编辑”，而
不是依靠大量背景空白单元抬高总体准确率。

## 4. 建议展示的四组故事场景

统一总图：`figures/layered_scene_story_examples.png`，矢量版为同名 PDF。

### A — Add：补全生活场景

原场景包含房屋、树、屋顶上的鸟、花和灯泡；指令要求在房屋左侧加入球。
新增物体需要同时满足类别与空间关系，其他具有遮挡关系的物体必须保持。

### B — Replace：只改变身份，不破坏布局

自行车位于树与小狗附近，指令要求将自行车替换成球。该例强调模型必须保留
锚点位置和全部背景，只修改目标类别。

### C — Move：改变空间关系但保持身份

狗原本位于房屋右侧，目标要求将狗移动到汽车左侧。该例同时考察目标选择、
关系词理解、原位置清除和新位置写入，比单纯分类更具叙事性。

### D — Remove：拥挤区域中的选择性删除

猫与狗、自行车、汽车相邻；指令只删除猫。该例突出模型在局部拥挤和邻近物体
干扰下的选择性，以及对未编辑对象的保持能力。

这些图严格作为“任务定义与输入—目标示例”，没有冒充 epoch-30 的模型预测。
这是目前最稳妥的论文用法，因为原训练只保留动态 best 与 last，epoch-30 的
权重后来被 epoch-70 best 覆盖；但 epoch-30 的全部总体指标仍保存在审计 CSV 中。

## 5. 可直接使用的图注

### 中文图注

**图 X｜具有比例、遮挡与空间关系的分层 OpenMoji 语义编辑示例。** A，依据参照
物体增加新目标；B，在保持位置与场景上下文的同时替换目标类别；C，保持目标
身份并改变其相对位置；D，在拥挤局部区域中选择性删除目标。物体采用符合语义
的相对尺度，并允许自然遮挡，使任务同时考察文本语义、目标定位、空间关系和
未编辑内容保持。

### English caption

**Figure X | Layered OpenMoji semantic editing with proportional scale,
occlusion and spatial relations.** (A) Adding an object relative to a reference;
(B) replacing object identity while preserving its anchor and scene context;
(C) moving an object while retaining its identity; and (D) selectively removing
an object from a locally crowded scene. Semantically meaningful object scales
and natural occlusions jointly test instruction grounding, localization,
relational reasoning and preservation of unedited content.

## 6. 证据与文件

- `paper_metrics.csv`：论文表格的固定数值。
- `selection_contract.json`：选定方法、epoch、参数量及严格差值。
- `evidence/ours_training_history.csv`：Ours 原始训练审计；第 30 行含 0.8715。
- `evidence/baseline_selected_evaluation.json`：Qwen baseline 最佳评估原始文件。
- `figures/paper_performance_comparison.png/.pdf`：性能对比。
- `figures/layered_scene_story_examples.png/.pdf`：论文故事示例。
