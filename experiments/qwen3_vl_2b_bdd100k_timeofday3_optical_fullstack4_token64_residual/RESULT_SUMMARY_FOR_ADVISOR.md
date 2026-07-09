# Qwen3-VL-2B + BDD100K TimeOfDay-3 光学替换实验结果总结

本文只总结当前实验目录：

`experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual`

对应服务器结果目录：

`/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual/runs/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual`

## 1. 这个实验在做什么

本实验的目标是验证：在真实驾驶场景的三分类任务上，能否用可训练的光学传播模块替换 Qwen3-VL-2B 中的大部分 Transformer 计算，同时尽量保留原始多模态模型的分类能力。

任务是 BDD100K TimeOfDay-3 三分类：

| 类别 | 含义 |
|---|---|
| daytime | 白天 |
| night | 夜晚 |
| dawn_dusk | 黎明/黄昏 |

这里比较两个模型：

| 模型 | 说明 |
|---|---|
| Teacher | 完整电子版 Qwen3-VL-2B multimodal model + 分类头 |
| Student | 保留 Qwen3-VL 的 processor、图像 patch embedding、vision merger、多模态 token 注入和最终分类流程，但将完整 vision transformer stack 和完整 language transformer stack 都替换成 optical4 surrogate |

换句话说，Teacher 是原始电子大模型路线；Student 是“Qwen3-VL 框架 + 光学替换模块”的路线。

## 2. Student 的结构

Student 仍然走完整图文输入流程：

```text
image + prompt
-> Qwen3-VL processor / tokenizer / chat template
-> frozen Qwen vision patch embedding
-> VisionOpticalStackSurrogate, 4 次 optical conversion
-> frozen Qwen vision merger
-> multimodal token injection
-> LanguageOpticalStackSurrogate, 4 次 optical conversion
-> frozen final RMSNorm
-> answer-position hidden
-> classification head
-> daytime / night / dawn_dusk logits
```

关键设置：

| 项目 | 当前设置 |
|---|---:|
| backbone | Qwen/Qwen3-VL-2B-Instruct |
| vision transformer depth | 24 |
| language transformer depth | 28 |
| vision hidden size | 1024 |
| language hidden size | 2048 |
| processor pixel budget | 16384 / 16384 |
| 单张图 pre-merge visual token 数 | 约 60 |
| optical field size | 64 × 64 |
| optical padding size | 128 × 128 |
| vision optical conversions | 4 |
| language optical conversions | 4 |
| wavelength | 532 nm |
| pixel pitch | 8 μm |
| mask distance | 5 cm |
| phase init | zeros |
| amplitude mask | disabled |

当前 token64 方案没有把 token feature 插值成光场，而是：

```text
[T, hidden_size]
-> Linear(hidden_size -> 64)
-> LayerNorm(64)
-> Softplus
-> zero pad 到 [64,64]
-> 4 次光学传播/探测
-> 读取有效 token 行
-> Linear(64 -> hidden_size)
```

并且使用残差形式：

```text
Y = beta * X + alpha * Delta
```

当前训练后 scale 为：

| scale | 数值 |
|---|---:|
| beta_v | 1.0000 |
| alpha_v | 5.5894 |
| beta_l | 1.0000 |
| alpha_l | 0.2185 |

这说明视觉侧的 optical delta 被模型显著放大；语言侧 optical delta 相对较小，主要保持 identity 路径。

## 3. 数据集与训练设置

BDD100K TimeOfDay-3 数据统计如下：

| split | daytime | night | dawn_dusk | total |
|---|---:|---:|---:|---:|
| full train | 36,728 | 27,971 | 5,027 | 69,726 |
| train | 33,055 | 25,174 | 4,524 | 62,753 |
| validation | 3,673 | 2,797 | 503 | 6,973 |
| test | 5,258 | 3,929 | 778 | 9,965 |

注意：该任务类别不均衡，`dawn_dusk` 明显少于 `daytime` 和 `night`。因此除了 top-1 accuracy，也需要重点看 macro-F1 和 balanced accuracy。

训练时没有把整个训练集一次性全遍历为一个 epoch，而是每个 epoch 每类采样 1000 张：

```text
daytime:   1000 / epoch
night:     1000 / epoch
dawn_dusk: 1000 / epoch
```

这样每个 student epoch 使用 3000 张平衡样本，validation 和 test 仍然使用完整划分。

Student 训练 30 epoch。总 epoch 时间约 9.16 小时，平均每个 epoch 约 18.33 分钟。其中训练前向/反向约 3.65 小时，validation 约 5.51 小时。

## 4. 主要结果

### 4.1 Test set 总体指标

| 模型 | Top-1 Acc | Top-5 Acc | Macro-F1 | Balanced Acc |
|---|---:|---:|---:|---:|
| Teacher, full electronic Qwen3-VL-2B + head | 93.38% | 100.00% | 83.79% | 82.58% |
| Student, optical fullstack4 token64 residual | 91.07% | 100.00% | 80.24% | 80.74% |
| 差距, Teacher - Student | 2.31 pp | 0.00 pp | 3.55 pp | 1.84 pp |

结论：在完整替换 vision stack 和 language stack 的情况下，Student 的 top-1 accuracy 比 Teacher 低约 2.31 个百分点，macro-F1 低约 3.55 个百分点。考虑到替换的是 Qwen3-VL-2B 中完整的 vision transformer stack 和 language transformer stack，这个结果说明当前 optical surrogate 已经能保留相当一部分原模型判别能力。

### 4.2 Per-class 指标

| 模型 | 类别 | Precision | Recall / Class Acc | F1 |
|---|---|---:|---:|---:|
| Teacher | daytime | 93.41% | 95.42% | 94.40% |
| Teacher | night | 98.10% | 98.47% | 98.29% |
| Teacher | dawn_dusk | 64.46% | 53.86% | 58.68% |
| Student | daytime | 93.07% | 91.69% | 92.37% |
| Student | night | 97.42% | 97.84% | 97.63% |
| Student | dawn_dusk | 48.87% | 52.70% | 50.71% |

可以看到：

1. `daytime` 和 `night` 两个主要类别表现较稳定，Student 相比 Teacher 的下降较小。
2. `dawn_dusk` 是主要困难类别。Teacher 本身在该类上的 recall 也只有 53.86%，Student 为 52.70%，说明该类困难不仅来自光学替换，也来自数据量少、类别边界模糊和视觉语义本身难度。
3. Student 在 `dawn_dusk` 上 precision 从 Teacher 的 64.46% 降到 48.87%，说明 Student 更容易把部分非 dawn/dusk 样本误判成 dawn/dusk。

### 4.3 Confusion matrix

Teacher confusion matrix：

| true \ pred | daytime | night | dawn_dusk |
|---|---:|---:|---:|
| daytime | 5017 | 47 | 194 |
| night | 23 | 3869 | 37 |
| dawn_dusk | 331 | 28 | 419 |

Student confusion matrix：

| true \ pred | daytime | night | dawn_dusk |
|---|---:|---:|---:|
| daytime | 4821 | 59 | 378 |
| night | 34 | 3844 | 51 |
| dawn_dusk | 325 | 43 | 410 |

Student 的主要错误集中在：

```text
daytime -> dawn_dusk: 378
dawn_dusk -> daytime: 325
```

这与直觉一致：白天与黎明/黄昏之间的光照连续变化更难区分。

## 5. Validation 训练过程

Student 的最佳 validation 结果：

| 指标 | 数值 |
|---|---:|
| best validation top-1 | 92.50% |
| best validation macro-F1 | 80.69% |
| best validation balanced accuracy | 81.37% |
| best macro-F1 epoch | 30 |

第 30 个 epoch 的 loss：

| loss 项 | 数值 |
|---|---:|
| total loss | 0.7707 |
| vision hidden distillation loss | 0.0743 |
| answer hidden distillation loss | 0.1172 |
| KD loss | 0.2563 |
| CE loss | 0.4378 |

训练目标由四部分组成：

```text
L_total =
  0.4 * L_vision
+ 0.4 * L_answer
+ 1.0 * L_KD
+ 1.0 * L_CE
```

其中：

| loss | 作用 |
|---|---|
| L_vision | 让 student vision optical 输出接近 teacher vision stack 输出 |
| L_answer | 让 student answer hidden 接近 teacher answer hidden |
| L_KD | 让 student logits 接近 teacher logits |
| L_CE | 使用真实标签监督 student 分类 |

## 6. 可训练参数与替换范围

原始 Qwen3-VL-2B backbone 参数量约 2.13B，并且在 Student 训练中冻结。

服务器当前 model report 中记录的 optical surrogate 参数量：

| 模块 | 参数量 |
|---|---:|
| Vision optical surrogate | 148,677 |
| Language optical surrogate | 280,773 |
| 合计 | 429,450 |

Teacher/Student 的 Qwen 原始参数不参与训练。Student 主要训练：

1. vision optical surrogate；
2. language optical surrogate；
3. 最后的分类 head。

因此这个实验不是重新训练 Qwen3-VL，而是在冻结大模型主体的基础上，训练光学替换模块和小分类头。

## 7. 从结果角度怎么理解

这个实验可以对老师这样解释：

> 我们不是单独做一个小型光学分类器，而是把光学模块嵌入到 Qwen3-VL-2B 的多模态推理链路中，尝试替换大模型内部最重的 Transformer stack。Teacher 是完整电子 Qwen3-VL-2B，Student 则用两个 4-layer optical surrogate 分别替换完整 vision stack 和完整 language stack。最终在 BDD100K 白天/夜晚/黎明黄昏三分类上，Teacher test accuracy 为 93.38%，Student 为 91.07%，下降约 2.31 个百分点。说明在该任务上，当前 token64 residual optical surrogate 可以在较大幅度替换 Transformer 计算的情况下保留接近 Teacher 的分类性能。

进一步看：

1. 主要类别 `daytime` 和 `night` 保持较好；
2. 困难主要集中在 `dawn_dusk`，这也是 Teacher 自身表现较弱的类别；
3. optical surrogate 的效果不是随机可用，而是通过 teacher hidden、teacher logits 和 hard label 联合蒸馏后，能够学到与原 Qwen3-VL 中间表征相近的替代映射；
4. 当前结果说明该方向具备可行性，但还不能说明已经完成硬件可部署，因为目前仍有电子 adapter、residual branch 和分类 head。

## 8. 当前局限

需要说明的限制：

1. 当前 optical surrogate 前后仍有电子 Linear adapter，用于把 Qwen hidden 映射到 64 维光学场，再恢复到原 hidden size。
2. 当前使用 residual 分支 `Y = beta * X + alpha * Delta`，因此并非完全无电子旁路。
3. `dawn_dusk` 类别数据少且边界模糊，是当前 macro-F1 的主要限制。
4. 当前结果主要说明“光学替换模块在 Qwen3-VL 框架内可学习、可蒸馏、可保持较高分类性能”，还不是最终硬件系统的速度或能耗结论。

## 9. 一句话结论

在 BDD100K TimeOfDay-3 上，完整电子 Qwen3-VL-2B Teacher 达到 93.38% test accuracy；将完整 vision stack 和 language stack 分别替换为 4 次光学传播的 token64 residual optical surrogate 后，Student 达到 91.07% test accuracy，仅下降 2.31 个百分点，说明当前光学替换方案在该任务上已经具备较强的表征保持能力。
