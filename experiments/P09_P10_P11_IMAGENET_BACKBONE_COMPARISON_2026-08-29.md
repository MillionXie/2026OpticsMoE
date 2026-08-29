# P09/P10/P11 ImageNet 光学 Backbone 受控比较

更新日期：2026-08-29

## 1. 一页结论

P09、P10、P11 均已完整训练 90 epochs，并完成最终 normal、optical-off、
random-phase 和 electronic-skip-off 推理；三个实验都生成了可迁移的
`backbone.pt`。在完全相同的参数预算、电子旁路和训练配置下，结果排序为：

```text
P11 token/channel 轴向传播 > P10 局部/全局双尺度传播 > P09 普通二维传播
```

P11 是当前应保留并进入下游迁移与 FA 四组实验的 source backbone：

- 相对 P09，P11 Top-1 提升 `1.536 pp`；
- 相对 P10，P11 Top-1 提升 `0.460 pp`；
- P11 在 90 个匹配 epoch 中有 86 个 epoch 的 validation Top-1 排名第一，
  所以优势不是只来自一次偶然的 checkpoint 峰值；
- P10 也相对 P09 提升 `1.076 pp`，说明为光学传播加入结构化归纳偏置
  总体上比八层完全相同的 50 mm 二维传播更有效；
- 三组仍各只有一个共同 seed，当前是强度较好的受控架构筛选结果，尚不是
  多 seed 统计显著性结论。

## 2. 公平比较条件

| 项目 | P09 | P10 | P11 |
|---|---|---|---|
| 光学算子 | 8×普通 50 mm 二维传播 | `[5 mm local → 50 mm global] × 4` | `[token-axis → channel-axis] × 4` |
| 光学相位参数 | 1,204,224 | 1,204,224 | 1,204,224 |
| adapter 电子参数 | 231,648 | 231,648 | 231,648 |
| residual 电子参数 | 733,472 | 733,472 | 733,472 |
| 临时任务头参数 | 650,603 | 650,603 | 650,603 |
| backbone 光学参数占比 | 55.511% | 55.511% | 55.511% |
| stem | 同一冻结 Qwen Patch/Position Stem | 同左 | 同左 |
| 训练 | ImageNet-1K、90e、global batch 192、seed 2026 | 同左 | 同左 |

三个实验的光学参数占比 `55.511%` 均指**排除临时 ImageNet 任务头的可迁移
backbone**。若把任务头计入，光学参数占比均为 `42.704%`。

P10 使用两张 RTX 4090；P09/P11 使用 4090+3090 的混合卡。因此训练吞吐
不能用于判断三种算子在相同硬件上的速度优劣，准确率和参数量比较不受影响。

## 3. 最终结果

所有 checkpoint 都按 validation Top-1 选择。

| 模型 | best epoch | Top-1 | Top-5 | val loss | 相对 P09 Top-1 |
|---|---:|---:|---:|---:|---:|
| P09 普通二维 | 90 | 49.812% | 74.224% | 2.30428 | — |
| P10 双尺度 | 90 | 50.888% | 74.956% | 2.24040 | +1.076 pp |
| P11 token/channel | 88 | **51.348%** | **75.552%** | **2.20967** | **+1.536 pp** |

直接差值：

- P10 − P09：Top-1 `+1.076 pp`，Top-5 `+0.732 pp`；
- P11 − P09：Top-1 `+1.536 pp`，Top-5 `+1.328 pp`；
- P11 − P10：Top-1 `+0.460 pp`，Top-5 `+0.596 pp`。

P11 epoch 90 为 `51.224%`，只比其 epoch-88 峰值低 `0.124 pp`；因此选择
epoch 88 并没有利用异常尖峰。P10 最佳为 epoch 90。

## 4. 匹配训练轨迹

| epoch | P09 Top-1 | P10 Top-1 | P11 Top-1 |
|---:|---:|---:|---:|
| 1 | 6.932% | **7.186%** | 6.712% |
| 5 | 24.712% | 25.276% | **25.388%** |
| 10 | 32.198% | 32.484% | **33.018%** |
| 15 | 36.430% | 36.708% | **37.392%** |
| 30 | 42.580% | 43.442% | **44.234%** |
| 60 | 48.044% | 49.142% | **49.832%** |
| 80 | 49.544% | 50.640% | **51.198%** |
| 90 | 49.812% | 50.888% | **51.224%** |

P10 在 4 个早期 epoch 排名第一，P11 在其余 86 个 epoch 排名第一，P09
没有在匹配 epoch 排名第一。90 轮 validation Top-1 的简单均值分别为：

- P09：`42.898%`；
- P10：`43.790%`；
- P11：`44.342%`。

三条曲线在 80 轮后都明显进入平台。90 epochs 对本轮架构筛选已经充分；
不应仅用原调度继续堆 epoch，下一步收益更可能来自下游迁移、结构消融或新的
训练目标。

## 5. 光学相位与融合

| 指标 | P09 | P10 | P11 |
|---|---:|---:|---:|
| mean `|Δphase|` | 2.0096 rad | 1.8118 rad | 1.6619 rad |
| phase moved >0.1 rad | 96.44% | 93.74% | 92.72% |
| best-epoch optical gate 均值 | 0.5394 | 0.5560 | 0.5398 |

三组的相位梯度均保持 finite/nonzero，相位也都发生大幅学习。因此 P09 落后
不能解释为“相位学习率太小或相位没有动”。不同算子需要的相位移动量不同，
也不能把更大的 `|Δphase|` 直接解释成更强的光学贡献。

P10 的平均光学 gate 稍高；P11 与 P09 的平均 gate 几乎一致。P11 的收益
因此更像来自 token/channel 结构化传播本身，而不是简单增加光学融合系数。
gate 是归一化后两分支的数值权重，不是能耗或实际计算比例。

## 6. 最终破坏性推理诊断

| 模型 | normal | optical-off | random-phase | electronic-skip-off |
|---|---:|---:|---:|---:|
| P09 | 49.812% | 0.262% | 0.078% | 0.032% |
| P10 | 50.888% | 0.344% | 0.088% | 0.080% |
| P11 | 51.348% | 4.556% | 0.086% | 0.098% |

可以支持的结论：

1. 三个模型随机化 phase 后都接近 1000 类随机水平，学习相位不可替代；
2. 关闭电子 residual 后三组也都崩溃，当前均是光电共适应系统；
3. P11 optical-off 仍有 `4.556%`，显著高于 P09/P10。P11 的电子路径保留了
   更多可分类信息，因此不能把 P11 的性能提升表述为“每一层都更加依赖光学”；
4. 即使如此，P11 关闭光学后仍下降 `46.792 pp`，正常性能显然不能由电子
   旁路单独维持。

按 0.1% chance 校正的 normalized optical dependence 约为：P09 `99.67%`、
P10 `99.52%`、P11 `91.31%`。该指标描述破坏性消融敏感度，不等于能耗、
MAC 占比，也不能替代重训后的纯光/纯电架构比较。

## 7. 当前能够和不能够得出的结论

### 能够支持

1. 相同预算下，结构化光学传播优于重复的普通二维传播；
2. token/channel 语义轴分解是当前三种算子中最有效的设计；
3. P11 的优势贯穿绝大多数训练过程，而非只出现在最终一次选模；
4. P10 证明局部/全局双尺度本身也有价值，但仍弱于 P11；
5. P11 应作为后续通用 source backbone，进入下游迁移和固定反馈研究。

### 还不能支持

1. 不能声称 P11 已在多随机种子上显著优于 P10/P09；
2. 不能仅凭 ImageNet 分类说明已经得到通用 backbone；还需要分类之外的
   检索、关键点、分割或其他感知任务；
3. 不能把 55.511% 参数占比解释为硬件能耗或延迟中的光学比例；
4. 不能把 destructive ablation 解释为光、电两路的独立精度贡献；
5. 不能把当前理想 axis operator 直接等同于已经搭建的真实柱面/4f 光路。

## 8. 建议决策

1. **冻结 P11 作为当前 source**，保留 epoch-88 `backbone.pt` 和完整 manifest；
2. P10 不需要继续改 recipe 重跑，可作为“物理尺度归纳偏置有效”的支持组；
3. 下一阶段优先做 P11 下游迁移，而不是继续增加 ImageNet epoch；
4. 下游任务仍保持 NoFT、BP-current、FA-pretrained、FA-random 四组；
5. 架构论文主表若严格限制四组，建议用 P09、P11、token-only、channel-only，
   P10 放补充材料；这样能直接回答 token 和 channel 两条轴是否都必要；
6. 在将 P11 的 `+0.460 pp` 相对 P10 差距作为强结论前，至少补 P10/P11 的
   配对多 seed，或先用多个下游任务验证排序是否稳定。

## 9. 证据入口

- [P09 optimization log](qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone/OPTIMIZATION_LOG.md)
- [P10 optimization log](qwen3_vl_patch_stem_8stage_dual_scale_optical_imagenet_backbone/OPTIMIZATION_LOG.md)
- [P11 optimization log](qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/OPTIMIZATION_LOG.md)
- [P11 光路与创新性评估](qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/P11_OPTICAL_IMPLEMENTATION_AND_NOVELTY_REPORT_2026-08-24.md)

服务器原始证据分别位于各实验的：

```text
runs/*_imagenet1k_pretrain_bs96_90e/result.json
runs/*_imagenet1k_pretrain_bs96_90e/metrics/history.json
runs/*_imagenet1k_pretrain_bs96_90e/checkpoints/backbone.pt
```
