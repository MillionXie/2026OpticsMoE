# P05 连续错位疫苗化：从事后恢复推进到预防 + 校准

更新日期：2026-08-20

## 1. 为什么 P04 之后不能只继续加消融

P04-S2 已经证明固定错位后的算子仍可学习：共同 P02 BP source 的理想 validation 为 73.40%，在 0.5/1/2 pixel 下，BP-current 适配后达到 66.04%--73.28%，FA-pretrained 达到 65.44%--71.42%，FA-random 只有 61.96%--69.20%。因此“当前 BP 是否拿得到部署后的最新关系”已经不是主要未知量。

真正的部署缺口是零微调时的脆弱性。同一个 source 在六个大偏移条件下只有 12.06%--49.68%。真实装调不能接受每次偏几个 pixel 就先坍缩再长时间重新训练，所以 P05 的核心问题改为：能否在部署前学习一个宽容的光学工作区，并在仍有残余误差时使用部署前最后算子高效校准。

## 2. 方法主线

完整方法只有两个阶段：

1. **连续错位疫苗化（prevention）**：从同一个 P02 BP seed 2026 endpoint 开始，每个训练 batch 连续采样一次全局刚性或逐层独立 `(dy, dx)`；最大偏移在前 8 epoch 从 0.25 pixel 线性增加到 2 pixel。
2. **固定部署适配（calibration）**：在训练 RNG 和模型选择均未见过的 deployment seed 9301 上固定 1/2 pixel 偏移，再运行原来的四组反馈规则。

疫苗化阶段不是第五种反馈方法。它只产生一个四组共享的新 source；部署适配主表仍严格只有 NoFT、BP-current、FA-pretrained 和 FA-random。

## 3. 疫苗化目标函数

每个 batch 同时执行理想前向和随机错位前向：

`L = 0.35 * CE(ideal) + 0.65 * CE(shifted) + 0.10 * KL(shifted || stopgrad(ideal))`

KL 温度为 2。理想视图用于防止鲁棒训练完全牺牲原始性能；错位监督直接优化部署分布；一致性项使分类语义对连续位移稳定。两个前向共享当前光学相位、受限电子残差和读出头，光学 gate 下限仍为 0.5，电子参数量仍为 416,666，没有放宽架构或电子预算。

训练位移是连续均匀采样，而不是只训练 0.5/1/2 三个离散点。每个 batch 以 0.5 概率使用八层共享的全局偏移，否则八层独立采样。这样训练目标是一个偏移分布，不是记住几组固定 mask。

## 4. 模型选择和严格留出

- 只使用 CIFAR-10 validation，当前阶段不查看 test。
- 训练 seed：2026；训练位移使用逐 epoch 独立 RNG。
- 疫苗化模型选择 seed：9201；锁定 ideal 加 global/layerwise 的 0.5/1/2 pixel，共七个环境。
- 选择指标：七个环境的平均 validation accuracy；同时报告最差环境和理想环境。
- epoch 0 原 P02 source 参与模型选择。如果全部鲁棒训练 epoch 都更差，则输出 source，而不是强行宣称改进。
- 第二阶段使用未见 deployment seed 9301，避免在模型选择方向上继续适配和汇报。

首轮进入后续正式实验的建议门槛：七环境平均准确率相对 source 至少提高 10 pp；最差环境至少提高 15 pp；理想 accuracy 下降不超过 3 pp。未达门槛时先分析训练曲线和失败几何，不使用 test 反向调参。

## 5. 第二阶段的四组问题

在疫苗化 source 上只选 1/2 pixel 两个强度、global/layerwise 两种几何：

| 组别 | 固定部署前向 | 部署后反向 |
|---|---|---|
| NoFT | 疫苗化 source + 未见固定错位 | 不更新，测预防效果 |
| BP-current | 同上 | 当前部署算子的精确 BP 上限 |
| FA-pretrained | 同上 | 局部相位梯度精确，跨层连接固定为疫苗化结束时算子 |
| FA-random | 同上 | 同形状固定随机跨层连接 |

这一步同时回答两件事：疫苗化是否提高零样本下限；在 source 已经变鲁棒后，部署前最后算子是否仍能接近 BP 并优于随机连接。

## 6. 已实现动作和复现入口

- `misalignment_vaccination.py`：连续 batch-wise 位移、课程强度、理想锚定、一致性训练、七环境模型选择、epoch 级 checkpoint 和可恢复训练。
- `deployment_adaptation.py`：增加显式 source checkpoint 模板，使四组可共享疫苗化 checkpoint；原 P04 默认路径保持不变。
- `configs/p05_misalignment_vaccination.yaml`：锁定 20-epoch 疫苗化协议。
- `configs/p05b_vaccinated_deployment_adaptation.yaml`：锁定 held-out seed 上的四组适配。
- command 45：单卡疫苗化；command 46/47：单卡适配和汇总；command 48：global/layerwise 双卡启动；command 49：等待疫苗化完成后自动启动第二阶段并汇总。

服务器启动：

```bash
PHYSICAL_GPU_INDEX=4 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/45_run_p05_misalignment_vaccination.sh

nohup env GLOBAL_GPU_INDEX=3 LAYERWISE_GPU_INDEX=4 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/49_wait_p05_and_run_adaptation.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/logs/p05_pipeline.log 2>&1 &
```

## 7. 后续不靠堆消融的推进方向

如果首轮连续错位训练有效，下一步是把它升级成多 seed/test 的鲁棒工作区结果，并加入校准样本效率曲线（例如 1%、5%、10%、100% 训练数据或固定 step 数），而不是继续横向添加网络变体。论文主图应展示“错位幅度—零样本准确率—校准后准确率”的连续曲线，以及 FA-pretrained 相对 BP 的校准效率和相对 FA-random 的优势。

之后再扩展到相位噪声、探测器噪声与横向错位的联合分布；大模型路线应复用相同概念，把光学模块作为冻结/低更新成本的视觉 token 算子，在下游任务中比较当前 BP 与预训练固定反馈，而不是直接把 CIFAR-10 结论外推到 LLM。

## 8. 服务器启动和前两轮检查（运行中，不是最终结果）

- Git 实现提交：`43fa9b61`；watcher 加固提交：`096c0418`。
- 服务器同步后：22 个单元测试通过；真实 P02 checkpoint 的双视图一 batch 前向/反向、validation 和 checkpoint smoke 通过。
- command 45 已在物理 GPU 4 启动；command 49 在 CPU 等待，疫苗化完成后将自动在 GPU 3/4 启动 P05-B。
- seed 9201 的 epoch-0 七环境基线：理想 73.40%；global 0.5/1/2 px 为 13.74%/59.90%/42.22%；layerwise 0.5/1/2 px 为 13.60%/14.36%/14.18%；平均 33.06%，最差 13.60%。

| checkpoint | ideal | global 0.5/1/2 px | layerwise 0.5/1/2 px | 七环境平均 | 最差 |
|---|---:|---:|---:|---:|---:|
| source / epoch 0 | 73.40% | 13.74 / 59.90 / 42.22 | 13.60 / 14.36 / 14.18 | 33.06% | 13.60% |
| epoch 1，训练上限 0.469 px | 71.52% | 34.50 / 59.70 / 45.18 | 26.26 / 18.70 / 16.78 | 38.95% | 16.78% |
| epoch 2，训练上限 0.688 px | 70.74% | 41.34 / 62.16 / 50.26 | 35.28 / 17.72 / 21.78 | **42.75%** | **17.72%** |
| epoch 3，训练上限 0.906 px | 70.70% | 47.88 / 62.00 / 51.60 | 40.70 / 21.34 / 25.28 | **45.64%** | **21.34%** |

到 epoch 3，七环境平均相对 source 提高 12.58 pp，已经通过平均性能 +10 pp 的首轮门槛；理想性能下降 2.70 pp，尚处预先声明的 3 pp 容忍线内。所有六个错位环境均相对 source 改善；最差环境提高 7.74 pp，距离预设 +15 pp 门槛仍有差距，当前最弱项是 layerwise 1 pixel。训练已进入 epoch 4，并继续按既定课程扩到 2 pixel，不能用以上中途结果替代 validation-best 最终汇总。
