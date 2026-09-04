# 正式结果：正常光电与同一最佳 checkpoint 去光

## 正式对象

- 配置：`configs/release/formal_alpha50_kd300_center100.yaml`
- checkpoint：`runs/lgvq_oeo109_alpha50_kd300_center100_v3/best_observed_test_checkpoint.pt`
- checkpoint SHA256：`d357fe51b888ecace74c050096febebc09c08abc21d6b42f533ecc3cf1f4de55`
- 最佳 epoch：75
- 数据：2250 train / 558 test / 无 validation
- 选模：每 5 epoch 测试，取 Spatial/Temporal 平均 SRCC 最高者
- 唯一训练对象：正常光电融合模型
- 去光定义：加载上述同一最佳 checkpoint，仅在推理时旁路全部光学计算
- 明确未做：没有另行训练、微调或选择纯电子模型

## 测试结果

| 模式 | 目标 | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---|---:|---:|---:|---:|---:|
| 正常光电 | Spatial | 0.6094 | 0.4412 | 0.6509 | 8.8897 | 6.8945 |
| 同一 checkpoint 去光 | Spatial | 0.4676 | 0.3208 | 0.3723 | 23.2923 | 19.8454 |
| 正常光电 | Temporal | 0.8393 | 0.6387 | 0.8548 | 7.2697 | 5.4683 |
| 同一 checkpoint 去光 | Temporal | 0.7859 | 0.5814 | 0.7269 | 15.9053 | 12.9282 |

开光相对去光的 SRCC 增益为 Spatial `+0.1418`、Temporal `+0.0534`；平均 SRCC 从 `0.6268` 提高到 `0.7244`，绝对增益 `+0.0976`，相对增益约 `15.6%`（以去光结果为分母）。在这一个已训练的光电 checkpoint 内，预测结果对光学分支存在明确依赖。

这个对照回答的是“同一个正常光电模型在推理时失去光学分支后下降多少”，不能解释为“重新训练一个最优纯电子模型后会得到多少”。后者没有执行，也不属于本次实验。

完整机器可读证据位于 [evidence/recommended](evidence/recommended)。其中 `separately_trained_electronic_baseline` 明确为 `false`。

## 与老师给出的 Optical 500×500 参考值比较

| 目标 | 本工程 SRCC | 参考 SRCC | 差值 | 本工程 PLCC | 参考 PLCC |
|---|---:|---:|---:|---:|---:|
| Spatial | 0.6094 | 0.6710 | -0.0616 | 0.6509 | 0.7106 |
| Temporal | 0.8393 | 0.8604 | -0.0211 | 0.8548 | 0.8784 |

当前版本尚未达到截图中的 Optical 500×500 参考性能，尤其 Spatial 仍有差距；不能把它描述成已经超过参考线。它严格满足当前推理架构约束，并且同一 checkpoint 去光后出现清晰的性能下降。

## 模型与相位审计

- 可训练参数：1,714,330；冻结参数：0。
- 电子计算与 4 个融合标量合计：743,926 参数；光学特征路径与光路由相关部分：970,404 参数。
- 四层融合 alpha：`0.5678 / 0.5677 / 0.5665 / 0.5670`，均高于 0.50。
- 6 组相位参数相对初始化的平均 wrapped RMS 变化：`0.2289 rad`。
- 并行专家相位变化 `0.3346 rad RMS`；并行光路由相位变化 `0.3476 rad RMS`。
- 正式预检确认推理图中没有 Qwen、Transformer、attention、mixer 或预训练/冻结主干。

训练期使用了权重 3.0 的二维标量软目标；它们只覆盖 2250 个训练样本。正式推理模型不加载教师网络或软目标文件，因此正常开光测试与同 checkpoint 去光测试使用完全相同的模型参数和最终电子读出。
