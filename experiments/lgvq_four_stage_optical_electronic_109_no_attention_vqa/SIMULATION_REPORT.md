# LGVQ 四层光电融合模型：仿真结果报告

> 本报告只描述仿真测试。没有包含或暗示任何实际光路结果。所有数值均由 `evidence/recommended` 中的固定 JSON 证据自动生成。

## 1. 正式对象与选模口径

- 正式配置：`formal_alpha50_kd300_center100.yaml`。
- 数据划分：2250 个训练视频、558 个测试视频，不设 validation。
- 训练共 100 epoch；epoch 1 以及此后每 5 epoch 在 test 上评估一次。
- 选模指标：Spatial SRCC 与 Temporal SRCC 的算术平均；最高值出现在 epoch 75，为 `0.7244`。
- checkpoint SHA256：`d357fe51b888ecace74c050096febebc09c08abc21d6b42f533ecc3cf1f4de55`。
- **test 被用于周期性选模**，因此这些数值应准确称为“best observed test”，不能当作从未参与决策的无偏最终测试估计。
- 训练期使用二维标量软目标（权重 3.0），但部署推理不加载教师模型或软目标。

## 2. 光电融合与同 checkpoint 去光对照

![同 checkpoint 的光学贡献](deployment/simulation_report/fig01_same_checkpoint_optical_contribution.png)

| 模式 | 目标 | SRCC | PLCC | RMSE |
|---|---|---:|---:|---:|
| 正常光电融合 | Spatial | 0.6094 | 0.6509 | 8.8897 |
| 同 checkpoint 旁路光学 | Spatial | 0.4676 | 0.3723 | 23.2923 |
| 正常光电融合 | Temporal | 0.8393 | 0.8548 | 7.2697 |
| 同 checkpoint 旁路光学 | Temporal | 0.7859 | 0.7269 | 15.9053 |

开光相对旁路光学：Spatial SRCC `+0.1418`、PLCC `+0.2786`、RMSE `-14.4026`；Temporal SRCC `+0.0534`、PLCC `+0.1279`、RMSE `-8.6355`。平均 SRCC 从 `0.6268` 提高到 `0.7244`，绝对增益 `+0.0976`，相对增益 `15.6%`。

这里的对照只回答：**同一个已训练光电模型在推理时失去光学分支会下降多少**。本工程没有单独训练、微调或选模一个纯电子模型，因此不能据此声称光电模型优于“重新训练到最优的纯电子基线”。

## 3. 融合系数与数值尺度

![融合 alpha 轨迹](deployment/simulation_report/fig02_fusion_alpha_trajectory.png)

最佳 checkpoint 的 Stage 1–4 alpha 为 `0.5678 / 0.5677 / 0.5665 / 0.5670`，均高于配置下限 `0.50`。每层融合前分别计算电子与光学特征 RMS，将两路归一到相同尺度后执行 `(1-alpha)E + alpha O`，再把输出 RMS 恢复到电子分支尺度；四层 `output_to_electronic_rms` 均为 `1.0000`。因此 alpha 是**尺度配平后的混合系数**，不能直接解释为探测器原始光功率占比。

| Stage | alpha | 电子 RMS（配平前） | 光学 RMS（配平前） | 输出/电子 RMS |
|---:|---:|---:|---:|---:|
| 1 | 0.5678 | 0.6904 | 0.5311 | 1.0000 |
| 2 | 0.5677 | 0.4918 | 0.3966 | 1.0000 |
| 3 | 0.5665 | 0.6914 | 0.3993 | 1.0000 |
| 4 | 0.5670 | 0.6312 | 0.2882 | 1.0000 |

## 4. 光路由诊断

![光路由诊断](deployment/simulation_report/fig03_optical_router_diagnostics.png)

| Router | 决策数 | E1–E4 平均概率 | E1–E4 Top-2 选择份额 | 捕获率 |
|---|---:|---|---|---:|
| Stage 1（并行） | 2,232 | 0.2756 / 0.1955 / 0.1721 / 0.3567 | 0.2135 / 0.2825 / 0.2256 / 0.2784 | 19.94% |
| Stage 3（串行） | 558 | 0.0826 / 0.5420 / 0.0460 / 0.3293 | 0.0000 / 0.5000 / 0.0000 / 0.5000 | 41.71% |

`selected_share` 以所有被选中的 Top-2 槽位为分母，四项和为 1。Stage 1 的四个专家均被使用；Stage 3 的选择份额为 `0 / 0.5 / 0 / 0.5`。由于每次必须选两个专家，这意味着 **558 次 Stage 3 决策全部只选择 E2 与 E4，E1 与 E3 从未进入硬 Top-2**。虽然 soft probability 在 E1/E3 上仍非零，但硬路由组合没有样本间变化，因此应明确报告为 **Stage 3 expert-selection collapse**。

捕获率表示四个路由探测窗口能量总和占对应 active detector 能量的比例；Stage 1 为 `19.94%`，Stage 3 为 `41.71%`。它衡量能量是否进入定义的路由窗口，不是分类/回归准确率。

## 5. 相位是否实际训练

![相位训练诊断](deployment/simulation_report/fig04_phase_training_diagnostics.png)

| 相位面 | 参数数 | 最终相位 std (rad) | 相对初始化 wrapped RMS (rad) | 变化 >0.05 rad 的像素 |
|---|---:|---:|---:|---:|
| Stage 1 expert | 190,096 | 0.5105 | 0.3346 | 86.68% |
| Stage 2 global | 228,484 | 0.4244 | 0.1749 | 24.09% |
| Stage 3 expert | 47,524 | 0.4401 | 0.2015 | 57.35% |
| Stage 4 global | 228,484 | 0.3928 | 0.0660 | 4.36% |
| Stage 1 router | 47,524 | 1.9821 | 0.3476 | 46.03% |
| Stage 3 router | 11,881 | 1.9999 | 0.2490 | 35.40% |

六个相位面的平均 wrapped RMS 位移为 `0.2289 rad`，说明整体并非停留在初始化。Stage 4 global 的位移最小（`0.0660 rad`，仅 `4.36%` 像素变化超过 0.05 rad），应视为当前最弱的相位学习环节；这与 Stage 3 路由塌缩是两个不同问题。

## 6. 科学结论边界

1. 可以陈述：在选中的光电 checkpoint 上，旁路所有光学计算会显著降低两项任务的 SRCC/PLCC 并提高 RMSE，模型对光学分支存在明确的推理依赖。
2. 不可以陈述：该结果证明光电模型优于一个独立充分训练的纯电子模型；这样的基线没有运行。
3. 不可以陈述：Stage 3 的四专家实现了有效动态分工；硬 Top-2 已塌缩为固定 E2+E4。
4. 当前数值是 test 参与选模后的 best-observed 结果，论文中应如实注明该口径。
5. 本报告是仿真证据汇总，不代表硬件复现结果。

## 7. 复现报告

从仓库根目录执行：

```powershell
python experiments\lgvq_four_stage_optical_electronic_109_no_attention_vqa\result_report.py
```

输出目录为 `deployment/simulation_report`。所有 PNG 使用 600 dpi；PNG/PDF 图中文字统一为 Arial 7 pt，画布高度为 5.0–5.4 cm。
