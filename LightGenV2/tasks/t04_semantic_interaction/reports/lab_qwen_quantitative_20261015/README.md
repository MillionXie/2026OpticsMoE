# OpenMoji 定量绘图：实验室 Ours 0.8655 / Qwen 0.8120

本目录是本次唯一的绘图交付入口，不是新的训练工程。不改模型、不重选 checkpoint，
不把仿真 0.8765 冒充实测。生成图片、表格和 QA 产物由本目录 `.gitignore` 忽略；
源码、测试和本说明通过 Git 同步，原始报告按 SHA 保存于任务的 runs 中。

## 结果

两份报告各有 1,000 个 TEST 场景，新增/替换/移动/删除各 250 个。
下表保持 0–1 尺度，差值单位为百分点（pp）。

| 指标 | Ours 实验室 | Qwen 数字推理 | 差值 pp |
|---|---:|---:|---:|
| Changed-cell accuracy | 0.865500 | 0.812000 | +5.3500 |
| Edit-grid IoU | 0.891083 | 0.743833 | +14.7250 |
| Object F1 | 0.953521 | 0.935763 | +1.7758 |
| Scene exact match | 0.762000 | 0.634000 | +12.8000 |
| Foreground category accuracy | 0.951167 | 0.935317 | +1.5850 |
| All-cell accuracy | 0.991806 | 0.985972 | +0.5833 |
| Preserved-cell accuracy | 0.996588 | 0.992766 | +0.3823 |
| Operation accuracy | 1.000000 | 1.000000 | 0.0000 |

| 操作 | Ours changed-cell | Qwen changed-cell | Ours edit IoU | Qwen edit IoU |
|---|---:|---:|---:|---:|
| Add / 新增 | 0.8840 | 0.6280 | 0.834000 | 0.519667 |
| Replace / 替换 | 0.7560 | 0.9120 | 0.996000 | 0.921333 |
| Move / 移动 | 0.8260 | 0.7280 | 0.738333 | 0.574333 |
| Remove / 删除 | 0.9960 | 0.9800 | 0.996000 | 0.960000 |

必须展示替换任务的劣势，不能宣称所有操作均优于 Qwen。主指标提升是 **5.35 个百分点**，
不是“提升 5.35%”；相对提升为约 6.59%。IoU 用百分数展示时总体差值为 14.725 pp，
也可按原尺度写为 +0.14725。表内显示值已四舍五入，绘图数据保留原报告精度。

## 给画图同学

- `01_overall_comparison.png/.svg`：总体五项主要指标，300 dpi PNG 与可编辑文字 SVG。
- `02_operation_comparison.png/.svg`：四类操作的 changed-cell accuracy 和 edit-grid IoU。
- `metrics.xlsx`：Comparison 表包含 5 组 × 8 项，共 40 行；Protocol 表包含来源与限制。
- `plot_data.csv`：同样 40 行，适合 Origin / Python / R。
- `plot_data.json`：包含完整原报告、来源 SHA、绘图源码 SHA、Git commit 和统计口径。

CSV 的 `ours_fraction` 和 `qwen_fraction` 是 0–1 数值；绘图显示百分数时乘 100。
`difference_pp` 已乘 100，不要再乘；`relative_change_percent` 是相对提升，不能混用。
Excel 的 D/E 列是原始数值、以百分数格式显示，F/G 为可重算公式。

Changed-cell accuracy 衡量目标编辑格子的类别预测是否正确；Edit-grid IoU 衡量编辑位置
集合的交并比，不等于类别正确率。Object F1 比较“类别、行、列”对象集合；Scene exact
match 要求整个输出网格完全正确。所有指标沿用原报告逐场景统计口径，再在组内取平均；
没有把全部格子重新拼接计算另一套指标。All-cell accuracy 易受大量未编辑格子影响，
所以主图不以它替代 Changed-cell accuracy。Operation accuracy 是操作类别识别，
两者均为 1 不代表编辑结果完美。

## 来源与版本身份

原始证据统一存放在本任务：

```text
runs/hardware/lab08655_qwen08120_comparison_20261015/
  ours_lab_full_report.json
  qwen_evaluation.json
  qwen_test_predictions.jsonl
  qwen_resolved_config.json
  source_manifest.json
```

Ours 来自本地已封存的 `.codex_tmp/openmoji_e45_full_report.json`，状态 complete，
1,000 TEST / 6,000 CCD。报告说明为冻结 epoch45、无下游重训、无逐帧强度归一化。

- Ours PT SHA256：`03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21`
- Ours 报告 SHA256：`ba51d8b425fc556f55e3c0506675c4223755673701854ad5c1d54b1f62b08dfc`

Qwen 来自训练服务器本仓库的
`runs/simulation/layered_scene_qwen_shared_s73_e100_20260920/selected_checkpoint_test_evaluation.json`，
选中 epoch30。完整冻结 Qwen3-VL-2B-Instruct 提取特征，训练适配器及共享架构的输出头；
不是在线问答生成图像。原始报告可由同目录 1,000 行 `test_predictions.jsonl` 重新汇总复核。

- Qwen PT SHA256：`0be775796d5f64df3fab6359eb354ec6666dfc29329484a620b619a6f7ec7cf8`
- Qwen 报告 SHA256：`dd22cbdc39e5e6b77606ee11b45c1dbcf49e5d4dd7791485afbb56870c1482f1`

这里使用原始封存报告；不是后来的 A100 展示图导出评估（其一个次要 Object F1 有轻微
漂移）。不得将不同评估文件中的指标混成一行。所有复制证据的完整路径/SHA 见 manifest。

比较限制：这是 TEST 参与选模的开发结果，不能称独立留出泛化；只有一轮封存实验，
没有多随机种子误差条。现有实验室汇总报告没有 TEST manifest SHA 或逐样本预测 ID，
所以本次只做报告级比较，未完成逐样本配对审计，不做显著性检验，也不编造逐图指标。
两者域不同：Ours 为实测光电推理，Qwen 为数字推理。未重新测速度/能耗或推断它们。

## 重新生成（从仓库根目录）

先取得上面的证据目录；它和生成产物都不进入 Git。Python 需要 numpy/matplotlib，
审计函数本身只用标准库，不加载 Torch、GPU 或硬件。

```powershell
python -m LightGenV2.tasks.t04_semantic_interaction.plot_lab_qwen_quantitative
python -m unittest LightGenV2.tasks.t04_semantic_interaction.tests.test_lab_qwen_quantitative
```

Excel/CSV 构建使用 Codex bundled Node 的 `@oai/artifact-tool`。本机已配置
`.codex_tmp/node_modules` 指向 bundled 依赖；其他电脑需要同等依赖，不是普通 Node 自带。
若只需重新作图，可直接使用已交付 CSV，不必安装该构建环境。

```powershell
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe" LightGenV2/tasks/t04_semantic_interaction/reports/lab_qwen_quantitative_20261015/build_workbook.mjs
```

构建时复核 80 个公式、40 行原始值，并渲染工作表截图到 `qa/` 进行视觉核对。
本次没有启动训练、连接实验室设备，或上传到师姐服务器。
