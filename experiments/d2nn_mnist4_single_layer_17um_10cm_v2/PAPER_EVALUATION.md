# MNIST-4 实验室评估与论文图

## 1. 不改变 CCD 判决

论文评估模块只读取 `ccd_evaluate.py` 已保存的四个原始 ROI 光强和标签：

```text
prediction = argmax(raw_energy_0, raw_energy_1, raw_energy_2, raw_energy_3)
```

它不会对 CCD 做归一化、背景扣除、非线性、resize 或学习式后处理。若 CSV 中的
`prediction`/`correct` 与四个原始 ROI 能量不一致，程序会直接拒绝生成结果。

## 2. quick40 和 formal400 必须分开

| 类型 | 每类样本 | 总样本 | 用途 |
|---|---:|---:|---|
| quick40 | 10 | 40 | 光路调试、快速判断方向；不得作为正式准确率或 mask 排名 |
| formal400 | 100 | 400 | 固定随机正式测试；可进入论文 mask 对比 |
| demo_topk | 不固定 | 不固定 | 仿真筛选演示；即使成功率高也不得称为 test accuracy |

程序同时依据 `suitable_for_accuracy_reporting`、总样本数和四类 support 判定身份，
不会仅凭文件夹名判断。只有每类恰好 100 张、总计 400 张的 fixed-random 结果进入
`formal400_mask_summary.csv`、配对 mask 对比和正式汇总图。quick40 仍会生成自己的
confusion/ROI 能量诊断图，但所有表和标题都标记为 `quick40_diagnostic`。
quick40 的 stage contract 即使保守地写成
`suitable_for_accuracy_reporting=false`，也不会被误判成仿真筛选的 biased demo；严格
40 张/每类 10 张和 `quick40`/`quick_*` profile 会共同限定这一例外。

同一个 mask 可以同时传入 quick40 和 formal400；二者使用 `mask + profile` 唯一标识，
不会覆盖或混合。

## 3. 单个 mask 的标准评估

原始 CCD 评估命令不变。默认会在 `hardware_evaluation_raw/paper_evaluation/` 自动生成
论文级输出：

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.ccd_evaluate `
  --config <hardware_export>\lab_model_config.yaml `
  --manifest <formal_stage>\samples.csv `
  --ccd-dir <formal_stage>\ccd_captured `
  --output-dir <formal_stage>\hardware_evaluation_raw
```

只有开发调试才使用 `--skip-paper-report`。正式交付不要关闭论文输出。

## 4. 多个 phase mask 的公平对比

先分别完成每个 mask 的 raw CCD 评估，然后一次性传入多个评估目录：

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.paper_evaluation `
  --run mse_ce025=<mask_ce025>\formal_fixed_random_100_per_class\hardware_evaluation_raw `
  --run mse_ce100=<mask_ce100>\formal_fixed_random_100_per_class\hardware_evaluation_raw `
  --run mse_ce025=<mask_ce025>\quick_fixed_random_10_per_class\hardware_evaluation_raw `
  --output-dir paper_mask_comparison
```

正式 mask 对比要求所有 formal400 使用完全相同的 400 个 sample key 和标签。任何
缺失、替换或不同随机子集都会报错，避免把数据差异误写成 mask 改进。

对两个 formal mask 还会输出基于相同样本的准确率差、macro-F1 差、两种不一致
样本数和 exact two-sided McNemar `p` 值。该 `p` 值只用于配对分类差异，不代替
效应量和置信区间。

## 5. 机器可读输出

```text
paper_evaluation/
├── paper_metrics.json
├── run_summary.csv
├── formal400_mask_summary.csv
├── per_class_metrics.csv
├── confusion_matrix_long.csv
├── sample_energy_metrics.csv
├── paired_formal400_mask_comparison.csv
├── figure_manifest.csv
├── output_inventory.json
└── figures/
    ├── confusion_<mask>__<profile>.{pdf,svg,png}
    ├── roi_energy_<mask>__<profile>.{pdf,svg,png}
    ├── formal400_mask_comparison.{pdf,svg,png}
    ├── formal400_per_class_f1.{pdf,svg,png}
    └── formal400_energy_margin.{pdf,svg,png}
```

指标定义：

- `accuracy`：全部样本的正确比例；正式 accuracy 附 two-sided 95% Wilson CI；
- 每类 `precision/recall/F1`：四个 one-vs-rest 二分类定义；
- `macro_*`：类别 0、1、2、3 的无权平均；
- `weighted_*`：按每类 support 加权；
- `balanced_accuracy`：四类 recall 的平均；
- `normalized_target_margin`：
  `(target ROI - strongest wrong ROI) / sum(four ROI)`；
- `target_energy_fraction`：目标 ROI 占四个探测 ROI 总能量的比例。

`sample_energy_metrics.csv` 保留每张样本的四 ROI 原值和派生 margin，方便后续重画
散点图、箱线图或挑选失败例；`output_inventory.json` 给出全部输出文件 SHA-256，
便于移交后确认文件未被替换。

## 6. 图片规范

- 字体请求：Arial；字号：7 pt；
- 图片高度：5.0 cm，位于用户要求的 4–6 cm 范围；
- 输出：矢量 PDF、可编辑 SVG、600 dpi PNG；
- 色彩：蓝/橙/绿/紫的色盲友好组合；
- confusion matrix 同时标注 count 和按真实类别归一化百分比；
- aggregate mask comparison 只画 formal400，并为 accuracy 绘制 95% Wilson CI。

实验室 Windows 通常自带 Arial。服务器若没有 Arial，`paper_metrics.json` 中的
`figure_style.font_resolved_path` 会为 `null`；这时预览图仍可生成且 SVG 明确请求
Arial，但论文最终导出前应在绘图电脑安装 Arial 后重跑，避免字体静默替换。

## 7. 统计使用注意

- quick40 只能判断明显故障，40 张样本的方差很大；
- formal400 的 Wilson CI 描述二项准确率不确定性，不包含重复搭光路的系统误差；
- 若要声明物理鲁棒性，建议未来重复独立对准/采集，并以实验重复为统计单位；
- `demo_topk` 使用仿真输出筛选过样本，只允许加
  `--allow-biased-diagnostic` 生成带警告的对准诊断，禁止进入论文正式表格。
