# MNIST-4 四候选 mask 实验室包

这个 ZIP 是实验室电脑的独立运行包，不需要 Git、Torch、torchvision 或 MNIST
下载。它包含四张正式的 `1920×1200`、8-bit 灰度 phase BMP、固定随机输入、
Meadowlark/TUCam SDK 与驱动、采集/评估代码，以及论文图表脚本。

## 数据口径

- `formal400`：每类固定随机 100 张，共 400 张，seed=42；只有这一组可报告
  hardware accuracy。
- `quick40`：严格取 formal400 每类 rank 0–9，共 40 张；仅用于对齐、曝光和
  mask 快速筛查，不能作为正式准确率。
- 四张 mask 使用完全相同的 key、标签与振幅输入。包内只保存一份共享的
  `478×478` 无损 PNG；实验室电脑按 `[273,273,751,751)` 逐像素、无缩放粘贴到
  `1024×1024` 黑底并生成 BMP。代码逐文件校验 PNG hash、完整画面 pixel hash、
  BMP hash、模式和尺寸。
- 服务器原先 `samples.csv` 中来自 epoch-52 `best.pt` 的 simulation prediction 和
  energy 与这四张候选 mask 不对应，因此已明确排除，不能当作四 mask 的证据。

四张候选的建议试验顺序为：

1. `post_robust_best`（epoch 12，validation 88.1212%）
2. `mid_robust_energy`（epoch 20，validation 88.0404%）
3. `pre_robust_best`（epoch 6，validation 88.1212%）
4. `early_robust`（epoch 11，validation 87.9192%）

这些数值只是候选 checkpoint 的仿真 validation，不是实测光路准确率。每张 BMP
及其 checkpoint 的 SHA-256、epoch、loss 和 phase 标准差见
`payload/phase_masks/phase_masks.json`；ZIP 内全部文件的 SHA-256 见
`bundle_manifest.json`。

## 第一次使用

从 ZIP 解压后的根目录执行：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r experiments\d2nn_mnist4_single_layer_17um_10cm_v2\requirements-lab.txt
```

然后编辑
`experiments\d2nn_mnist4_single_layer_17um_10cm_v2\lab_hardware_config.yaml`：

- 填写实测 `camera.device_roi_xywh=[left,top,width,height]`；四个数必须能被 4
  整除；
- 确认曝光时间；
- 确认振幅 SLM 的实际温度与 `lut_file` 是 30 °C 还是 70 °C。

振幅极性固定为 `255=白/亮/透光，0=黑/暗/遮光`。phase SLM 仍由实验员手动
加载；采集程序会核对 BMP 名称、`1920×1200 L` 格式和 SHA-256，并在开始前要求
人工确认屏幕上确实显示这一张。

## 推荐的实验流程

先用第一张 mask 创建 quick40 会话：

```powershell
.venv\Scripts\python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_session --profile quick40 --mask post_robust_best --output-dir sessions\post_robust_best\quick40
```

手动加载：

```text
sessions\post_robust_best\quick40\phase_to_play\post_robust_best_epoch012_1920x1200.bmp
```

先做无采集验证，再采集并生成明确标为 diagnostic 的评估：

```powershell
.venv\Scripts\python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline --phase validate --stage-dir sessions\post_robust_best\quick40

.venv\Scripts\python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline --phase acquire --stage-dir sessions\post_robust_best\quick40

.venv\Scripts\python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline --phase evaluate --stage-dir sessions\post_robust_best\quick40 --allow-quick40-diagnostic
```

其余三张 mask 只需替换 `--mask` 和输出目录；程序仍从同一 quick40 manifest
重建相同的 40 张输入。选定物理效果最好的 mask 后，再创建并采集 formal400：

```powershell
.venv\Scripts\python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_session --profile formal400 --mask post_robust_best --output-dir sessions\post_robust_best\formal400

.venv\Scripts\python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline --phase all --stage-dir sessions\post_robust_best\formal400
```

如需重建同一路径，只允许对带有本项目 ownership marker 的会话使用
`lab_session --overwrite`；代码不会清理任意未知目录。`lab_pipeline --clear-output`
只用于明确重采当前会话 CCD，请先备份已有采集。

## CCD 与评估合同

相机最终必须保存 `478×478`、8-bit、`L` 灰度图。评估器拒绝其他形状或模式，
CCD 后不做背景扣除、单帧归一化、log、激活、动态拉伸或 resize；预测只计算四个
固定 `59×59` ROI 的原始像素和再 `argmax`：

```text
0: [162,162,221,221]   1: [257,162,316,221]
2: [162,257,221,316]   3: [257,257,316,316]
```

评估器另外执行只读质量检查，但不会改变 CCD 或四区能量：单帧均值不高于 1、
饱和像素不少于 5%，或四个 ROI 的相对跨度不高于 2% 时，该帧会被标为无效。
formal400 中只要存在无效帧，程序会先保存诊断文件再默认报错，并且不会生成可报告
accuracy 或论文汇总图；应修正出光、曝光或 ROI 后重新采集。`--allow-invalid-formal`
只供排错，所得结果仍明确标为不可报告。

每次评估会输出：

- `hardware_predictions_raw.csv`：逐样本四区原始能量、预测、正确性与 QC 原因；
- `hardware_metrics_raw.json`：accuracy/diagnostic、混淆矩阵、QC 汇总和采集合同；
- `paper_evaluation/`：源数据 CSV、`paper_metrics.json`、`output_inventory.json`；
- Arial 7 pt、5 cm 高的 PDF、可编辑 SVG 与 600-dpi PNG 图。

比较四张 formal400 mask 时，四个会话必须都有相同 400 keys，然后执行：

```powershell
.venv\Scripts\python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.paper_evaluation --run post_robust_best=sessions\post_robust_best\formal400\hardware_evaluation --run mid_robust_energy=sessions\mid_robust_energy\formal400\hardware_evaluation --run pre_robust_best=sessions\pre_robust_best\formal400\hardware_evaluation --run early_robust=sessions\early_robust\formal400\hardware_evaluation --output-dir reports\formal400_four_masks
```

脚本会额外给出每类 precision/recall/F1、Wilson 95% CI、配对 McNemar 检验、
每样本 ROI 能量指标和所有结果文件的 hash，便于后续论文绘图交接。
