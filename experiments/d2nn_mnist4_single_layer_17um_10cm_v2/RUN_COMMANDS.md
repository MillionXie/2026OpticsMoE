# Commands

所有命令均从仓库或解压包根目录执行。本文件只是命令记录，不是 `.sh`。

## 1. 单元测试

```bash
python -m pytest experiments/d2nn_mnist4_single_layer_17um_10cm_v2/tests -q
```

## 2. 服务器生成正式实验室 ZIP

```bash
python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_package \
  --export-dir experiments/d2nn_mnist4_single_layer_17um_10cm_v2/runs/mnist4_single_layer_17um_10cm_v2_angle_roi/hardware_export_10cm_v2_angle_roi \
  --mask-dir experiments/d2nn_mnist4_single_layer_17um_10cm_v2/runs/mnist4_single_layer_17um_10cm_v2_angle_roi/mask_candidates/phase_bmp \
  --output experiments/d2nn_mnist4_single_layer_17um_10cm_v2/runs/mnist4_single_layer_17um_10cm_v2_angle_roi/mnist4_angle_roi_four_mask_lab_bundle.zip \
  --overwrite
```

正式包包含 vendor SDK；只有开发测试时才允许加 `--omit-vendor-sdk`。生成后同时
得到 `.zip.json`，内含 ZIP SHA-256、大小和 CRC/hash 校验结果。

## 3. 实验室安装轻量环境

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r experiments\d2nn_mnist4_single_layer_17um_10cm_v2\requirements-lab.txt
```

## 4. 创建 quick40 会话

```powershell
.venv\Scripts\python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_session --profile quick40 --mask post_robust_best --output-dir sessions\post_robust_best\quick40
```

可选 mask 名：`post_robust_best`、`mid_robust_energy`、`pre_robust_best`、
`early_robust`。四张 mask 的 quick40 key 完全相同。

## 5. 验证、采集、quick40 诊断

```powershell
.venv\Scripts\python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline --phase validate --stage-dir sessions\post_robust_best\quick40

.venv\Scripts\python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline --phase acquire --stage-dir sessions\post_robust_best\quick40

.venv\Scripts\python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline --phase evaluate --stage-dir sessions\post_robust_best\quick40 --allow-quick40-diagnostic
```

quick40 不能报告正式 accuracy。

## 6. 创建并完整运行 formal400

```powershell
.venv\Scripts\python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_session --profile formal400 --mask post_robust_best --output-dir sessions\post_robust_best\formal400

.venv\Scripts\python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline --phase all --stage-dir sessions\post_robust_best\formal400
```

formal400 默认对近黑、严重饱和或四 ROI 几乎无差异的帧执行只读 QC；存在无效帧时
先写出诊断再拒绝生成可报告 accuracy。不要用 `--allow-invalid-formal` 结果写论文，
该参数只用于定位采集问题。

## 7. 汇总四张 mask 的 formal400 论文图

```powershell
.venv\Scripts\python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.paper_evaluation --run post_robust_best=sessions\post_robust_best\formal400\hardware_evaluation --run mid_robust_energy=sessions\mid_robust_energy\formal400\hardware_evaluation --run pre_robust_best=sessions\pre_robust_best\formal400\hardware_evaluation --run early_robust=sessions\early_robust\formal400\hardware_evaluation --output-dir reports\formal400_four_masks
```

详细合同、配置注意事项、图表输出和安全说明见 `README_FIRST.md`（仓库中为
`LAB_BUNDLE.md`）。
