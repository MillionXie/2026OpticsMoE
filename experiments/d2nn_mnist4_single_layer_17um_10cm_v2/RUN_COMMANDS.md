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
# Robust CCD/zero-order continuation (recommended sim-to-real mask)

This continuation starts from the audited angle-ROI mask and trains only the
478x478 phase tensor.  It adds training-only independent ±2-pixel input/phase/
pre-CCD shifts, 5% block phase bypass, coherent 0--5% amplitude/phase zero-order
intensity, 0.8--1.2 detector gain and truncated biased CCD noise
N(+1%,1%) in [-1%,+3%].  Evaluation and hardware CCD remain untouched raw
linear intensity; classification remains four raw ROI sums followed by argmax.
The checkpoint is selected by three fixed-seed stochastic validation trials;
clean validation breaks ties and the sealed test split is evaluated only after
selection.  The robust candidate is forced to come from an epoch after the
perturbation sampler has been enabled; the continuation checkpoint remains an
explicit baseline rather than being relabelled as a newly trained robust mask.

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2 \
  --config experiments/d2nn_mnist4_single_layer_17um_10cm_v2/configs/release/mnist4_single_layer_17um_10cm_v2_angle_roi_ccd_robust.yaml \
  --phase all
```

The exported `formal400` amplitudes, selected phase BMP, theoretical grayscale/
binary CCDs and later measured comparison must come from this same run.  Do not
mix an older phase candidate with the new simulation references.

Export the exact clean CCD feature produced by the actual 8-bit amplitude and
phase BMPs (both `demo_topk` and the unbiased formal profile):

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.ccd_feature_export \
  --export-dir experiments/d2nn_mnist4_single_layer_17um_10cm_v2/runs/mnist4_single_layer_17um_10cm_v2_angle_roi_ccd_robust_rv3/hardware_export_10cm_v2_angle_roi_ccd_robust_rv3 \
  --device cuda --batch-size 16
```

Use `raw_linear_fp32/*.npz` for numerical sim-to-real comparison.  The
grayscale, viridis, binary, ROI overlays and contact sheet are display-only.
