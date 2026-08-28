# 实验室唯一操作文档

只按本文件从上到下执行。不要再看 `hardware_sdk` 中的历史说明，也不需要运行
`detector_homography apply`、`Get-FileHash` 或准备 `raw_roi.npy`。

所有 PowerShell 命令均在“能直接看到 `experiments` 文件夹”的仓库根目录执行：

```powershell
conda activate xml
```

## 1. 第一次实验只改两个文件

### 1.1 `experiments\lab_qwen\config\hardware.yaml`

只确认：

- `amplitude_slm.lut_file` 是实际使用的 30C 或 70C LUT；
- `camera.exposure_us` 是临时曝光时间。

ROI、合同路径和 SHA 不要手填，步骤4会自动填写。

### 1.2 `experiments\lab_qwen\config\geometry.yaml`

当前已写入本次测得的逻辑坐标：

```yaml
top_left: [1626, 281]
top_right: [358, 285]
bottom_right: [363, 1547]
bottom_left: [1631, 1545]
```

这里已经处理了 CCD 左右镜像，四个 orientation 开关必须保持 `false`。

## 2. 两个 SLM 对齐

目录：

```text
experiments\lab_qwen\calib\dual
```

严格按 `k1_pair_manifest.json` 同文件夹配对加载振幅和相位：

1. `01_checker_c64`
2. `02_large_blocks_c48_x`
3. `03_large_blocks_c48_y`

振幅约定固定为 `255=白/透光，0=黑/遮光`。

## 3. Fresnel距离和四点位置

振幅 SLM 始终加载：

```text
experiments\lab_qwen\calib\fresnel\A_WHITE.bmp
```

相位 SLM 按顺序加载：

1. `P1_POINT.bmp`：移动 CCD，寻找10 cm焦面；
2. `P4_POINT.bmp`：读取四个逻辑顶点；
3. `P9_POINT.bmp`：检查中点、中心和几何畸变。

如果点太小不方便肉眼观察，只把相位换成相同编号的 `*_CROSS.bmp`。不要改变振幅。

每次确认相位已加载后，可以用下面的命令保存一张全传感器图。以 P4 为例：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\config\hardware.yaml `
  --input-dir experiments\lab_qwen\calib\fresnel `
  --output-dir experiments\lab_qwen\work\fresnel_p4 `
  --phase-mask experiments\lab_qwen\calib\fresnel\P4_POINT.bmp `
  --file-manifest experiments\lab_qwen\calib\fresnel\amplitude_manifest.csv
```

## 4. 一条命令完成四点合同、SHA和正式配置

确认 `geometry.yaml` 坐标无误后，只运行：

```powershell
python -m experiments.lab_qwen.setup_geometry
```

它会自动完成：

- 生成 `experiments\lab_qwen\config\geometry.json`；
- 生成对应 `.sha256`；
- 将 ROI、合同路径和 SHA 写入 `hardware.yaml`；
- 开启正式的478×478四点透视校正；
- 保持网络下游不再翻转。

看到 `"status": "ready"` 就完成。本实验不需要 `detector_homography apply`。

## 5. 曝光标定

相位 SLM 加载：

```text
experiments\lab_qwen\calib\exposure\phase\phase_zero.bmp
```

然后运行：

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration exposure `
  --config experiments\lab_qwen\config\hardware.yaml
```

程序固定采集32个灰度、每个灰度3帧。查看：

```text
experiments\lab_qwen\results\exposure\brightness_response.png
```

如果有饱和，只修改 `hardware.yaml` 的 `camera.exposure_us`，重新运行步骤4使曝光配置同步，
再重新执行本步骤。

## 6. 仿真与实测光场一致性

手动加载此目录中唯一的相位 BMP：

```text
experiments\lab_qwen\agree\04_language_global\phase_to_play
```

采集：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\config\hardware.yaml `
  --stage-dir experiments\lab_qwen\agree\04_language_global --clear-output
```

评价并画图：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_evaluate `
  --session-dir experiments\lab_qwen\agree --stage language_global

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_report `
  --evaluation-dir experiments\lab_qwen\agree\agreement_evaluation `
  --output-dir experiments\lab_qwen\results\agreement
```

## 7. 最后一层快速测试

手动加载此目录中唯一的相位 BMP：

```text
experiments\lab_qwen\last\04_language_global\phase_to_play
```

依次运行：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\config\hardware.yaml `
  --stage-dir experiments\lab_qwen\last\04_language_global --clear-output

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.offline_quick_finetune `
  --session-dir experiments\lab_qwen\last --device auto --epochs 10

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.result_report `
  --root experiments\lab_qwen `
  --session-dir experiments\lab_qwen\last `
  --output-dir experiments\lab_qwen\results\last
```

## 8. 四层逐层实验

实验室当前先采第一层：手动加载
`four\01_vision_expert\phase_to_play` 中唯一相位，然后运行：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\config\hardware.yaml `
  --stage-dir experiments\lab_qwen\four\01_vision_expert --clear-output
```

将整个 `experiments\lab_qwen\four` 复制回服务器仓库的相同位置。后续严格按：

```text
vision_expert → vision_global → language_expert → language_global
```

实验室每次只做一件事：服务器生成下一层目录后，把更新后的 `four` 目录复制回来，加载
该目录唯一相位，运行与上面相同的 `acquire_folder` 命令并修改阶段文件夹名称。

以下命令只在服务器仓库根目录执行，GPU固定为4。先设置公共变量：

```bash
MODULE=experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge
PROJECT=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5
CONFIG=$PROJECT/configs/release/stage2_joint_hardware_canonical_ccd.yaml
SESSION=experiments/lab_qwen/four
```

第一层实测后，微调并导出第二层：

```bash
CUDA_VISIBLE_DEVICES=4 python -m $MODULE --config $CONFIG \
  --checkpoint experiments/lab_qwen/model/ema.pt --session-dir $SESSION \
  --stage vision_expert --phase finetune
CUDA_VISIBLE_DEVICES=4 python -m $MODULE --config $CONFIG \
  --checkpoint $SESSION/checkpoints/after_vision_expert.pt --session-dir $SESSION \
  --stage vision_global --phase export --upstream-source measured
```

第二层实测后，微调并导出第三层：

```bash
CUDA_VISIBLE_DEVICES=4 python -m $MODULE --config $CONFIG \
  --checkpoint $SESSION/checkpoints/after_vision_expert.pt --session-dir $SESSION \
  --stage vision_global --phase finetune
CUDA_VISIBLE_DEVICES=4 python -m $MODULE --config $CONFIG \
  --checkpoint $SESSION/checkpoints/after_vision_global.pt --session-dir $SESSION \
  --stage language_expert --phase export --upstream-source measured
```

第三层实测后，微调并导出第四层：

```bash
CUDA_VISIBLE_DEVICES=4 python -m $MODULE --config $CONFIG \
  --checkpoint $SESSION/checkpoints/after_vision_global.pt --session-dir $SESSION \
  --stage language_expert --phase finetune
CUDA_VISIBLE_DEVICES=4 python -m $MODULE --config $CONFIG \
  --checkpoint $SESSION/checkpoints/after_language_expert.pt --session-dir $SESSION \
  --stage language_global --phase export --upstream-source measured
```

第四层实测后完成最终微调：

```bash
CUDA_VISIBLE_DEVICES=4 python -m $MODULE --config $CONFIG \
  --checkpoint $SESSION/checkpoints/after_language_expert.pt --session-dir $SESSION \
  --stage language_global --phase finetune
```

最终 checkpoint 为：

```text
experiments/lab_qwen/four/checkpoints/after_language_global.pt
```
