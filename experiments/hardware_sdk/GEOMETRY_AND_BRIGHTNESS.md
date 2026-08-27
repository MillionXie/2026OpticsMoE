# CCD 四点几何与振幅亮度标定

所有命令从统一仓库根目录执行：

```powershell
Set-Location E:\code\guest\2026OpticsMoE
conda activate xml
```

实验数据应放在对应工程的 `experiments/<project>/hardware_sessions/` 下。YAML
和命令不要硬编码另一台电脑的绝对路径。

## 1. 快速亮度标定

正式配置默认采集 32 个近似均匀的灰度值，包含 0 和 255；每个灰度抓 3 帧并取
中值。相机在整个扫描中只打开一次，自动曝光关闭。相机驱动用于取中值的临时 NPY
会立即删除，结果目录只保留：

- `slm_response.csv`：全部定量数据和输入图案 SHA-256；
- `exposure_summary.json`：采样合同、闭态漏光、动态范围、饱和及单调性检查；
- `brightness_response.svg/.pdf/.png/.tiff`：原始响应和仅用于曲线显示的归一化响应；
- `exposure_preview.png`：接近 0、128、255 的三张必要预览，共用同一显示范围。

定量图使用 Python/matplotlib，Arial 7 pt；若实验室电脑没有 Arial，会明确打印并
在 summary 中记录实际 fallback 字体。响应图为 183 mm × 55 mm，预览图为
183 mm × 55 mm，栅格输出为 600 dpi，SVG/PDF 保留可编辑文字。

运行：

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration generate --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml
```

`generate` 后先在相位 SLM 上固定加载
`experiments\hardware_sdk\artifacts\calibration\masks\phase\phase_zero.bmp`。只有确认这张
零相位图已经加载，且自动曝光关闭、曝光时间和增益已经固定后，才运行 32×3 采集；整个
灰度扫描期间不得更换相位图：

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration exposure --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml
```

需要自定义灰度时，用严格递增且包含 0/255 的列表替代 `gray_point_count`：

```yaml
exposure_calibration:
  exposure_times_us: [5000]
  gray_values: [0, 16, 32, 64, 96, 128, 160, 192, 224, 240, 255]
  frames_per_gray: 3
```

灰度 0 的测量值称为“振幅 SLM 闭态漏光”，不是独立背景帧。程序不采集或扣除
背景。`(I-I_closed)/(I_open-I_closed)` 只用于响应曲线显示，绝不改写网络 CCD 帧。

## 2. 生成四点 Homography 合同

Hough 变换用于找直线；将已知四顶点变为正方形应使用 projective homography。先把只读
示例复制到设备校准归档目录，再编辑复制件：

```powershell
Copy-Item experiments\hardware_sdk\configs\detector_homography_478.example.yaml `
  experiments\hardware_sdk\artifacts\calibration\detector_homography_478.lab.yaml
```

四个点必须按它们在模型光场中的身份填写为 `top_left`、`top_right`、
`bottom_right`、`bottom_left`。不要仅按它们在 CCD 图上的上下左右排序。若 4F 系统
发生旋转或镜像，逻辑标签正是 homography 恢复 canonical 方向的依据。

```powershell
python -m experiments.hardware_sdk.workflows.detector_homography fit --config experiments\hardware_sdk\artifacts\calibration\detector_homography_478.lab.yaml --output experiments\hardware_sdk\artifacts\calibration\detector_homography_478.contract.json
```

输出同时包含：

- 合同内部的 canonical payload SHA-256；
- 整个 JSON 文件的 SHA-256；
- 同名 `.sha256` sidecar。

478×478 是偶数尺寸。四个光学顶点映射到连续像素边界 `-0.5/477.5`，而不是
像素中心 `0/477`。如果有 n9 中点和中心坐标，应填写模板中的独立验证字段；只用
n4 四点无法验证精度，因为四点总能被一个 homography 精确拟合。

可先离线处理一张原始设备 ROI 图检查：

```powershell
python -m experiments.hardware_sdk.workflows.detector_homography apply --contract experiments\hardware_sdk\artifacts\calibration\detector_homography_478.contract.json --expected-file-sha256 <FIT命令输出的文件SHA> --input raw_roi.npy --output rectified_478.tif
```

## 3. 正式采集的两种互斥模式

### Legacy 模式

```yaml
camera:
  detector_geometry:
    enabled: false
```

链路仍为设备矩形 ROI → 旧 resize → 固定 bit-depth 转换。保存方向是
`legacy_camera_native`，下游是否翻转由旧实验配置决定。

### Canonical homography 模式

```yaml
camera:
  device_roi_xywh: [left, top, width, height]
  saved_frame_size_wh: [478, 478]
  saved_frame_resize_mode: auto  # 启用 homography 时不会执行这个旧 resize
  saved_frame_bit_depth: 8
  saved_frame_input_range: [0, 65535]
  detector_geometry:
    enabled: true
    contract_file: ../artifacts/calibration/detector_homography_478.contract.json
    expected_file_sha256: <64位文件SHA>
```

链路严格为：原始设备 ROI → 一次 bilinear intensity homography → 固定范围转
8-bit → 478×478 mode-L PNG。不会先 resize，也不会再 resize、逐图拉伸、gamma
或背景扣除。

输出已是 `canonical_model_xy`，采集清单会记录：

```text
orientation_canonicalized = true
downstream_loader_flip_required = false
```

因此下游的 `flip_vertical` 和 `flip_horizontal` 必须都是 false。若仍需要翻转，说明
四点逻辑身份填错，应修正四点合同；禁止在 canonical 输出后追加双翻。相位 SLM
BMP 的既定纵向导出翻转属于另一坐标合同，不受这里影响。

正式采集会校验：设备 ROI 与合同一致、合同文件 SHA 一致、payload SHA 一致、输出
严格为 478×478，并把几何合同 SHA 和每张 CCD 文件 SHA 写入
`capture_manifest.csv`。缺失或混用时直接报错，不会退回 legacy resize。
