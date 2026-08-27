# Warmstart5 双 SLM、焦面与 ROI 标定

> 本文后续 v2 内容是封存 formal ZIP 的历史合同。当前新光路请优先使用完整方孔
> Fresnel v3、k=1 即播套装和四点 canonical homography，见
> `experiments/hardware_sdk/GEOMETRY_AND_BRIGHTNESS.md`、
> `experiments/hardware_sdk/generators/slm_patterns/V3_CALIBRATION_COMMANDS.md` 与
> `LAB_VALIDATION_BUNDLE.md`。新流程输出已经是模型方向，禁止再做旧版横纵双翻。

本说明只使用 ZIP 中 `payload/calibration/` 下的正式资产。不要用厂商 SDK
示例图，也不要用历史 `fresnel_phase_array_...` 中 508 像素间距的 n4 图替代。

## 1. 先做双 SLM 像素级对齐

下面两张图必须成对加载：

```text
dual_slm_checker_grating/amplitude_checker_255open_c64_1024x1024.bmp
dual_slm_checker_grating/phase_grating_xy_in_255open_cells_c64_p8_1920x1200.bmp
```

- 振幅命令 255=白/开/透光，0=黑/关/遮光；播放器不得反相；
- 相位只在振幅 255 的白格中加载横纵交替光栅，在振幅 0 的黑格中为零相位；
- 相位图已经执行纵向翻转，未执行横向翻转；不得再次 flip、resize 或 recenter；
- 调整 4F、两个 SLM 的物理中心和姿态，使白格中的光栅边界与振幅格边界达到接近
  像素级重合，再进入焦面/ROI 标定。

不要把 `acquire_folder` 指向整个 `payload/calibration/` 或任何同时含多组 BMP 的
目录，它会按文件夹批量播放。推荐手动加载这一对并使用相机实时预览；若要保存一
张记录，先建立只含推荐振幅 BMP 的单文件目录：

```powershell
$ROOT = (Get-Location).Path
$PAIR = Join-Path $ROOT "payload\calibration\dual_slm_checker_grating"
$ONE = Join-Path $ROOT "calibration_preview\checker_grating"
New-Item -ItemType Directory -Force (Join-Path $ONE "input") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $ONE "ccd") | Out-Null
Copy-Item (Join-Path $PAIR "amplitude_checker_255open_c64_1024x1024.bmp") (Join-Path $ONE "input") -Force

# 先手动把 phase_grating...bmp 加载到相位 SLM；该命令不控制相位 SLM。
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\hardware_sdk\configs\tucam_meadowlark_calibration_windows.yaml `
  --input-dir (Join-Path $ONE "input") `
  --output-dir (Join-Path $ONE "ccd") `
  --phase-mask (Join-Path $PAIR "phase_grating_xy_in_255open_cells_c64_p8_1920x1200.bmp") `
  --limit 1
```

这里使用绝对 `$ROOT` 是因为显式 `--input-dir/--phase-mask` 会相对配置文件目录解析。
首次标定必须使用 `tucam_meadowlark_calibration_windows.yaml`：它不要求尚未得到的
`device_roi_xywh`，不裁剪、不缩放，以 8-bit PNG 保留相机完整原始几何。标定帧只有
少量几张，完成 n4 后即可删除。不要用这一 bootstrap 配置采正式 Qwen 数据。
`--validate-only` 只做静态运行库和文件检查，不打开设备，不能替代实时预览。

两个 BMP 的尺寸和 SHA 位于 ZIP 根目录的 `bundle_manifest.json` →
`formal_dual_slm_checker_grating_contract`。该合同只使用 ZIP 相对路径；生成电脑上
原始 `pair_manifest.json` 的绝对路径没有被复制进包。

## 2. n1 找焦面

先加载：

```text
fresnel_roi_vertex_array_532nm_17um_8um_v2/amplitude_bmp/amplitude_focus_full_white_1024x1024.bmp
```

再依次试播 `phase_bmp/phase_fresnel_n1_center_z{5,10,15}cm_...bmp`，移动 CCD
寻找最集中的中心焦点。Qwen 模型的名义传播距离是 10 cm，因此正式模型优先使用
`phase_fresnel_n1_center_z10cm_532nm_8um_1920x1200.bmp` 核对。
需要落盘 n1/n4/n9 时，沿用第 1 节的单振幅目录做法，并把 `--phase-mask` 换成对应
菲涅尔 BMP；在 n4 给出 ROI 之前始终使用 calibration bootstrap 配置。

## 3. n4 直接确定 ROI 四顶点

切换到：

```text
fresnel_roi_vertex_array_532nm_17um_8um_v2/amplitude_bmp/amplitude_roi478_white_black_1024x1024.bmp
fresnel_roi_vertex_array_532nm_17um_8um_v2/phase_bmp/phase_fresnel_n4_exact_roi_vertices_z10cm_532nm_8um_1920x1200.bmp
```

中央振幅白窗严格为 478×478、255，外围为 0，没有缩放或插值。物理换算是：

```text
478 × 17 µm = 8126 µm = 1015.75 × 8 µm
```

相位 SLM 连续像素边界坐标的中心是 `(980,590)`，逻辑 ROI 四顶点是：

```text
(472.125,82.125)      (1487.875,82.125)
(472.125,1097.875)    (1487.875,1097.875)
```

n4 的四个焦点就是完整 ROI 的四个物理顶点，横纵间距均为 1015.75 个相位像素；
不要再像旧 508 间距图那样向外推半个间距。根据相机上的四个焦点设置物理
`device_roi_xywh`。四个值必须都是 4 的倍数（例如宽高 472 或 480，不能填 478）；
把结果写入正式 `tucam_meadowlark_1024_windows.yaml` 后，先用 `--validate-only` 和
`--limit 3` 试拍复核，再开始正式任务。工作流在保存前只做一次固定 resize，输出
478×478、8-bit、mode `L` PNG，落盘后
不再 resize。

## 4. n9 检查全 ROI 几何

继续播放同距离的：

```text
phase_fresnel_n9_exact_roi_vertices_edge_midpoints_center_z10cm_532nm_8um_1920x1200.bmp
```

九个焦点覆盖四顶点、四边中点和中心，用于检查旋转、剪切、非线性畸变和有效
孔径裁切。n4/n9 本身高度对称，不能单独判断翻转身份；翻转关系以前一步非均匀
checker/grating 对齐和已知的纵向导出约定为准。

方向合同分三步且不能混用：相位 BMP 导出时已经纵向翻转，播放器不再翻；CCD 按
相机原方向落盘、不翻；模型读取 CCD 后再执行配置固定的纵向+横向翻转。

菲涅尔图使用 0.92 Nyquist 安全圆孔径。安全孔径外是平坦零相位，不代表光学
遮光，因此 n4/n9 必须配合中央 478 白窗振幅使用。标定和正式采集都不凭空做
背景扣除。

## 5. 文件校验

打包前或解压后可运行：

```powershell
python -m experiments.hardware_sdk.generators.fresnel_roi_vertex_contract `
  --calibration-dir payload\calibration\fresnel_roi_vertex_array_532nm_17um_8um_v2
```

它会验证 9 张相位 BMP、2 张振幅 BMP、四/九点位置、纵向翻转、0.92 安全孔径、
逐文件 SHA 和数值传播伪峰约束。最终 v2 源 manifest 的 SHA-256 是：

```text
88dae667691dc823c09c41e355014d75efc40457e84450073323e6b69b748a2b
```

校验通过、曝光不过饱和后，才执行 quick210 第四层快速采集或完整四层逐层采集。
