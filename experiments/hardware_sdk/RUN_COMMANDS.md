# Hardware SDK 最简命令

所有命令都从仓库根目录执行，并且都是完整单行命令。不要分行复制。

## 1. 安装轻量依赖

    cd D:\code\guest\2026OpticsMoE

    python -m pip install -r experiments\hardware_sdk\requirements-light.txt

## 2. 只修改一个配置

当前唯一主配置：

    experiments\hardware_sdk\configs\tucam_windows.yaml

在 camera 部分填写：

    exposure_us: 5000.0
    require_device_roi: true
    device_roi_xywh: [left, top, width, height]

TUCam 四项 ROI 数值必须是 4 的倍数。width=956、height=956 时，正式输出直接
保存为 956×956，不做软件 resize。

播放延迟已经统一为：

    settle_delay_ms: 200

## 3. 相机单独检查

    python -m experiments.hardware_sdk.tools.camera_smoke_test --config experiments\hardware_sdk\configs\tucam_windows.yaml --output-dir experiments\hardware_sdk\artifacts\demos\tucam_smoke --frames 3

查看：

    experiments\hardware_sdk\artifacts\demos\tucam_smoke\frame_000_preview.png

并检查 NPY shape 是否等于配置 ROI 的 height×width。

## 4. 可选：生成 ROI 人工观察图和曝光图案

    python -m experiments.hardware_sdk.workflows.roi_calibration generate --config experiments\hardware_sdk\configs\tucam_windows.yaml

主要输出：

    experiments\hardware_sdk\artifacts\calibration\masks\amplitude\verify_roi_5points.bmp
    experiments\hardware_sdk\artifacts\calibration\masks\amplitude\verify_roi_5rectangles.bmp
    experiments\hardware_sdk\artifacts\calibration\masks\amplitude\verify_roi_outline.bmp
    experiments\hardware_sdk\artifacts\calibration\masks\exposure\gray_000.bmp ... gray_255.bmp

这些图只供人工观察，不执行自动 ROI 计算。

## 5. 可选：0～9 播放与相机顺序校验

    python -m experiments.hardware_sdk.demos.amplitude_camera_demo --config experiments\hardware_sdk\configs\tucam_windows.yaml

查看：

    experiments\hardware_sdk\artifacts\demos\tucam_digit_sequence\input_vs_capture_order.png

当前默认 200 ms。若右侧实拍图和左侧数字完全对应，不需要再调整。

只生成 0～9 BMP、不打开设备：

    python -m experiments.hardware_sdk.demos.amplitude_camera_demo --config experiments\hardware_sdk\configs\tucam_windows.yaml --generate-only

## 6. 推荐执行一次：0～255 曝光响应检查

    python -m experiments.hardware_sdk.workflows.roi_calibration exposure --config experiments\hardware_sdk\configs\tucam_windows.yaml

该流程直接统计原始相机强度，不扣 background。输出：

    experiments\hardware_sdk\artifacts\calibration\results\tucam\slm_response.csv
    experiments\hardware_sdk\artifacts\calibration\results\tucam\response_curve.png
    experiments\hardware_sdk\artifacts\calibration\results\tucam\response_curve_normalized.png
    experiments\hardware_sdk\artifacts\calibration\results\tucam\exposure_preview.png

根据曲线人工确定曝光时间，再修改同一个 tucam_windows.yaml。光强、曝光、增益、
ROI 或主要光路不变时，不需要每次正式实验都重跑。

## 7. 正式采集

把本轮全部 1920×1080、8-bit 灰度 BMP 放入：

    experiments\hardware_sdk\data\amplitude_to_play\

手动加载当前层相位 mask，然后运行：

    python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_windows.yaml --clear-output

输出：

    experiments\hardware_sdk\data\ccd_captured\*.npy
    experiments\hardware_sdk\artifacts\logs\tucam\capture_manifest.csv
    experiments\hardware_sdk\artifacts\logs\tucam\resolved_devices.json

这些文件就是相机硬件 ROI 的原始结果。当前流程没有 background、没有批量扣背景、
没有几何变换，也没有额外 postprocess 命令。

## 8. Grocery 多层实验

每层只需要：

1. 下载该层 amplitude_to_play BMP 到硬件电脑。
2. 清理本地 data/amplitude_to_play 并复制本层 BMP。
3. 手动加载该层相位 mask。
4. 执行第 7 节正式采集命令。
5. 将 data/ccd_captured 中的同名原始 NPY 上传到服务器对应层目录。
6. 在服务器执行 Grocery 工程对应层的电子处理或微调。
7. 下载下一层振幅，重复以上步骤。

## 9. 旧 DVP 相机

    python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\legacy\dvp_windows.json --clear-output

## 10. 测试

    python -m pytest experiments\hardware_sdk\tests -q

    python -m pytest experiments\hardware_sdk\generators\slm_patterns\tests -q
