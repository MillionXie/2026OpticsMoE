# Runtime data

- `amplitude_to_play/`: 当前一轮要按文件名顺序播放的 1920×1080、8-bit BMP。
- `ccd_captured/`: 相机 SDK 直接返回的原始 ROI 帧，不做 resize 或归一化。
- `processed/`: background 扣除以及可选面积下采样后的定量结果；不做几何变换。

这些目录中的实验数据默认不提交 Git；程序和配置通过固定目录名交接。
