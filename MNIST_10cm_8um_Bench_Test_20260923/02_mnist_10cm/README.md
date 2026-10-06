# MNIST-4 快速索引

- 推荐相位：`phase/B_RECOMMENDED_native8_best.bmp`；旧 post-robust 对照：`phase/A_old_post_robust_resampled_8um.bmp`。
- 两张相位均为 1920×1200、8-bit BMP，532 nm / 10 cm / 8 µm，整幅反灰度，逻辑 X+Y 双翻转已做完，**无需再处理**。
- `inputs_40_fixed/*.bmp` 是 1920×1080、8-bit 振幅图；文件名中的 `_y0` 到 `_y3` 是标签。先试 `mnist_i00013_y0.bmp`、`mnist_i00014_y1.bmp`、`mnist_i00038_y2.bmp`、`mnist_i00032_y3.bmp`。
- 相位 A/B 历史同批 40 张实拍为 24/40、37/40；新相机/ROI 必须重新测，不可沿用旧值。
- 需要定量识别时，先以当前小相机的四角标定做一次透视和镜像映射，再对逻辑 478×478 上四个固定 59×59 框积分、取最大值。原相机图上的位置不是固定旧坐标。
