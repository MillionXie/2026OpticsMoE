# Architecture

## 为什么和 FSS-1000 不同

SALICON 标注的是人类注视点及其连续概率密度。模型输出不是物体边界，也不使用
阈值产生前景 mask。最后一层 logits 经过整张图的 spatial softmax，确保概率非负且
总和为 1。

## Spatial token mapping

Teacher hook 读取 Qwen Vision 最后一个原生 block 的 merger 前 packed hidden。
Student 在第一个 Vision block 入口取得 patch/position hidden 并由 Optical MoE
处理。两者都使用运行时 `image_grid_thw` 恢复二维 token 网格；token 数不匹配会
直接报错，禁止 crop、truncate 或猜测 `14×14`。

## Physical geometry

| 项目 | 数值 |
|---|---:|
| 输入 / expert | 224×224 |
| expert grid | 4×4 |
| expert pitch | 254 px |
| active footprint | 986×986 |
| FFT canvas | 1026×1026 |
| Top-K | 4 |
| expert layers | 1 |
| global phase | 986×986 |
| wavelength | 532 nm |
| pixel pitch | 8 μm |
| propagation | 10 cm |

输入相关 router 只在第一次加载时计算。OEO 对每个已选专家独立
`square detection → LayerNorm → ReLU`，然后重新施加同一组 routing weights；
未选专家继续置零。最后 global phase 后只传播和做 CCD 平方探测。

## Checkpoint policy

正式配置以官方 validation `CC` 选择最佳 checkpoint。SALICON 官方 test
ground truth 不公开，因此本地可复现报告明确命名为 `official_validation`，
不会声称是私有 test leaderboard 成绩。
