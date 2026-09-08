# LGVQ 时间质量：A100 batch=8 正式复测

## 合同

- 模型：冻结 `Qwen3-VL-2B-Instruct` + 五质量词评分头。
- 输入：每视频 4 个互不重复的 448×448 帧；每个 batch 为 8 个不同视频。
- 测试集：固定 558 个 LGVQ test 视频；69 个完整 batch，末尾 6 个视频。
- 显式 warm-up：0；第一个 test batch 纳入统计。
- 计时边界：第一个原生 Vision Transformer block 输入，到标量质量分数在 GPU 上就绪。
- 不计：模型加载、MP4 解码、裁切/缩放、processor/tokenizer、H2D、Vision block 0 之前的 patch embedding。

## 性能

| SRCC | KRCC | PLCC | RMSE | MAE |
|---:|---:|---:|---:|---:|
| 0.769409 | 0.567888 | 0.780005 | 8.716609 | 6.763984 |

与同 checkpoint 的 batch=16 正式结果 SRCC 0.768844、PLCC 0.779399 基本一致；差异来自批量浮点计算，未重新训练、未重新选权重。

## 速度和能耗

| 指标 | batch=8 结果 |
|---|---:|
| 模型边界 mean | 210.180 ms / 8 视频 |
| median | 206.873 ms / 8 视频 |
| P95 | 208.089 ms / 8 视频 |
| 第一个冷 batch | 435.288 ms / 8 视频 |
| 吞吐 | 38.063 video/s |
| 吞吐等效时间 | 26.273 ms/video |
| A100 active mean / sampled peak | 206.980 / 274.650 W |
| active energy | 43.503 J / batch；5.438 J/video |
| idle-subtracted energy | 30.756 J / batch；3.845 J/video |
| 250 W 额定上界 | 52.545 J / batch；6.568 J/video |

若统一比较 16 个视频，Qwen batch=8 需要连续执行两次：

- 时间：420.360 ms / 16 视频；
- active energy：87.006 J / 16 视频。

按当前光电系统完整输出代理 `11.207 ms、0.901 J / 16视频`，同 16 视频负载下 Qwen batch=8 相比 Ours 慢约 **37.51×**、active energy 约为 **96.58×**。若只比较一个 8 视频 batch，并假设光路仍执行一次固定 16 槽位传播，则倍率分别为 18.76× 和 48.29×；这只是负载利用率对照，不是新的 batch=8 光学性能实验。

原始机器可读结果：`evidence/t06_temporal_batch8_formal_a100.json`。

服务器完整输出：

`/DATA/DATA1/guest3/2026OpticsMoE_a100_measurements/20260908_temporal_batch8_formal/formal/batch_08/`
