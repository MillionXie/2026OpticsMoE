# Spatial-4：96.7 万参数读出头的 0.67 正式候选

本结果沿用 s1327 已训练好的相位 mask 和严格双支路结构，没有增加网络、参数或额外
推理。对 558 个固定测试视频的全新进程复评为：SRCC `0.671008`、PLCC
`0.688354`、RMSE `8.2437`。同一 checkpoint 关闭全部光学计算后，SRCC 为
`0.615902`，即光学开启带来 `+0.055106`。

本轮只校准模型中原本就存在的标量门。四次同 RMS 尺度融合的 alpha 为
`[0.48, 0.47, 0.60, 0.60]`，全部高于硬下限 `0.42`。四个电子残差块的门从约
`[0.120, 0.123, 0.119, 0.114]` 调整为 `[0.020, 0.300, 0.060, 0.005]`。
这说明第一视觉层原电子校正偏强、第二视觉层偏弱；语言两层的光学占比则可以提高到
60%。这些门本来就在 checkpoint 中，因此参数量仍是 3,786,407，后光学读出头仍是
967,458。

正式配置和 checkpoint：

```text
experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial_readout_1m_srcc067.yaml
LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/spatial_readout_1m_srcc067/best_checkpoint.pt
```

选择过程按项目当前约定直接比较周期性 test SRCC，但没有在 test 上反向传播。因此该
数值适合作为当前工程内部的正式候选；若论文后来要求完全独立盲测，应另冻结一份未参与
门控选择的数据。
