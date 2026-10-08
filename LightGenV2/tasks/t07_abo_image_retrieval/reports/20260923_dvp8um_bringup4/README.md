# 2026-09-23 ABO 图搜图：DVP 8 μm 四商品六层闭环

这是关灯后完成的四个不同商品 smoke test，用于确认标定、双 SLM、DVP 相机、六层逐层推理与电子读出的完整链路；它不是 800 条 query 的正式实测 R@1。

## 最终使用配置

- CCD 全传感器四角（TL/TR/BR/BL）：`[900,142]`, `[4310,142]`, `[4297,3551]`, `[874,3544]`
- 相机实际曝光：`20002 μs`；模拟增益：`1.0`
- 每次振幅 SLM 换图后等待：`240 ms`
- 网络稀疏输入的实测 `p99` 范围：`14–29 / 255`
- 全部 192 次标定/推理采集的最大饱和像素比例：`0`
- 未对每张 CCD 图做独立光度归一化
- 模型包 SHA256：`c9926cbaaa1ef066d9657a8028dffc130a33915aa9f392551573dc79894192d0`

曝光扫描对全白灰度 255 推荐了 7 ms（`p99=200`，饱和约 `0.0035%`）。但实际网络输入远比全白图稀疏，7 ms 时信号过弱，因此六层闭环使用 20 ms；实际模型帧没有饱和。

## 四商品结果

四个 query 的商品 ID 均不同。它们在相同模拟 gallery 上的纯仿真结果为 `4/4`，六层实测特征结果为 `1/4`（diagnostic hit@1=`0.25`）。这只是四样本诊断，不能作为正式数据集准确率。

阶段与仿真 CCD 的平均 PCC：

| 阶段 | 平均 PCC | 最佳相位变换 | CCD 方向 |
|---|---:|---|---|
| vision router | 0.3368 | none_normal | flip_v |
| vision expert | 0.1187 | h_inverse | rot270 |
| vision global | 0.1201 | none_normal | rot270 |
| language router | 0.3189 | h_inverse | flip_v |
| language expert | 0.0813 | hv_inverse | rot270 |
| language global | 0.0865 | hv_inverse | rot270 |

最终四个描述符与各自纯仿真描述符的余弦相似度为：`0.4654, 0.3773, 0.5240, 0.7977`。

结论：曝光、ROI、设备控制和六层数据流已经跑通；当前主要瓶颈不是饱和或环境光，而是 expert/global 阶段的仿真实测域差异。router 两层尚能保持约 0.32–0.34 PCC，但 expert/global 明显掉到约 0.08–0.12。当前结果不适合直接扩展为 800 query 正式测试，应先做 8 μm 原生重训、实测适配或针对 expert/global 的标定修正。

## 唯一建议查看的结果

- `final/four_image_report.json`：权威完整报告，包括逐样本预测、每阶段 PCC、方向选择和全部采集统计。
- `final/previews/`：六层实测与仿真的可视化对照。
- `final/ccd/`：透视矫正后的 478×478 CCD PNG。
- `final/phase/`：实际筛选和使用的相位 BMP。
- `final/measured_features.pt`：四样本六层实测张量与最终描述符。
- `final/simulation_reference.npz`：同四样本的纯仿真参考。
- `final/four_image_flow.log`：完整运行日志。
- `08_four_distinct_stage_calibrated_review.zip`：上述最终运行的便携归档。

远端原始结果目录：

`E:\code\guest\2026OpticsMoE\ABO_I2I_Lab_DVP_8um\runs\bringup4_20260923\08_four_distinct_stage_calibrated`
