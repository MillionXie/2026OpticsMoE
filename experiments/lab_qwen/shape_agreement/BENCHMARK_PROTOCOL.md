# 形状输入 × 形状相位 mask：仿真—实测一致性基准

本会话固定使用 532 nm、17 µm 逻辑采样、518×518 传播画布、中心 478×478
有效光场、10 cm 角谱传播和 0.65° k 空间截止。6 个非对称振幅形状分别与
6 个几何相位 mask 组合，共采集 36 帧。非对称图形用于暴露左右/上下翻转错误，
而不是在评估时自动纠正方向。

## 正式主结果

- 主参考：`transport_quantized`，即相位量化为 8-bit 后的仿真结果。
- 主域：`linear`，即 CCD 原始非负强度域。
- 主方向：固定四点 homography 后的 `canonical_model_xy`；不逐帧配准、不搜索翻转。
- 不做背景扣除，不做逐帧 min-max 归一化。
- 全部 36 帧共用一个全局能量增益，仅用于报告绝对能量比例；PCC、SSIM、
  shape-NRMSE 和余弦相似度仍反映空间分布一致性。

## 指标

- `pcc_full`：整幅图 Pearson 相关系数。
- `pcc_signal`：理论光能 99% 信号区域内的 PCC。
- `ssim`：按单帧均值作无量纲化并固定截断后的结构相似度。
- `shape_nrmse`：双方各自按总能量归一化后的 NRMSE，越低越好。
- `cosine_similarity`：非负强度向量的余弦相似度。
- `centroid_distance_px`：实测与仿真光强质心距离，单位为 478 网格像素。
- `outside_energy_fraction`：实测能量落在理论 99% 信号区外的比例。
- `energy_ratio_raw/calibrated`：原始/全局增益校准后的能量比例。
- `saturation_fraction`：CCD 饱和像素比例。

`best_orientation_diagnostic` 只用于排查配置错误。它不会改变任何主指标；若大量
样本的最佳诊断方向不是 `identity`，应回去检查四个逻辑角点标签，不应从多个
翻转结果中挑最高值作为正式结果。
