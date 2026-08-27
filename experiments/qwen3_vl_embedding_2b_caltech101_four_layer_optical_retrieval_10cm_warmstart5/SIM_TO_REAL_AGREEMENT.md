# 仿真与实际 CCD 一致性协议

本协议回答的是“同一个振幅 BMP 和同一个相位 BMP 经过真实光路后，CCD
强度是否与传播仿真一致”。它和 retrieval accuracy 是两条互补证据链，不能相互
替代。

## 1. 三种条件

- `calibration`：人为设计的非对称探针，包括暗场、均匀场、宽十字、四象限灰度、
  横/纵光栅、棋盘和固定随机块。它们只用于亮度尺度、方向和硬件稳定性诊断；
  不计入 held-out task 汇总。
- `evaluation`：test split 中每类按 sample key SHA-256 固定排序选两张，共 20 个
  独立模型输入。选择过程完全不读取 CCD 或 PCC。
- `repeatability`：固定两个 evaluation 输入，各播放三次。正式汇总先在同一个
  `canonical_key` 内平均，重复帧不会被当作独立样本扩大 n。

默认 quick 实验共 8 个 calibration 探针、20 个独立 task 探针和 4 个额外重复帧，
合计 32 次曝光。若时间非常紧，可在 YAML 把 `task_test_per_class` 改为 1。

## 2. 两套仿真参考

- `ideal_model_fp32`：训练模型在 eval 模式下产生的原始 float32 CCD 强度。
- `transport_quantized`：从实际导出的 uint8 compact amplitude、uint8 compact
  phase 解码后重新传播，包含 8-bit 量化与振幅 99.5 percentile clipping 的影响。

其中 phase 必须跟本工程 hardware bridge 的实际编码互逆：编码为
`floor(mod(phase,2π) × 256/(2π))`，解码为 `gray × 2π/256`。因此 255 表示
`255/256` 圈而不是完整 `2π`；这与另一套未被本 bridge 调用的 `/255` round 编码不可
混用。该公式会固化到每个 stage 的 `agreement_contract.json`。

因此 `ideal_model_fp32` 与 `transport_quantized` 的差别是数字导出上限；真实 CCD
应主要与 `transport_quantized` 比较。两套参考均以压缩 NPZ 保存并在 manifest 中
绑定 SHA-256。

## 3. 严格配对

评估使用 `(stage, capture_key, canonical_key, repeat_index)` 配对，并逐个验证：

- checkpoint 与 resolved config SHA；
- compact amplitude、重建 amplitude BMP SHA；
- compact phase、重建 phase BMP SHA；
- acquisition manifest 中的 phase SHA；
- CCD 和两套理论参考 SHA。

任一缺失、重复或哈希不匹配都会中止，不会通过文件名模糊匹配继续统计。

## 4. 坐标与方向

主指标只接受已经通过同一个、预先声明的四顶点透视矩阵变换到 model coordinates
的 478×478 CCD。四个点对应顺序必须固定为 model `TL, TR, BR, BL`。

先用 `experiments.hardware_sdk.workflows.detector_homography` 生成并校验 contract，
再在 `tucam_meadowlark_1024_windows.yaml` 中设置
`camera.detector_geometry.enabled: true`、`contract_file` 及其精确文件 SHA-256。
正式 capture manifest 的每一行必须声明：

```text
orientation_canonicalized=true
saved_frame_orientation=canonical_model_xy
downstream_loader_flip_required=false
background_subtraction=false
per_frame_minmax_normalization=false
```

评估器还要求所有帧使用同一个 detector geometry file/payload SHA。旧的 axis resize、
下游 flip 或未绑定 geometry contract 的 CCD 会被拒绝，不能冒充正式主指标。

评估器会额外比较 `identity / flip_vertical / flip_horizontal / rotate_180` 四个候选，
但它们只使用 calibration 探针产生方向诊断。最佳候选不会逐帧或自动应用到主指标；
如果最佳候选不是 identity，应修正固定的 canonical transform 后重新注册。

禁止以下会人为提高 PCC 的操作：

- 每张图单独平移或选最优翻转；
- 每张图 min-max、直方图匹配或 gamma 拟合；
- 未采集真实暗场却做背景扣除；
- 只选择 PCC 较高的 test 图。

## 5. 指标

线性域直接使用非负 CCD 强度，报告：

- full-frame PCC；
- simulation 99% 累积能量区域内 PCC；
- 11×11 Gaussian-window SSIM；
- shape-NRMSE；
- calibration probes 统一拟合后的能量比；
- saturation fraction；
- 质心 x/y 和距离误差；
- simulation signal support 外的实测能量比例。

network-input 域对实际和仿真执行完全相同的：

```text
clamp nonnegative
→ divide by that frame's mean
→ relative clip at 12
→ log1p
→ exact AdaptiveAvgPool 478×478 → 224×224
```

再计算同一组形态指标。PCC 对增益和偏置不敏感，因此任何结论必须同时检查能量比、
饱和率、shape-NRMSE 和质心误差。

## 6. Quick 与四层路径

- quick：只导出 `language_global`，`--upstream-source simulation`。这是最快的
  单层物理替换检查。
- isolated four-stage：一次导出四层但各层 upstream 都仿真，用于定位哪一层光路
  偏差最大。
- sequential four-stage：前一层采集并注册后，下一层用
  `--upstream-source measured` 导出。当前层只能和该 measured-upstream 条件下同时
  生成的 conditional theory 比较，不能与原始全仿真 CCD 混用。

所有命令从仓库根目录执行，例如：

```text
E:\code\guest\2026OpticsMoE
```

不使用旧 ZIP 解压目录或机器专属绝对路径。session 统一放在本项目的
`hardware_sessions/` 下。
