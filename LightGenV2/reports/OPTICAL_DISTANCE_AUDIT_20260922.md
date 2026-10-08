# LightGenV2 全任务衍射距离审计（2026-09-22）

## 审计结论

当前正式 task 代码中没有 15 cm 传播。所有具有物理角谱传播的正式 profile 均为
10 cm；T05 尚无模型，T12 的 `compact_fft` 是无物理单位的 Fourier smoke backend，
两者都不能填写为 10 cm 或 15 cm。

| Task | 当前有效传播距离 | 采样间距 | 结论与代码入口 |
|---|---:|---:|---|
| T01 Object retrieval | 0.10 m | 17 µm | robust loader 将通用 MoE core 覆盖为 10 cm / 17 µm |
| T02 Keypoint detection | 0.10 m | 17 µm | `robust_vision.distance_m` 为 0.10 |
| T03 Saliency | 0.10 m | 17 µm | 继承 T02 合同；硬件配置也强制 0.10 |
| T04 Semantic interaction | 0.10 m | 17 µm | 模型构图显式写入 0.10 / 17 |
| T05 Video classification | N/A | N/A | 仅规划 README，无模型和传播代码 |
| T06 Video quality | 0.10 m | 17 µm | Spatial/Temporal 配置、settings、硬件 guard 一致 |
| T07 ABO image retrieval | 0.10 m | 17 µm | 独立 optics 实现的 transfer 明确乘 0.1 |
| T08 ABO image-text retrieval | 0.10 m | 17 µm | 复用 T01 settings/modeling |
| T10 Expert scaling | 0.10 m | 17 µm | `study.json`；现由共享物理合同校验 |
| T11 Lifelong optics | 0.10 m | 17 µm | 全部正式 JSON 配置一致 |
| T12 Text-to-image (`audited_dc20`) | 0.10 m | 17 µm | 复用 T01 正式 DC20 core |
| T12 Text-to-image (`compact_fft`) | N/A | N/A | 纯 FFT/结构 smoke，不是 Fresnel/ASM 物理传播 |
| Public T06 Temporal 0.8044 | 0.10 m | 17 µm | 已固定物理契约测试 |
| Public T07 ABO | 0.10 m | 17 µm | 独立 optics 实现明确为 10 cm |

## 容易误读的旧字段

T01、T07、T08 的分层 YAML 会从更早的 retrieval 配置继承
`optical.physics = 0.05 m / 16 µm`。当前正式构图先经过 10 cm robust loader，后者
明确执行：

```text
settings.pixel_pitch_um = settings.language_optical_pixel_pitch_um
settings.expert_interlayer_distance_m = settings.language_optical_distance_m
```

随后 Vision 和 Language 都构造同一个 `HomogeneousMoEOpticalCore`，因此实际生效
的是 `language_optical = 0.10 m / 17 µm`。5 cm 字段是未生效的历史配置残留，建议
后续发布配置中删除或移入显式 `legacy_unused`，避免再次误判。

T02/T03 的旧 base 也保留 `optical.physics`，但它们的距离同样为 0.10 m，不存在
距离冲突；逻辑输入采样使用 `robust_vision.pixel_pitch_um = 17`。

## 15 cm 的来源

仓库中 15 cm 只出现在 `experiments/hardware_sdk/generators/slm_patterns` 的
5/10/15 cm Fresnel 标定图与找焦说明中。它是实验台校准候选，不是任何正式 task
checkpoint 的传播距离。

## 本轮架构处理

- 新增无 Torch 依赖的 `LightGenV2.common.optical_contract.OpticalContract`，统一
  使用 `nm / µm / m` 字段名，避免 `distance_m`、`propagation_cm` 和裸常量混用。
- 新增唯一标准传播 profile `REFERENCE_532NM_17UM_10CM`。它不写死 task 的孔径；
  task 通过 `with_aperture(...)` 绑定 active/canvas 后，再进行设备 raster 换算与
  误差报告。
- T10 的配置检查开始使用共享物理合同。
- T12 明确暴露 `settings.optical_contract`：只有 `audited_dc20` 返回 10 cm 合同，
  `compact_fft` 和纯电子 backend 返回 `None`。
- 不修改已有模型参数名、传播计算或 checkpoint 载入路径。

下一步应按任务逐步让 settings 输出结构化 `OpticalContract`，但不能一次性替换
历史传播代码；每次迁移必须与原 transfer buffer 和逐样本输出做数值等价检查。
