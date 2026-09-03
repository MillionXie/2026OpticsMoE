# 六个光学阶段：相位可视化与硬件 BMP 导出

本页只负责**训练完成后的固定 mask 导出**。它不会重新训练，也不会改动
`518` 仿真面、`478` 有效场、`17 µm` 逻辑像素、`10 cm` 传播距离、
Top-2 router 或任何网络结构。

Spatial 与 Temporal 是两个独立的单指标模型。两者有不同的 prompt、读出头、
融合参数和六张物理相位 mask，禁止混用 YAML 与 checkpoint。

## 1. Spatial 导出

在仓库根目录执行：

```powershell
$PROJECT = "experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54"
$CONFIG = "$PROJECT/configs/release/spatial.yaml"
$CHECKPOINT = "$PROJECT/runs/lgvq_spatial_qwenfront_o2_16f54/best_observed_test_checkpoint.pt"
$OUTPUT = "$PROJECT/runs/lgvq_spatial_qwenfront_o2_16f54/hardware_mask_export"

python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.export_hardware_masks `
  --config $CONFIG `
  --checkpoint $CHECKPOINT `
  --output-dir $OUTPUT `
  --phase-center-x 980 `
  --phase-center-y 590
```

Linux 命令相同，只需把 PowerShell 的反引号换成反斜杠，或写成一行。

## 2. Temporal 导出

只把上一节三个路径换成：

```powershell
$CONFIG = "$PROJECT/configs/release/temporal.yaml"
$CHECKPOINT = "$PROJECT/runs/lgvq_temporal_qwenfront_o2_16f54/best_observed_test_checkpoint.pt"
$OUTPUT = "$PROJECT/runs/lgvq_temporal_qwenfront_o2_16f54/hardware_mask_export"
```

其余命令完全相同。程序会核对 checkpoint 内的 `target_name`、精确 prompt 和
架构标签；Spatial/Temporal 放错时直接停止，不会静默导出错误 mask。

## 3. 中心与翻转

- 相位 SLM：`1920×1200`、`8 µm`，默认中心 `(980,590)`。
- 振幅 SLM：`1024×1024`、`17 µm`，默认中心 `(512,512)`。
- 沿用既有硬件合同：相位在栅格化前**上下翻转一次**，不左右翻转。
- 只有实测确认相位 SLM 中心变化时，才修改 `--phase-center-x/y`。
- 若硬件方向重新标定，CLI 支持 `--no-phase-flip-vertical` 与
  `--phase-flip-horizontal`。不要靠肉眼同时在播放器中再翻一次。

例如相位中心改成 `(975,594)`：

```powershell
python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.export_hardware_masks `
  --config $CONFIG --checkpoint $CHECKPOINT --output-dir $OUTPUT `
  --phase-center-x 975 --phase-center-y 594
```

## 4. 输出顺序与含义

硬件上的六次光学阶段顺序固定为：

1. `vision_router`：16 个 `54×54` router mask 位于 4×4 frame lane 中心；
2. `vision_expert`：每个 lane 内 2×2 专家，共 64 个 `54×54` mask；
3. `vision_global`：一张完整 `478×478` mask；
4. `language_router`：一张 `109×109` router mask 位于有效场中心；
5. `language_expert`：沿用原 2×2 位置的四张 `109×109` mask；
6. `language_global`：一张完整 `478×478` mask。

16 帧是同一光学面上的 4×4 并行布局，不是把相位 SLM 连续播放 16 次。

导出目录内容：

```text
hardware_mask_export/
├── phase_preview.png                         # 六阶段相位总览，模型方向
├── amplitude_layout_preview.png              # 六阶段输入支撑布局
├── logical_phase_478_canonical/               # 未翻转、模型坐标，8-bit PNG
├── phase_payload_478_hardware_orientation/    # 已应用硬件翻转的 478 PNG
├── phase_slm_1920x1200/                       # 真正加载到相位 SLM 的 8-bit BMP
├── amplitude_layout_1024x1024/                # 17 µm、1:1 的布局检查 BMP
├── preview/                                   # 每阶段绝对/相对相位图
├── statistics/phase_statistics.json
├── statistics/phase_tile_statistics.csv
└── hardware_mask_export_report.json           # SHA256、中心、方向和尺寸合同
```

`amplitude_layout_1024x1024/*.bmp` 中白色只表示该阶段输入场的几何位置；它们是
对准/审计模板，**不是视频样本的正式振幅输入**。正式输入仍须由推理/硬件桥根据
每个视频的中间特征逐样本生成。

## 5. 尺寸核查（不可改光路）

- 逻辑有效宽度：`478×17 µm = 8.126 mm`；
- 相位 SLM 栅格：`round(478×17/8) = 1016` 像素；
- 相位 SLM 实际宽度：`1016×8 µm = 8.128 mm`；
- 总宽度误差：`2 µm`，由非整数像素比例造成，采用关于共同光轴对称的
  physical-coordinate nearest 映射，不做相位插值，也不会单侧累积误差；
- 默认中心 `(980,590)` 时，相位有效框为 `[472,82,1488,1098)`；
- 振幅 478 有效区在 1024 面板上是 1:1，不缩放，默认框为
  `[273,273,751,751)`。

加载前请检查 `hardware_mask_export_report.json` 中 checkpoint SHA256、目标名称、
中心与翻转；相位播放器直接加载 `phase_slm_1920x1200/*.bmp`，不要再执行第二次
17/8 重建或第二次翻转。
