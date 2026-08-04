# 自动化 SLM/CCD 实验流程

## 实际数据流

该 Student 同时包含 Vision 与 Language 两套光学网络，所以每个样本需要四次曝光：

```text
Vision expert mask + sample amplitude
→ CCD-1
→ per-expert LN → ReLU → 原 routing weight → 未选专家置零

Vision global mask + reload amplitude
→ CCD-2
→ pooling/LN/ReLU → output adapter/residual
→ frozen Qwen merger/DeepStack/token injection → Language router

Language expert mask + sample amplitude
→ CCD-3
→ per-expert LN → ReLU → 原 routing weight → 未选专家置零

Language global mask + reload amplitude
→ CCD-4
→ pooling/LN/ReLU → output adapter/residual → frozen RMSNorm
→ LayerNorm(224) → Linear(224,64) → L2 normalize → retrieval
```

平方律只在 CCD 发生一次。电子处理读取的文件已经是强度，不会再次平方。

## 会话目录

```text
hardware_sessions/<session>/
├── 00_manifest/play_order.csv
├── 00_masks/{01_vision_expert,...,04_language_global}/
├── 00_input_images/{original,processor_224}/
├── 01_vision_expert/
│   ├── amplitude_to_play/*.bmp
│   ├── ccd_captured/*.npy
│   ├── simulation_reference/ccd_intensity/*.pt
│   ├── comparison/{ccd_vs_theory.csv,summary.json,*.png}
│   └── electronic_output/*.json
├── 02_vision_global/
├── 03_language_expert/
├── 04_language_global/
└── 05_retrieval/
    ├── metrics.json
    ├── retrieval_results.csv
    ├── confusion_matrix.csv
    ├── confusion_matrix.png
    └── automation_summary.json
```

`play_order.csv` 是四个平面共同遵守的顺序。相位 mask 在导出前已按当前折叠光路进行上下翻转；振幅 BMP 为 1920×1080，相位 BMP 为 1920×1200。

## CCD 注册与对比

配置的 `capture.roi_xywh` 对全传感器图做严格裁剪；不做任意 resize。MoE4 要求物理 ROI 956×956，再做精确 2×2 block mean 得到 478×478；MoE16 要求 986×986。

每层对 measured/theoretical CCD 输出：

- mean-normalized MSE（默认，去除整体曝光倍率）；
- mean-normalized MAE；
- Pearson correlation coefficient (PCC)；
- measured/theoretical mean 与 max；
- 若干并排对照图。

MSE 对曝光倍率敏感，因此默认先分别除以均值；PCC 本身对整体线性增益不敏感。饱和、裁剪、几何错位不会被这一步消除。

## SDK 隔离

`hardware_devices.py` 定义稳定接口：

```text
SLMDriver.open / display_file / close
CameraDriver.open / capture / close
```

现有 HOLOEYE 和 DVP 只是两个插件。DVP 上传包为旧 Python ABI，因此由 `dvp_capture_worker.py` 在 Python 3.5 中常驻打开相机；Qwen 与电子处理继续运行在 Python 3.11。后续换厂商只新增 driver 并修改 YAML。

完整命令见 `RUN_COMMANDS.md`。
