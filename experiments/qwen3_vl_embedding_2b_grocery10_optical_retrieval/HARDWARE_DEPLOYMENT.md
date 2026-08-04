# Grocery10 四平面硬件部署

Student 同时替换 Vision 和 Language stack，每个样本依次经过四个实物平面：

```text
Vision expert 相位 + 振幅输入 → CCD-1
→ per-expert LN/ReLU → 再施加原 routing weight → 未选专家清零
→ Vision global 相位 + 重载振幅 → CCD-2
→ pooling/LN/ReLU/output adapter/residual
→ frozen Qwen merger/DeepStack/token injection → Language router
→ Language expert 相位 + 振幅输入 → CCD-3
→ per-expert LN/ReLU → 再施加原 routing weight → 未选专家清零
→ Language global 相位 + 重载振幅 → CCD-4
→ pooling/LN/ReLU/output adapter/residual/frozen RMSNorm
→ LayerNorm(224) → Linear(224,64) → L2 normalize → retrieval
```

CCD 文件已经是平方律强度，电子桥不会再次平方。

## Session 目录

```text
hardware_sessions/<session>/
├── 00_manifest/{play_order.csv,deployment.json}
├── 00_masks/{01_vision_expert,...,04_language_global}/
├── 00_input_images/{original,processor_224}/
├── 01_vision_expert/
│   ├── amplitude_to_play/*.bmp
│   ├── ccd_captured/*.npy
│   ├── simulation_reference/ccd_intensity/*.pt
│   └── electronic_output/*.json
├── 02_vision_global/
├── 03_language_expert/
├── 04_language_global/
└── 05_retrieval/{metrics.json,retrieval_results.csv,confusion_matrix.*}
```

`play_order.csv` 是四层共同使用的唯一顺序。振幅 BMP 为 1920×1080；相位 BMP 为 1920×1200，主实验导出时已按当前折叠光路上下翻转。

## CCD 注册

`capture.roi_xywh` 对完整相机帧做严格裁剪，不做任意 resize。MoE4 需要 956×956 ROI，再做精确 2×2 block mean 得到 478×478；MoE16 需要 986×986。

每层电子桥会保存相对于仿真中间结果的 MSE、MAE、relative L2 和 cosine。归一化、ROI 与曝光必须保持固定，不能用自动曝光掩盖饱和、裁剪或几何错位。

实验室采集端只使用 `hardware_sdk/amplitude_to_play` 与 `ccd_captured` 两个交接目录；它不执行本文件中的模型运算。硬件 driver、曝光/增益、SLM preload 和 ROI 标定见 [共享硬件说明](../hardware_sdk/README.md)，四层上传与处理命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
