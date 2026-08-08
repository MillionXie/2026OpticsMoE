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
│   ├── ccd_captured/*.png
│   ├── simulation_reference/ccd_intensity/*.pt
│   └── electronic_output/*.json
├── 02_vision_global/
├── 03_language_expert/
├── 04_language_global/
└── 05_retrieval/{metrics.json,retrieval_results.csv,confusion_matrix.*}
```

`play_order.csv` 是四层共同使用的唯一顺序。振幅 BMP 为 1920×1080；相位 BMP 为 1920×1200，主实验导出时已按当前折叠光路上下翻转。

## CCD 注册

当前硬件配置先执行 `capture.roi_xywh` 裁剪，再按配置依次执行
`capture.flip_vertical`（上下翻转）与 `capture.flip_horizontal`（左右翻转），
然后使用 nearest-neighbor 注册到模型要求的物理 ROI。MoE4 为 `956×956`，随后进行
严格的 `2×2 mean binning` 得到 `478×478`。最近邻不会生成双线性灰度，
但也不能修正旋转、透视或错误 ROI；每张图的注册信息保存在该层
`registered_ccd/*.json`。

ROI 坐标按相机原始图填写，翻转发生在 ROI 裁剪之后。仿真 CCD 已在模型坐标系，
不会被翻转。当前实验观察表明无需上下翻转，因此 MoE4 默认配置为
当前重建光路经四种方向实测后采用
`flip_vertical: true, flip_horizontal: true`；两个开关仍保留用于重新标定。

## 相机缓存参数

- `warmup_frames: 3`：相机流刚打开时先读取并丢弃 3 帧，用于稳定曝光、增益和驱动缓存。
- `discard_frames_after_display: 1`：每次 SLM 切换 BMP 并等待 40 ms 后，再丢弃 1 帧，避免保存到仍属于上一张图案的缓存帧。

两者都不是训练 batch，也不会减少需要保存的样本数。前者每次打开相机只执行一次；后者每切换一张 BMP 执行一次。

## 实测后逐层适配

`hardware_finetune.py` 将实测 CCD 作为不可微、固定的物理边界，并且只优化该
边界后面的模块。因此 Layer-1 CCD 可优化 Vision OEO、Vision global、完整
Language 光路和最终 readout；Layer-4 CCD 只能优化最终 detector normalization
和 64D readout。每层 best checkpoint、更新后的下游 mask 和下一层振幅都有独立
追溯目录。完整命令和 checkpoint 串联方式见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。

`capture.roi_xywh` 对完整相机帧做严格裁剪，不做任意 resize。MoE4 需要 956×956 ROI，再做精确 2×2 block mean 得到 478×478；MoE16 需要 986×986。

每层电子桥会保存相对于仿真中间结果的 MSE、MAE、relative L2 和 cosine。归一化、ROI 与曝光必须保持固定，不能用自动曝光掩盖饱和、裁剪或几何错位。

实验室采集端只使用 `hardware_sdk/amplitude_to_play` 与 `ccd_captured` 两个交接目录；它不执行本文件中的模型运算。硬件 driver、曝光/增益、SLM preload 和 ROI 标定见 [共享硬件说明](../hardware_sdk/README.md)，四层上传与处理命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
