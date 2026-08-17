# 四层光路数据协议

## 唯一职责

实验室电脑负责：播放、采集、固定 ROI、area resize、固定强度映射、保存
`478×478 uint8 PNG`、重建完整 SLM BMP。它不做背景扣除、逐图拉伸、翻转、模型
归一化或训练。

服务器负责：生成紧凑 SLM payload、校验 basename/尺寸/位深、按配置翻转 CCD、
模型前向、逐层下游微调。它不再处理或长期保存 1176/956 原始图，也不生成逐样本
float32 CCD PT 缓存。

## 数据流

```text
服务器 compact_amplitude/phase 478 PNG
    ↓ 仅传紧凑文件
实验室 nearest 2× 重复 + 精确居中补零
    ↓
完整 amplitude 1920×1080 BMP / phase 1920×1200 BMP
    ↓ 光路
相机 ROI（仅内存 uint16）
    ↓ area resize + 固定 0..65535→0..255
478×478 uint8 PNG（不翻转）
    ↓ 上传
服务器按 YAML 翻转并直接送入本层 CCD readout
```

紧凑 SLM 重建是确定性的：一个逻辑像素严格重复为 `2×2` 物理像素，有效区为
`956×956`，然后零填充到完整 SLM。`reconstruction_manifest.csv` 保存源/目标 SHA256
及有效区边界，basename 不改变，因此 CCD、amplitude 和主 manifest 可一一对应。

## Session 目录

```text
session/
├── manifest.csv
├── 01_vision_expert/
│   ├── compact_amplitude/       # 服务器生成，传到实验室
│   ├── compact_amplitude_manifest.csv  # 量化 scale、SHA256、basename
│   ├── compact_phase/           # 服务器生成，传到实验室
│   ├── amplitude_to_play/       # 仅实验室重建
│   ├── phase_to_play/           # 仅实验室重建
│   ├── ccd_captured/            # 实验室上传的 478 uint8
│   ├── finetune_metrics.json     # 本层实测 query-gallery 测试性能
│   └── transport_spec.json
├── 02_vision_global/
├── 03_language_expert/
├── 04_language_global/
└── checkpoints/
    ├── after_vision_expert.pt
    ├── after_vision_global.pt
    ├── after_language_expert.pt
    └── after_language_global.pt
```

## 顺序约束

每一层必须执行：`export → 实验室重建/采集/上传 → finetune`。下一层 export 必须使用
上一层 `after_*.pt`，并读取前面已经上传的实测 CCD。这样下一层 amplitude 确实来自
实测上游，而不是重新回到纯仿真。

## 保存策略

- 实验室可选保留少量原始 uint16 标定图；正式全量只保存 478 uint8。
- 服务器长期保留 checkpoint、manifest、transport spec、实测 CCD 和性能指标；紧凑
  amplitude PNG 在实验室确认 SHA256 后可以删除，需要时由对应 checkpoint 重新生成。
- 完整 SLM BMP 只在实验室生成，可随时由紧凑 PNG 重建。
- 不保存 `simulation_ccd/`、`ccd_registered/*.pt`、逐样本 block input/output。
- CCD 强度固定按 YAML 的同一输入范围映射，不允许逐图 min/max 或 percentile 拉伸。
  如果标定确认有效信号长期只占 uint16 的一小段，应统一修改
  `camera.saved_frame_input_range` 的上限，而不是对每张图自动拉伸。
- SLM amplitude 因模型已经做 input-RMS 归一化，使用逐样本 99.5 percentile 量化到
  uint8；每张图的 scale 写入 `compact_amplitude_manifest.csv`，不能与 CCD 映射混用。
- 当前协议没有暗场输入，因此不执行背景扣除。
