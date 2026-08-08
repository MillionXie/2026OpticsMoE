# Grocery10 MoE4 Hardware Deployment

## 光学与 CCD 尺寸

- 仿真 logical active area：478×478。
- 实物 CCD ROI：956×956。
- 实物采集后执行严格 2×2 block mean，得到 478×478；这不是插值。
- 振幅 BMP：1920×1080、8-bit grayscale。
- 相位 BMP：1920×1200、8-bit grayscale；当前导出按折叠光路执行垂直翻转。
- `capture.flip_vertical` 与 `capture.flip_horizontal` 只作用于实测 CCD，不作用于仿真张量。

CCD 厂商 ROI 应直接设置成 956×956。`nearest_resize` 只保留作尺寸容错和诊断，不应替代正确 ROI/光路标定。

## Session 结构

```text
hardware_sessions/<session>/
├── 00_manifest/
│   ├── play_order.csv
│   ├── deployment.json
│   └── sample_metadata/
├── 00_masks/{01_vision_expert,...,04_language_global}/
├── 00_input_images/{original,processor_224}/
├── 01_vision_expert/
├── 02_vision_global/
├── 03_language_expert/
├── 04_language_global/
├── 05_retrieval/
└── 06_hardware_finetune/
```

`play_order.csv` 是四层共同使用的唯一 manifest。每条记录包含 `role`、`dataset_split`、`sample_id`、SKU 与原图路径。任何一层都不得改文件 basename 或播放顺序。

每个物理层目录包含：

```text
amplitude_to_play/      实验室播放的振幅 BMP
ccd_captured/           实验室上传的同名 CCD
registered_ccd/         ROI、方向、尺寸和 binning 审计记录
simulation_reference/   理论 CCD/光场
electronic_output/      CCD 后电子桥及仿真误差记录
```

## 因果微调规则

实测 CCD 是不可微的固定边界：

- CCD-1 后：可训练 Vision OEO/global、完整 Language optics 与最终 readout。
- CCD-2 后：可训练 Vision detector/output bridge、完整 Language optics 与 readout。
- CCD-3 后：可训练 Language OEO/global/detector 与 readout。
- CCD-4 后：只训练 final detector normalization 和 64D retrieval readout。

本层和所有上游相位均冻结。每次适配输出独立 checkpoint、参数清单、split 审计、更新后的下游 phase BMP 和下一层 amplitude BMP。

## 数据边界

```text
gallery → 登记图库/训练 prototype
train   → 默认硬件域适配
query   → 默认独立 test
```

`adaptation.include_test_split=false` 时，test 不参与参数更新。改为 `true` 是允许的实验性 transductive calibration，但最终结果不再是独立测试性能。

完整可复制命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
