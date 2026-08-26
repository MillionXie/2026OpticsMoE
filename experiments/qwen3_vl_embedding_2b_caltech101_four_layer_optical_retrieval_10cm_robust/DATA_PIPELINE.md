# 四层硬件数据流程

## 1. 服务器和实验室的职责

服务器负责：

- 从 checkpoint 生成每层478×478紧凑振幅 PNG；
- 生成每层478×478紧凑相位和1920×1200原生相位 BMP；
- 校验 manifest、basename、尺寸、位深和极性合同；
- 读取上传的实测 CCD，完成逐层下游微调和检索评估；
- 保存 checkpoint、配置、manifest、实测 CCD 和指标。

实验室电脑负责：

- 把紧凑振幅重建为1024×1024原生 Meadowlark BMP；
- 手动把本层唯一的1920×1200相位 BMP 加载到相位 SLM；
- 高速播放振幅 SLM，并由 TUCam 采集 CCD；
- 直接保存478×478、8-bit灰度 PNG；
- 保持 basename 不变并上传到同一层目录。

实验室流程不做背景扣除、逐图对比度拉伸、翻转或训练。正式采集不需要长期保存大尺寸16-bit原图。

## 2. 正常振幅极性

整个新工程只有一个极性合同：

```text
command 255 = white = bright/open = 透光
command   0 = black = dark/closed = 遮光
invert_before_export = false
```

`compact_amplitude` 和 `amplitude_to_play` 都遵守这个合同。禁止使用旧 `_inv`、`inverted` 或“黑色代表透光”的标定包。

正常极性的棋盘格与光栅标定产物由下列配置生成：

```text
experiments/hardware_sdk/generators/slm_patterns/configs/dual_slm_17um_8um.yaml
```

推荐检查配对位于独立目录，目录内只有两张 BMP：

```text
recommended_checker_grating_pair/
├── amplitude_checker_255open_c64_1024x1024.bmp
└── phase_grating_xy_in_255open_cells_c64_p8_1920x1200.bmp
```

相位光栅只出现在振幅白色/透光位置；两张文件不能与历史反相版本交叉配对。

## 3. Session目录

```text
hardware_sessions/four_layer_10cm_robust_run1/
├── manifest.csv
├── 01_vision_expert/
│   ├── compact_amplitude/             # 服务器生成，478 PNG
│   ├── compact_amplitude_manifest.csv
│   ├── compact_phase/                 # 服务器生成，478 PNG
│   ├── amplitude_to_play/             # 实验室重建，1024 BMP
│   ├── phase_to_play/                 # 1920×1200，本层恰好1张BMP
│   ├── ccd_captured/                   # 实验室输出，478 PNG
│   ├── acquisition_logs/
│   ├── transport_spec.json
│   └── finetune_metrics.json
├── 02_vision_global/
├── 03_language_expert/
├── 04_language_global/
└── checkpoints/
    ├── after_vision_expert.pt
    ├── after_vision_global.pt
    ├── after_language_expert.pt
    └── after_language_global.pt
```

`manifest.csv` 是样本顺序和 basename 的唯一依据。每个 `ccd_captured/*.png` 必须与本层 `amplitude_to_play/*.bmp` 同 stem。

## 4. 紧凑振幅与原生振幅

服务器对每张模拟振幅使用固定算法编码到8-bit：非负截断后以99.5 percentile作为量化尺度，结果和尺度写入 `compact_amplitude_manifest.csv`。

实验室执行1:1重建：

```text
478×478 uint8 PNG
→ 不缩放
→ 以配置中心放入1024×1024
→ 外围填0（黑色/遮光）
→ 8-bit灰度BMP
```

重建不再次归一化、不翻转，也不把255和0交换。

## 5. 相位导出

相位训练张量保持478×478、17 µm逻辑采样。导出时：

```text
phase = phase mod 2π
→ uint8 0..255
→ 按配置做纵向翻转
→ 17/8物理坐标nearest栅格化
→ 约1016×1016原生有效区
→ 放到1920×1200中心(980,590)
```

这不是普通图片 resize。相位中心是硬件标定参数，修改中心后重新执行导出即可，不改变训练模型。

## 6. CCD保存与服务器读取

实验室相机配置：

```text
experiments/hardware_sdk/configs/tucam_meadowlark_1024_windows.yaml
```

必须先填写真实 `camera.device_roi_xywh`、曝光时间、LUT和固定输入强度范围。相机工作流把硬件 ROI 用统一规则重采样到478×478，并按统一 `saved_frame_input_range` 映射到8-bit。禁止逐帧 min/max 或 percentile 拉伸。

实验室保存的 CCD 不翻转。服务器加载时根据正式 YAML 执行配置的纵向和横向翻转，再输入 CCD normalizer。没有实拍暗场，因此不做背景扣除。

## 7. 四层依赖关系

| 当前层 | export使用的checkpoint | export依赖的实测CCD | fine-tune输出 |
|---|---|---|---|
| `vision_expert` | 训练EMA checkpoint | 无 | `after_vision_expert.pt` |
| `vision_global` | `after_vision_expert.pt` | 第1层 | `after_vision_global.pt` |
| `language_expert` | `after_vision_global.pt` | 第1～2层 | `after_language_expert.pt` |
| `language_global` | `after_language_expert.pt` | 第1～3层 | `after_language_global.pt` |

每层必须完成：

```text
服务器export
→ 实验室重建/播放/采集
→ 上传同名CCD
→ 服务器fine-tune
→ 才能export下一层
```

提前导出下一层会使输入重新退回仿真上游，破坏正式四层实测链。

每层fine-tune还执行参数级采集合同检查：任何决定本层已播放振幅或已采CCD的上游参数均保持冻结，只允许测量边界后的电子处理、尚未采集的后续光学层和检索头更新。若修改trainable范围破坏该合同，程序会在建立优化器前终止；不能用旧CCD继续训练已经改变其输入生成器的模型。

## 8. 最后一层快速模式

快速 session 与正式四层 session 分开保存。使用：

```text
stage = language_global
upstream_source = simulation
```

export时前三层全部使用训练仿真，服务器直接生成第四层理论振幅。fine-tune时只安装第四层实测 CCD，前三层继续仿真。输出为 `after_language_global.pt` 和第四层 `finetune_metrics.json`。

## 9. 保存与传输策略

长期保留：

- 训练和逐层 checkpoint；
- resolved config、environment和manifest；
- 每层478×478实测 CCD；
- acquisition logs、transport spec和评估指标；
- 最终4张相位 BMP及其导出报告。

可重建、无需长期保留：

- `amplitude_to_play` 的1024×1024 BMP；
- 中间 `compact_amplitude`；
- 仿真 CCD；
- 逐样本 float32 tensor cache。

删除可重建数据前，应先确认 `compact_amplitude_manifest.csv`、checkpoint和配置已保存，并确认实验室上传数据完整。
