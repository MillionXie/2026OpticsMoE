# Grocery10 分阶段 SLM/CCD 实验

## 当前最佳 Student 是否包含 Language Model？

包含。当前可复现的最佳版本不是 Vision-only，而是同时替换 Vision stack 和
Language stack 的多模态 Student。Qwen 原生参数全部冻结，Vision/Language 各有一套
独立光路：

```text
Qwen patch embedding
→ Vision input adapter (1024→224) + LN + Softplus
→ Vision electronic Top-4 router
→ Vision 16×(224×224) expert phase（共享一张 986×986 mosaic）
→ 10 cm → CCD-1
→ 每个 expert 独立 LN → ReLU → 乘回同一 routing weight → 未选 expert 置零
→ Vision 986×986 global phase
→ 10 cm → CCD-2
→ pool 224×224 → LN → ReLU → output adapter / residual
→ frozen Qwen vision merger、单个 DeepStack injection、token embedding
→ Language input adapter (2048→224) + LN + Softplus
→ Language electronic Top-4 router
→ Language expert phase
→ 10 cm → CCD-3
→ 同样的 OEO 电子处理
→ Language global phase
→ 10 cm → CCD-4
→ pool/LN/ReLU → output adapter/residual → frozen final RMSNorm
→ LayerNorm(224) → Linear(224,64) → L2 normalize
```

因此一张图最终需要四次曝光。四个相位 mask 对全部样本固定；变化的是每个样本的
振幅图。真实采集按“平面优先”运行，即一个平面一次播放完整文件夹，而不是按样本
交错切换四个平面。

## 目录

正式配置默认生成：

```text
hardware_runs/grocery10_epoch159_ema_stage_first/
├── 00_manifest/
│   ├── play_order.csv
│   ├── deployment.json
│   ├── model_config.yaml
│   └── student_checkpoint.pt
├── 00_input_images/
│   ├── original/
│   └── processor_224/
├── 00_masks/
│   ├── 01_vision_expert/
│   ├── 02_vision_global/
│   ├── 03_language_expert/
│   └── 04_language_global/
├── 01_vision_expert/
│   ├── token_field_224/
│   ├── amplitude_to_play/
│   ├── ccd_captured/
│   ├── electronic_output/
│   └── simulation_reference/{complex_field_real_imag,ccd_intensity,ccd_preview,...}/
├── 02_vision_global/
├── 03_language_expert/
├── 04_language_global/
└── 05_retrieval/
```

`play_order.csv` 是四轮播放共同遵守的唯一顺序。第一轮振幅由 `prepare` 直接生成；
第二、三、四轮的正式 `amplitude_to_play` 必须分别由上一轮真实 CCD 处理后生成。
`simulation_reference` 永远只作对照，不会被真实处理命令当作输入，除非显式添加
`--use-simulation`。

仿真输出严格区分两类对象：`complex_field_real_imag/*.pt` 是探测前复光场，按
`[real,imag,H,W]` 保存；`ccd_intensity/*.pt` 是平方律后的非负强度。相位预览只来自
前者。真实 CCD 只能提供后者，不能从单张 intensity 恢复复相位。正式 110 样本包在
同时保存四平面复场、强度、BMP 和预览时预计需要约 8–10 GB；空间不足可将配置中的
`simulation.save_complex_fields` 关闭，但不会影响真实处理流程。

## CCD 文件要求

每拍完一轮，将文件放入对应的 `ccd_captured/`。文件 basename 必须与播放的振幅 BMP
完全相同，例如：

```text
00000__gallery__Arla-Ecological-Medium-Fat-Milk__12ab34cd.bmp
→ 00000__gallery__Arla-Ecological-Medium-Fat-Milk__12ab34cd.tif
```

允许 `.pt`、`.npy`、`.tif`、`.tiff`、`.png`，要求：

* 内容是 CCD 已经完成平方律探测的光强，不是复场或振幅；代码不会再次平方；
* 无损、单通道，优先 16-bit TIFF/PNG 或 float32 NPY；禁止 JPEG；
* 默认已经完成暗场扣除、几何配准和裁剪，shape 必须严格为 986×986；
* 如果保存全传感器图，必须在硬件配置中显式填写 `capture.roi_xywh=[x,y,986,986]`；
* 不允许隐式 resize。几何偏移应通过相机/4f 配准修正，不能靠缩放掩盖。

`capture.dark_level` 可以配置固定暗电平。曝光增益的全局倍数通常会被训练时的
LayerNorm 消除，但饱和、裁剪、空间非均匀响应不会，因此仍应保持同一曝光和标定。

## 完整真实光路命令

以下命令均从仓库根目录执行。

### 0. 生成四张共享 mask、第一轮振幅、输入图像和仿真参考

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_hardware_deployment.yaml \
  --phase prepare
```

### 1. Vision expert

加载 `00_masks/01_vision_expert/*bmp`，播放
`01_vision_expert/amplitude_to_play/`，采集到
`01_vision_expert/ccd_captured/`，然后执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_hardware_deployment.yaml \
  --phase process_vision_expert
```

它执行训练时完全相同的 per-expert LN、ReLU、乘回原 routing weight 和 hard-zero，
生成第二轮振幅。

### 2. Vision global

加载 Vision global mask，播放第二轮振幅，采集到第二轮 `ccd_captured/`，执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_hardware_deployment.yaml \
  --phase process_vision_global
```

该电子桥包含 detector pooling/LN/ReLU、Vision output adapter/residual、冻结 Qwen
vision merger/DeepStack/token injection，以及 Language input adapter/router；输出第三轮
Language expert 振幅。因此 Language 并没有被跳过。

### 3. Language expert

拍摄第三轮并执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_hardware_deployment.yaml \
  --phase process_language_expert
```

### 4. Language global 与最终检索

拍摄第四轮并执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_hardware_deployment.yaml \
  --phase process_language_global
```

最终 64D 向量、Top-1/Top-3/MRR、逐样本结果和混淆矩阵保存在 `05_retrieval/`。

## 不接硬件的全流程验证

下面会把保存的仿真 CCD 当作“采集文件”，依次走同一套四段电子处理，专门检查目录、
命名、不会二次平方、replay 与原仿真的数值一致性：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_hardware_deployment.yaml \
  --phase all_simulation
```

也可以对任一 `process_*` 命令添加 `--use-simulation` 单独调试。

## BMP 约定

* 振幅：1920×1080，8-bit grayscale BMP，986×986 有效区居中，边界
  `[467,47,1453,1033]`；每张图独立除以其最大值，除数写入 metadata。
* 相位：1920×1200，8-bit grayscale BMP，986×986 有效区居中，边界
  `[467,107,1453,1093]`；物理相位 mod 2π 线性映射到 0–255。
* 仿真和 SLM 都是 8 µm，因此 scale factor=1，不做插值；代码只允许整数 nearest
  scaling，并会拒绝超出 SLM 的尺寸。

实际设备若存在非线性灰阶—相位或灰阶—振幅响应，应在设备控制端应用标定 LUT；本
导出文件采用理想线性编码，并保留 `.pt` 原始物理值用于重新编码。
