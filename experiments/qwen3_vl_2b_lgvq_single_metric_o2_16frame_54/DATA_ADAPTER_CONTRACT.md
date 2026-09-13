# Spatial-4 原始 LGVQ 数据适配合同

本文件说明如何把已有的原始 LGVQ 视频转换成正式 `SRCC=0.671008`
checkpoint 所需的唯一一组输入。缓存是为了避免每个 epoch 重复运行冻结前端，
不是额外的数据集，也不是新的预测分支。

## 必需资产

- 原始 LGVQ：`prompt_cls.json`、`MOS.txt`、`videos/`。
- 官方 Qwen3-VL-2B 本地权重：只用于一次性生成冻结的图像 patch/位置特征和
  Spatial prompt embedding；学生推理不运行任何 Qwen Transformer/Attention block。
- `deployment/preprocessing/quality_stem_state.pth`：本模型专用的冻结 Conv5
  输入变换，339,312 个 FP32 参数，约 1.29 MiB。它只注入电子残差 E1，
  不直接连接最终 MOS 读出头，也不是另一份预测 checkpoint。
- `deployment/checkpoints/best_observed_test_checkpoint.pt`：唯一的正常加光学生模型。

从原始视频计数时，学生模型 3,786,407 个参数加上冻结 Conv5 的 339,312 个参数，
合计 4,125,719 个本项目参数。官方 Qwen 冻结前端参数单独归属于基础模型，不复制
进轻量交接包。

## 固定采样和张量合同

每个视频均匀采 4 帧，采样 offset 为 `0.0`，每帧中心裁剪短边的 65%，再缩放：

- Qwen Vision patch/位置前端：`[N,4,196,1024]`，FP16；不执行 Vision block
  和 merger。
- 同一 Qwen 缓存内的固定 14 通道质量量：`[N,4,196,14]`，FP16；通道顺序
  写在缓存元数据中。
- 原始 RGB：`[N,4,3,224,224]`，UINT8；供小型自写 Conv E1 修正使用。
- 冻结 Conv5 输出：`[N,4,196,192]`，FP16；只注入 E1。
- Spatial prompt：tokenizer + 冻结 `embed_tokens`，`[1,L,2048]`，FP16，附 mask。

四种视频相关张量必须具有完全相同的 `sample_ids` 顺序。正式划分是按 prompt
group 固定划分的 2250 train + 558 test，不设 validation；不能按文件枚举顺序
自行重排。

## 从原始视频重建

在 ZIP 解压根目录执行。PowerShell 示例：

```powershell
conda activate xml
$P = 'experiments\qwen3_vl_2b_lgvq_single_metric_o2_16frame_54'
$D = 'D:\path\to\LGVQ'
$Q = 'D:\path\to\Qwen3-VL-2B'
$O = "$P\deployment\data"
New-Item -ItemType Directory -Force $O | Out-Null

python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.prepare_manifest `
  --dataset-root $D `
  --output "$O\lgvq_train2250_test558.csv" `
  --seed 42

python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.cache_qwen_front `
  --dataset-root $D `
  --model-path $Q `
  --vision-output "$O\qwen3vl_front_4f_196x1024_quality14.pt" `
  --language-output "$O\qwen3vl_front_spatial_prompt_2048.pt" `
  --target spatial `
  --manifest "$O\lgvq_train2250_test558.csv" `
  --frame-count 4 `
  --token-grid 14 `
  --batch-size 2 `
  --device cuda

python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.cache_raw_frame_view `
  --manifest "$O\lgvq_train2250_test558.csv" `
  --output "$O\lgvq_four_frames_center100_224_uint8.pt" `
  --sampling-offset 0.0

python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.cache_quality_stem `
  --frame-cache "$O\lgvq_four_frames_center100_224_uint8.pt" `
  --checkpoint "$P\deployment\preprocessing\quality_stem_state.pth" `
  --output "$O\lgvq_quality_conv5_4f_196x192.pt" `
  --batch-size 32 `
  --device cuda
```

随后把 `configs/deployment/spatial4_readout_1m_lab.yaml` 中 `data.dataset_root`
指向 `$D`。其余四个缓存文件名已经与上述输出一致。运行：

```powershell
python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.run `
  --config "$P\configs\deployment\spatial4_readout_1m_lab.yaml" `
  --phase preflight
```

只有 preflight 报告全部 sample ID、形状、dtype、Qwen 前端指纹和 checkpoint
架构合同一致后，才进行仿真评估、硬件采集或微调。不要用 `strict=False` 绕过检查。
