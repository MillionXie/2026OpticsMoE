# LSP Extended + LSP 关键点与姿态估计

该独立实验把仓库中已经验证过的 Qwen Vision / Optical MoE16 接口用于单人二维姿态估计，原有实验不会被修改。

## 数据协议

- 14 个关节，顺序严格采用 LSP：右踝、右膝、右髋、左髋、左膝、左踝、右腕、右肘、右肩、左肩、左肘、左腕、颈部、头顶。
- 默认自动下载的是公开可用的 HR-LSPET：它是 LSP Extended 的高分辨率
  重标注子集，实际含 9,428 张图像，全部用于训练。
- 原始 LSP 的前 1,000 张加入训练，后 1,000 张仅作测试。
- 默认正式规模因此为 10,428 train / 1,000 test；没有 validation。若手动提供
  已失效官网曾发布的 10,000 张低分辨率 LSPET，并把
  `lspet_expected_count` 改为 `10000`，则规模为 11,000 / 1,000。
- checkpoint 只按训练损失保存。test 每轮可观察，但不参与选权重或 early stopping。

程序不再依赖已经失效的 `sam.johnson.io`：原始 LSP 优先从
`LiuRunky/Leeds_Sports_Pose` 的 Hugging Face 镜像下载，LSPET 优先从
MPI-INF 的公开 HR-LSPET 镜像下载。HR-LSPET 压缩包约 2.86 GB，HTTP
下载支持 `.part` 断点续传；中途断网后重新执行同一命令即可继续。
Hugging Face 默认使用 `https://hf-mirror.com`，也可以用环境变量
`HF_ENDPOINT` 覆盖。所有镜像失败时，错误会逐项列出原因。也可手动解压成：

```text
data/lsp_pose/
├── lsp_dataset/
│   ├── images/
│   └── joints.mat
└── lspet_dataset/
    ├── images/
    └── joints.mat
```

不同再分发版本对 `joints.mat` 第三通道的可见性编码可能相反。正式配置默认 `coordinates_in_image`：有有效坐标且人体裁剪后仍在图内的关节参与训练和评测。可显式设置 `third_channel_zero_visible` 或 `third_channel_one_visible`。

## 两条模型路径

电子 Teacher 上限：

```text
224×224 RGB
→ frozen Qwen3-VL-Embedding-2B Vision patch/position embedding
→ frozen native Vision blocks
→ last pre-merger spatial hidden [B,1024,Hq,Wq]
→ LayerNorm + Linear(1024,128)
→ lightweight convolutional pose decoder
→ 14×56×56 heatmaps
```

Optical Student：

```text
224×224 RGB
→ frozen Qwen patch/position embedding
→ Linear(Dv,224) + LayerNorm + Softplus
→ [T,224] zero-padding to [224,224]
→ electronic top-4 router
→ 4×4 homogeneous expert bank (16 experts, one 224×224 phase plane each)
→ OEO square detection / per-expert normalization / ReLU / routing reload
→ one 986×986 active global phase
→ 10 cm propagation
→ square-law CCD on 986×986 active area
→ pool/LN/ReLU to [224,224]
→ restore runtime Qwen 2-D token grid using image_grid_thw
→ same lightweight pose-head topology
→ 14×56×56 heatmaps
```

整个 Language Model 不运行。Qwen 原生参数和未使用的 optical hidden output adapter 都被冻结。学生可训练 input adapter、adapter norm、router、专家相位、global phase、OEO 参数和 pose head。

## 监督、指标和坐标约定

训练目标是可见关节 Gaussian heatmap masked MSE，加一个较小的 differentiable coordinate SmoothL1。Student 额外使用 router balance / importance loss。

- `PCK@0.2 torso`：归一化尺度为两条“肩到对侧髋”距离的均值。
- `PCKh@0.5`：head scale 为 `2 × distance(neck, head_top)`。
- 同时报告 mean pixel error、torso-normalized mean error 和逐关节 PCK。

图像先按人物关节包围框做带 margin 的方形裁剪，再 resize 到 224。训练时图像和关键点共享 scale/center/flip 变换；左右翻转同时执行正确的关节索引交换。热图是 56×56，但指标会解码回 224×224 坐标。任何 token/grid 数量不一致都会直接报错，不会静默 reshape、crop 或 truncate。

## 输出

```text
runs/lsp_pose_optical_moe16/
├── resolved_config.yaml
├── dataset.json
├── data_split.csv
├── checkpoints/
├── metrics/
│   ├── teacher_model.json
│   ├── student_model.json
│   ├── *_training_history.csv
│   ├── *_inference.json
│   ├── *_predictions.csv
│   └── comparison.json
└── figures/
    ├── *_training_curves.png
    ├── teacher_inference/
    └── student_inference/
```

姿态图中绿色为 ground truth，红色为 prediction。
