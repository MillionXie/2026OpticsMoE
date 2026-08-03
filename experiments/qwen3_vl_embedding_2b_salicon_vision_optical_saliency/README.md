# SALICON Qwen Vision Optical Saliency

这个实验预测“人会看向图像哪里”，不是 FSS-1000 那种前景/背景二值分割。输入为
`224×224 RGB` 图像，输出为归一化的人类注视概率密度图
`[B,1,224,224]`。

## 数据与划分

代码自动准备 SALICON 2015r1：

- 官方 train：10,000 张有注视标注的 COCO 2014 图像，用于训练；
- 官方 validation：5,000 张有注视标注的图像，用于选择 checkpoint 和评测；
- 官方 test 的标注不公开，本实验不把它伪装成可评测 test。

大体积 fixation JSON 使用流式解析，避免把整份标注膨胀为数十 GB 的
Python 对象。离散 fixation 和连续 density map 会一次性生成到
`cache/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/prepared_maps/`。
下载使用 `.part` 文件并支持服务器允许时的 HTTP Range 续传。

## 电子上限

```text
224×224 RGB
→ frozen Qwen3-VL-Embedding-2B Vision
→ final pre-merger spatial hidden [T,1024]
→ restore runtime token grid
→ lightweight continuous saliency decoder
→ raw map logits [1,224,224]
→ spatial softmax
→ fixation density
```

只训练电子 saliency decoder，Qwen 始终冻结且保持 eval。

## Optical Student

光学部分严格复用当前已验证的单层 baseline：

```text
224×224 RGB
→ frozen Qwen patch/position embedding
→ Linear(1024,224) + LayerNorm + Softplus
→ electronic Top-4 router
→ 4×4 homogeneous MoE16
   (16 independent 224×224 phase-only experts, one phase layer each)
→ 10 cm propagation + per-expert square detection/LN/ReLU/reload
→ 986×986 global phase
→ 10 cm propagation
→ 986×986 active CCD ROI on a 1026×1026 FFT canvas
→ detector pooling/LN/ReLU to [224,224]
→ read the first T token rows and restore image_grid_thw
→ the same lightweight saliency decoder
→ density map
```

Student 不运行 language model，不做全局池化，也不调用不参与空间预测的 hidden
output adapter。可训练部分为 vision input adapter、router、16 张 expert phase、
global phase、OEO/CCD 必要参数和 saliency head。

## Loss 与指标

默认损失：

```text
KLD + 0.5(1-CC) + 0.25(1-SIM) - 0.1 NSS
+ 0.03 router balance + 0.005 router importance
```

报告 `KLD↓ / CC↑ / SIM↑ / NSS↑ / AUC-Judd↑ / MAE↓`。可选
`salicon_mask_kd.yaml` 只蒸馏电子 Teacher 的最终密度图。因为缓存的 Teacher
map 位于原始 224×224 坐标系，该配置会关闭随机 crop/flip，防止空间错位。

输出位于实验目录内的 `runs/salicon_vision_optical_saliency/`，不会写到仓库
根目录的公共 `runs/`。其中包含配置、数据 manifest、模型参数报告、checkpoint、
训练曲线、指标 JSON、Teacher/Student 预测图和光学相位图。

数据来源与定义见 [SALICON](https://salicon.net/)；指标实现依据
[SALICON evaluation](https://github.com/NUS-VIP/salicon-evaluation)。
