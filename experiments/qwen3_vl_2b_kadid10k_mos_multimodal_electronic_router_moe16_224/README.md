# KADID-10k DMOS · Qwen3-VL-2B · Multimodal Optical MoE16-224

这个独立实验把
`qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224`
的 vision + language 光学 MoE16-224 蒸馏框架迁移到 KADID-10k。原 SPAQ
工程不受影响。

## 为什么换 KADID-10k

SPAQ 是真实拍摄图像上的主观质量评价；图像内容、拍摄设备和主观评分共同变化，
模型容易学习内容偏差。KADID-10k 使用 81 张参考图像生成 25 类、每类 5
级人工失真，共 10,125 张失真图像。它更适合先检验光学网络是否能学习清晰的
失真强度—质量关系。

KADID-10k 的官方 `dmos.csv` 分数范围为 1–5，本工程按“分数越高，质量越好”
解释：

- 训练时映射为 `(DMOS - 1) / 4`；
- loss 在 0–1 空间计算；
- MAE、RMSE、预测 CSV 和散点图转换回原始 1–5 分制；
- SRCC/PLCC 对线性缩放不敏感。

## 无内容泄漏划分

不能随机按失真图像划分。同一参考图像的不同失真版本若分别进入训练集和测试集，
会造成明显的内容泄漏。

本工程以 `reference_image` 为单位，用 seed 42 固定划分：

- 80% reference IDs 进入训练集；
- 20% reference IDs 进入测试集；
- 81 个参考图像时，默认是 65 个 train references 和 16 个 test references；
- 对应完整数据通常为 8,125 张 train、2,000 张 test；
- 划分写入 `data_split.json`，已有文件与当前数据不一致时直接报错。

teacher head 的内部 validation 也按 reference ID 划分。Student 延续当前 SPAQ
实验习惯：每个 epoch 在 test 上报告结果并按 test SRCC 保存 best，因此该 best
值属于 selection-biased test 指标，不应当在论文中当作严格独立的最终测试估计。

## 自动下载

正式和 smoke 配置默认：

```json
"download": true
```

若 `data/kadid10k` 中没有完整数据，程序会从 KADID-10k 官方服务器下载约
3.1 GB 的 `kadid10k.zip`，支持 `.part` 断点续传，执行 ZIP 路径安全检查后递归
发现 `dmos.csv` 和图像。默认解压成功后删除 ZIP，节省空间。

也可以手动准备；目录层级可以保留官方 ZIP 的原结构，loader 会递归发现：

```text
data/kadid10k/
  .../dmos.csv
  .../images/I01_01_01.png
  ...
```

若存在多个有效 `dmos.csv`，必须显式配置 `annotations_file`，不会静默猜测。

## 模型

Teacher：

```text
RGB distorted image + fixed DMOS prompt
→ frozen full Qwen3-VL-2B vision stack
→ frozen full Qwen3-VL-2B language stack
→ trainable normalized linear regression head
→ normalized DMOS
```

Student：

```text
frozen Qwen patch embedding
→ Optical Vision MoE16-224 (4 stages)
→ frozen vision merger + native 3-point DeepStack injection
→ Optical Language MoE16-224 (4 stages)
→ frozen final RMSNorm
→ same regression-head structure
→ normalized DMOS
```

每个 Optical MoE：

- 4×4 共 16 个 224×224 专家；
- electronic input-dependent top-4 router；
- 每个专家 4 个相位 stage；
- 专家间隔 30 pixels；
- 有效面 986×986，传播 canvas 1026×1026；
- 每段传播距离为 10 cm；
- detector 观察 986×986 有效区并池化回 224×224；
- Transformer identity residual 保留；
- attention prelude 默认关闭；
- vision 与 language 都保留 Qwen teacher hidden 蒸馏。

Qwen processor pixel budget 从 25,600 提高到 **37,632**。对常见纵横比，它会提供
更多视觉 token，同时仍受 `max_visual_tokens=224` 的硬检查约束；超过 224 会明确
报错，不会裁剪 token。

## Loss

```text
L =
  1.0 × normalized vision hidden MSE
+ 1.0 × normalized answer hidden MSE
+ 0.5 × teacher prediction SmoothL1
+ 1.0 × ground-truth SmoothL1
+ 0.1 × pairwise ranking loss
+ 0.1 × Norm-in-Norm
+ 0.01 × router balance
```

Norm-in-Norm 将一个 batch 内的预测和标签分别去均值并除以向量范数，再比较二者
形状。它主要约束相对排序和相关性，对整体平移/正比例缩放不敏感，因此不能替代
绝对分数 loss。本工程仅给它 0.1 的辅助权重，主要监督仍是 ground-truth
SmoothL1 和 teacher losses。

## Cache

冻结 Qwen 的 processor 输出和 teacher hidden 存在工程内 `cache/`，与 `runs/`
分离，便于不同调参 run 复用。cache identity 包含：

- KADID dataset 与 DMOS task；
- annotation path 与 reference split digest；
- 1–5 分数范围及方向；
- prompt；
- Qwen model ID；
- processor pixel budget 37,632；
- dtype 和 attention implementation。

改变任何身份字段都会使用新的 cache；旧 SPAQ/KADID 分类 cache 不会被误用。

## 快速开始

见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。`prepare_data` 会自动下载；完整
`--phase all` 会依次准备数据、缓存 Qwen 输入和 teacher hidden、训练 teacher
head、生成 teacher prediction、训练 student 并推理比较。
