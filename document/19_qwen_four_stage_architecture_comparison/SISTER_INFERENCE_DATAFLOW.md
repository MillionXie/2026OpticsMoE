# 师姐 `teams/inference` 工程说明

## 1. 这个目录的定位

`teams/inference/inference` 是一个 ABO easy100 image-to-title 的 inference-only 包。
它能：

- 加载 Qwen 与一份 `abo_i2t_inference.pt`；
- 对 2400 张商品图和 100 个官方英文标题编码；
- 计算图像到标题的 R@1/R@3/R@5/R@10、MRR 和 median rank；
- 复用四级硬件导出/CCD 回放流程。

它不能：

- 训练或更新参数；
- 仅凭当前目录重现该权重的优化过程；
- 单独运行而不依赖主仓库中的 warmstart/robust/Qwen 公共代码和外部 Qwen 权重。

权重文件只保存光电替换模块、任务 readout 和小型 adapter，不包含完整 Qwen 2B
checkpoint。

## 2. 数据任务

候选库固定为 100 个产品的 100 条官方英文标题，每个产品 1 条。query 是 2400 张
商品图，每类 24 张。

两种输入都被包装为 Qwen chat：

```text
system: Represent the exact catalog product for image-to-title retrieval.
user image: <image>
```

或：

```text
system: Represent the exact catalog product for image-to-title retrieval.
user text:  <official English product title>
```

同一个 batch 不混合图像与文本。

## 3. 图像 query 的前向

图像前向与我们的 warmstart5 四级主体相同：

```text
224×224 image
→ frozen Qwen patch/position front-end
→ [196,1024]
→ Vision expert hybrid block
→ Vision global hybrid block
→ [196,1024]
→ frozen Qwen main merger
→ [49,2048] image tokens inserted into prompt
→ Language expert hybrid block
→ Language global hybrid block
→ [L,192]
→ mean+max pooling
→ [384]
```

光学传播、MoE4 top-2、CCD `478→224→192` readout、四个融合门以及 10 cm 物理
模型都直接复用我们的实现，并没有在 `teams/inference` 里另写一套。

## 4. 标题 gallery 的前向

纯文本标题没有 `pixel_values`，所以不会执行 Vision 前端、Vision expert 或 Vision
global。它从 Qwen token embedding 开始：

```text
title tokens
→ frozen Embedding(...,2048)
→ Language expert hybrid block
→ Language global hybrid block
→ mean+max
→ [384]
```

因此硬件采集时：

- image query 需要四级 CCD；
- text title 只需要 Language expert/global 两级 CCD；
- 程序按 modality 自动判断某个样本在某一级是否需要采集。

## 5. 师姐版本的 readout

我们的 warmstart5 使用单层投影；师姐版本把它加强为：

```text
[384]
→ LayerNorm(384)
→ Linear(384,128)
→ GELU
→ Linear(128,64)
→ L2 normalize
→ base embedding [64]
```

该 readout 有 58,304 个参数。

## 6. 图像/文本双适配器

基础 64 维 embedding 后，图像和标题分别进入一个 rank-32 residual adapter：

```text
x [64]
→ non-affine LayerNorm(64)
→ Linear(64,32, no bias)
→ GELU
→ Linear(32,64, no bias)
→ x + update
→ L2 normalize
```

image adapter 与 text adapter 参数不共享。每个 4,096 参数，两者共 8,192 参数。

最终：

```text
score(i,j) = image_adapter(e_image_i) · text_adapter(e_title_j)
```

由于两端已经 L2 归一化，点积就是 cosine similarity。得到 `2400×100` 分数矩阵，
每行由高到低排序。

## 7. 与我们模型参数结构的关系

| 组件 | 师姐版本参数量 | 推理时状态 |
|---|---:|---|
| Vision 四级主体中的 Vision 部分 | 1,330,431 | frozen/eval |
| Language 四级主体中的 Language 部分 | 1,723,135 | frozen/eval |
| nonlinear readout | 58,304 | frozen/eval |
| dual rank-32 adapter | 8,192 | frozen/eval |
| **inference weight 合计** | **3,120,062** | 全部不可训练 |

此外仍需外部 Qwen 冻结前端权重；它不在 12.5 MB 的 inference weight 内。

## 8. 权重与数据防混用

加载器检查：

- weight format 必须为 `abo_i2t_inference_v1`；
- 必须标记 `inference_only=true`；
- manifest SHA-256 必须与 2400 query 清单一致；
- source-config SHA-256 必须与 100 标题定义一致；
- state dict 只能包含预期的四组权重，不允许额外字段。

这些校验能防止把另一数据集、另一候选标题库或另一网络 head 的权重静默装入。

## 9. 硬件四级顺序

顺序固定：

```text
vision_expert → vision_global → language_expert → language_global
```

每级支持 `export → validate → evaluate`：

1. export 生成该级紧凑振幅输入与相位图；
2. 实验室播放并采集同名 `478×478` CCD；
3. validate 检查数量、文件名和 CCD 合同；
4. evaluate 用已经采集的上游/当前 CCD 替换对应仿真输出。

`--max-per-class 1` 只可做连通性检查。正式 2400-query 结果要求全部 query 与全部
100 title 都满足相应的实测阶段。

## 10. 能和不能从当前目录得出的结论

可以确认：

- 前向网络、shape、任务人口、权重格式和硬件顺序；
- README 声明完整集复现目标 `R@1=0.8321`；
- 最终使用 nonlinear readout 和 dual adapter。

不能确认：

- 这份权重训练了多少 epoch；
- 训练 loss、学习率、训练/验证划分和选模规则；
- dual adapter 是与主体联合训练还是训练后单独拟合。

原因是该目录有意只发布 inference entry point，checkpoint provenance 只记录基础权重、
adapter 与数据身份 SHA，没有训练日志。给老师汇报时应把 83.21% 表述为“该交付包
README 声明的完整集 R@1”，不要补写不存在的训练过程。
