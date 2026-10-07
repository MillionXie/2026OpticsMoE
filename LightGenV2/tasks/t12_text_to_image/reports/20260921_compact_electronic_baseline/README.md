# T12 紧凑单步电子 baseline（2026-09-21）

## 结论

当前保留的紧凑候选是：

```text
文本 -> 冻结 Qwen -> 4.27M residual MLP 条件适配器
     -> 真实 Gaussian seed + 326.83M BK-SDM-v2-Tiny UNet（调用一次）
     -> 49.49M VAE decoder（调用一次） -> 512×512 RGB
```

它没有扩散循环。相比原 SD-Turbo 的 865.91M UNet，紧凑 UNet 减少 62.26%；按“第一个 residual MLP block 的输入到 PIL 图片”计时，RTX 4090、batch 1 中位数为 50.01 ms，batch 4 为 160.11 ms。

四类物体均能生成并可识别，但鞋带、桌腿和织物等细节仍明显模糊。因此它是可继续蒸馏和换光的紧凑候选，不应宣称已经取代原 SD-Turbo 质量基线。

## 数据规模

- ABO CC BY 4.0 单物体子集：train/val/test = 800/100/100，共 1,000 张图片。
- 四类平衡：chair、lamp、shoe、table；每张图只保留一个居中物体。
- 800 张训练图产生 484 条去重的结构化 caption。
- 额外加入 768 条文本组合，不增加图片，用于调用冻结教师生成器扩展条件覆盖。
- 蒸馏缓存：1,252 条训练提示词 × 8 个真实随机 seed = 10,016 个训练 latent；100 条验证提示词 × 4 seed = 400 个验证 latent。

所以“图片只有 800 张”是事实，但训练监督不只有 800 对。额外样本来自同一单步教师在不同文本和随机 seed 下产生的 latent，不把它们冒充真实图片。

## PCA 的作用和取舍

PCA 不是文生图或 VAE 的必需组件。它只用于兼容现有 SD-Turbo/BK-SDM 的 `77×1024` CLIP 条件接口：Qwen 的 2048 维 pooled feature 先预测 128 个系数，再由固定 PCA 基还原为完整 token tensor。

如果直接用一层 `2048 -> 77×1024`，单层就约有 161.5M 权重；当前 residual MLP 只有 4.27M 可训练参数。代价是保存约 10.17M 个固定 PCA buffer（约 40.7 MB，FP32）。验证集上，Qwen 还原条件与原生 CLIP 条件余弦相似度为 0.97338，normalized MSE 为 0.05379。

因此本轮不优先删除 PCA：它不是主要画质瓶颈，而且已经大幅避免直接大投影。待生成主干清晰度稳定后，可以再训练原生少 token 条件接口并与 PCA 版本做配对比较。

## 压缩与质量对照

| 方案 | UNet 参数 | 相对原 UNet | Qwen 全链路 latent nMSE | 结论 |
|---|---:|---:|---:|---|
| SD-Turbo 原 UNet | 865.91M | 基准 | 教师 | 质量参考，太大 |
| 随机小型 latent student | 4.25M–27.24M | -96.9% 至 -99.5% | 0.481–0.502 | 只有色块和轮廓，否决 |
| BK-SDM-v2-Tiny，原生条件蒸馏 | 326.83M | -62.26% | 0.31657（原生 CLIP 条件） | 可识别但模糊 |
| Tiny，再用真实 Qwen 条件细化 | 326.83M | -62.26% | **0.36432** | 当前紧凑候选 |
| BK-SDM-v2-Small，两轮蒸馏 | 485.79M | -43.90% | 0.39314 | 参数增加但没有清晰度收益，否决 |

Tiny 的 Qwen 条件细化在第 2 epoch 最佳；继续到 epoch 3/4 已发生验证过拟合。潜空间梯度约束只带来很小的数值提升，没有肉眼可见地解决模糊，因此不再继续堆 epoch。

## 参数和延迟边界

- Qwen 条件适配器：4,273,280 个可训练参数。
- 选中紧凑 UNet：326,825,604 参数。
- VAE decoder 容器：49,490,179 参数；完整 VAE（含只在编码时用到的 encoder）为 83,653,863 参数。
- 实际生成尾部（适配器 + UNet + VAE decoder）：约 380.59M 参数；冻结 Qwen 文本前端另计。
- 延迟起点：适配器中第一个 `ResidualMLP` 的输入；不包含 Qwen 编码和适配器最前面的 LayerNorm/Linear/SiLU stem。
- 延迟终点：VAE 解码结果完成张量到 PIL 的转换。
- RTX 4090，40 次热启动：batch 1 = 50.01 ms median / 53.00 ms p95；batch 4 = 160.11 ms median / 162.35 ms p95。
- 生成调用严格为一次 compact UNet + 一次 VAE decode，没有采样循环。

## 下一步

暂不继续缩小推理网络，也暂不接光。下一轮优先采用只增加训练成本、不增加推理电子模块的方法：教师中间特征蒸馏，加 decoded-image perceptual/patch loss；若仍然模糊，再引入训练期判别器。达到固定人工审计阈值后，再从 Tiny 的中间 block 开始做“电子 residual 与光 block 并行”的逐块替换。

正式实现：`compact_turbo.py`、`compact_turbo_training.py`、`compact_turbo_run.py`、`compact_turbo_infer.py`。服务器运行目录：

- Tiny 原生条件蒸馏：`/DATA/DATA1/guest3/t12_assets/runs/bksdm_v2_tiny_one_step_finetune_full_b32`
- Tiny Qwen 条件细化：`/DATA/DATA1/guest3/t12_assets/runs/bksdm_qwen_detail_e4`
- Small 否决试验：`/DATA/DATA1/guest3/t12_assets/runs/bksdm_v2_small_native_e2`

结构测试：服务器 27 passed。所有本任务训练、评测结束后 GPU 3/4 均回落到 12 MiB；A100 未用于本轮训练。
