# ABO easy100 冻结 Qwen3-VL-Embedding-2B 基线

## 结论

本数据是图搜文：每张 test 商品图在 100 个官方英文商品标题中检索唯一正确标题。模型为
`Qwen3-VL-Embedding-2B`，全部 2,127,532,032 个参数冻结，可训练参数为 0；没有使用 train、
没有微调大模型，也没有训练额外读出头。

| 项目 | 正式结果 |
| --- | ---: |
| test 图像数 | 2,400 |
| R@1 | 0.737083 |
| R@5 | 0.933750 |
| R@10 | 0.960417 |
| MRR | 0.823033 |
| mean / median rank | 2.6654 / 1 |
| CUDA mean / median | 26.0516 / 25.7413 ms/图 |
| CUDA P5 / P95 | 25.4964 / 28.0221 ms/图 |
| idle / active mean / active peak | 67.72 / 152.50 / 156.84 W |
| active board energy | 3.9729 J/图 |
| idle-subtracted energy | 2.2087 J/图 |

## 测量合同

- 性能使用官方全部 2,400 张 test；train 的 4,800 张完全没有进入本基线。
- 速度/功耗固定 `batch=1`，先做 50 次不计时预热，再从每个商品取 2 张，共 200 张。
- 起点为第一个原生 Vision Transformer block 输入；终点为图像 embedding 归一化、与预计算
  的 100 个标题 embedding 求相似度并完成完整排序。
- 文件 I/O、解码、processor/tokenizer、patch embedding、模型加载、标题库预计算不计入。
- 功率原始样本由 `nvidia-smi power.draw` 以 10 ms 请求间隔记录，仅模型在线窗口标为 active。
- 正式测量前 5090D 无其他计算进程；环境为 PyTorch 2.8.0+cu128、bfloat16。

## 证据位置

完整产物位于任务目录的
`runs/simulation/qwen_frozen_5090d_easy100_controlled/`，包括报告 JSON、2,400 条预测、
200 条计时、原始功率采样、图像/标题 embedding、运行命令、汇总图、当次精确源码和
`ARTIFACTS.sha256`。本地下载后已再次逐项通过 SHA256 校验。

关键不可变校验：模型 `model.safetensors` SHA256 为
`c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f22d486964464509c1`；test CSV SHA256 为
`4ca5834abf4bb4d614b41f05b5b2481b3d41720dd78c1c37b05923c97657dfbb8`。
