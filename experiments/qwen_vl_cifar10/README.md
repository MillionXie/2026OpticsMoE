# Qwen3-VL on CIFAR-10

这是一个独立的 Qwen3-VL CIFAR-10 实验模块，不依赖仓库中的 CLIP visual prompting 代码。

具体运行命令请只看 [COMMANDS.md](COMMANDS.md)。常用参数已经放入 `configs/*.json`，不需要再维护很长的命令行。

## 路径约定

本模块不会再根据仓库根目录随意创建 `data/` 或 `runs/`：

- 默认数据目录：`experiments/qwen_vl_cifar10/data/`
- 默认结果目录：`experiments/qwen_vl_cifar10/runs/qwen_vl_cifar10/`
- `configs/*.json` 中的相对路径以 JSON 文件所在目录为基准。
- 命令行显式提供的相对路径以当前终端目录为基准。
- 所有解析后的绝对路径都会写入本次运行的 `config.json`。

## 环境

当前机器已经创建并验证以下环境：

```powershell
conda activate qwen3vl-cifar10
```

主要版本：

- Python 3.11
- PyTorch 2.11.0+cu128
- torchvision 0.26.0+cu128
- Transformers 4.57.3
- Accelerate 1.14.0
- PEFT 0.19.1

CPU fallback 可用，但 Qwen3-VL 实际建议使用 GPU。当前 RTX 4070 Laptop 只有 8GB 显存，先使用 2B 模型和较小 batch size。

## 模式

- `mlp`（默认）：冻结完整 Qwen3-VL，只训练 `Linear -> GELU -> Dropout -> Linear` 分类头。
- `generate`：使用原生生成能力输出 CIFAR-10 类名。
- `lora`：最小 LoRA 扩展，只微调用户选择的模块。

本项目不实现、不启用 `tune_mm_mlp`，并明确拒绝把 projector/mm_mlp 设置为 LoRA target。

支持模型：

- `Qwen/Qwen3-VL-2B-Instruct`（默认）
- `Qwen/Qwen3-VL-4B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3-VL-30B-A3B-Instruct`
- `Qwen/Qwen3-VL-235B-A22B-Instruct`

## 配置文件

现有配置：

- `configs/smoke_mlp_2b.json`：32/32 样本的 smoke test，batch size 1。
- `configs/mlp_2b.json`：完整 MLP 训练，batch size 2、20 epochs。
- `configs/generate_2b.json`：前 100 个测试样本的生成对照。

加载规则：JSON 提供默认值，显式 CLI 参数优先。例如：

```powershell
python main.py --config configs/mlp_2b.json --batch-size 4
```

## 图像和特征

`image_size=32` 表示 CIFAR-10 未修改的原图大小，不会触发手动 resize。图像以 32x32 PIL RGB 传入 Qwen processor，由 processor 完成内部 resize/tokenize。只有显式设置 `resize_to` 才会在 processor 前调整尺寸。

支持的 `feature_source`：

- `visual_tokens_mean`（默认）
- `vision_pooler`
- `multimodal_image_tokens_mean`
- `last_hidden_mean`

如果当前 Transformers 实现不支持某个来源，程序会给出可用 fallback。`feature_dim` 自动推断，不硬编码。

## Feature cache

MLP 配置默认启用 feature cache：

- `train_features.pt`
- `test_features.pt`

缓存包含 features、labels、类别名、模型、feature source、image size、resize、dtype、split 和样本数。只有 metadata 全部匹配才会复用。缓存读取时间不会计入 Qwen 特征提取时间。

## 输出

每次成功运行都会在配置的 `output_dir` 保存：

- `config.json`
- `metrics.json`
- `predictions.csv`
- `confusion_matrix.csv`
- `best_head.pt`

MLP cache 运行还会保存特征文件；LoRA 模式额外保存 `best_lora_adapter.pt`。

`metrics.json` 包含准确率、macro-F1、各类准确率、loss、模型加载时间、特征提取速度、分类头训练速度、cold/steady latency、端到端吞吐和 CUDA peak memory。

