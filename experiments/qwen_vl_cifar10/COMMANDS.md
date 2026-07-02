# Qwen3-VL CIFAR-10 运行指令

所有命令都在本目录执行。配置文件中的相对路径以配置文件自身所在目录为基准，不受 PowerShell 当前目录影响。

H20 上的 32B 纯推理、分阶段计时和中间特征审计见
[`H20_INFERENCE.md`](H20_INFERENCE.md)。该流程不训练分类头。

当前三个配置都把数据保存到：

```text
experiments/qwen_vl_cifar10/data/
```

实验结果保存到：

```text
experiments/qwen_vl_cifar10/runs/<实验名>/
```

## 1. 进入环境和实验目录

```powershell
conda activate qwen3vl-cifar10
cd C:\Users\Xml12\OneDrive\2026OpticsMoE\experiments\qwen_vl_cifar10
```

确认解释器正确：

```powershell
python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```

预期包含：`2.11.0+cu128 True 4.57.3`。

## 2. Smoke test

先运行这个。它只使用 32 个训练样本和 32 个测试样本，batch size 为 1：

```powershell
python main.py --config configs/smoke_mlp_2b.json
```

输出位置：

```text
runs/smoke_mlp_2b/
```

首次运行会下载 Qwen3-VL-2B 权重和 CIFAR-10 数据。

## 3. 完整 MLP 实验

```powershell
python main.py --config configs/mlp_2b.json
```

输出位置：

```text
runs/mlp_2b/
```

默认 batch size 为 2，适配当前 8GB RTX 4070 Laptop。确认显存充足后，可以临时覆盖为 4，无需修改 JSON：

```powershell
python main.py --config configs/mlp_2b.json --batch-size 4
```

特征提取和分类头训练现在使用独立 batch size。服务器配置使用较小的
`feature_batch_size` 控制 Qwen 显存，同时使用 `head_batch_size=512`，避免分类头产生几十万
个低效小 batch。

服务器 4090 的 4B、8B 和多卡 30B/32B 配置及显存依据见
[`SERVER_MODELS.md`](SERVER_MODELS.md)。单张 4090 推荐的最大 BF16 模型是 8B：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python main.py \
  --config configs/mlp_8b_server_1x4090.json --smoke-test
```

## 4. Generate 对照实验

```powershell
python main.py --config configs/generate_2b.json
```

输出位置：

```text
runs/generate_2b/
```

## 5. 查看结果

```powershell
Get-Content runs/smoke_mlp_2b/metrics.json
Import-Csv runs/smoke_mlp_2b/predictions.csv | Select-Object -First 10
```

每次成功运行的目录中包含：

```text
config.json
metrics.json
predictions.csv
confusion_matrix.csv
best_head.pt
train_features.pt
test_features.pt
```

`generate` 模式不会生成 feature cache；`lora` 模式还会生成 `best_lora_adapter.pt`。

运行还会保存 `timing.json`、`run_summary.png` 和 `run_summary.pdf`。多模型完成后生成论文汇总图和 CSV：

```bash
python -m experiments.qwen_vl_cifar10.visualize_results
```

## 6. 路径规则

- 不传 `--config` 时，默认路径仍然固定在本实验目录的 `data/` 和 `runs/qwen_vl_cifar10/`。
- JSON 中的 `data_root`、`output_dir` 相对于该 JSON 文件解析。
- 命令行显式传入的相对路径相对于当前 PowerShell 目录解析。
- 命令行参数优先级高于 JSON，例如 `--batch-size 4` 会覆盖配置中的值。
