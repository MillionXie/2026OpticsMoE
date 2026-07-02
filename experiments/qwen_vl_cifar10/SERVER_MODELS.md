# Qwen3-VL model selection on the lab server

## Reproducible precision policy

The comparison configs use BF16 weights and the same `visual_tokens_mean` feature path. This
keeps model-size comparisons interpretable. Quantized checkpoints are deliberately excluded:
quantization changes both accuracy and runtime and must be reported as a separate experimental
factor.

The official [Qwen3-VL collection](https://huggingface.co/collections/Qwen/qwen3-vl) contains
2B, 4B, 8B, 30B-A3B MoE, 32B dense, and 235B-A22B MoE checkpoints. Hugging Face reports the 8B
checkpoint as about 9B parameters, so BF16 weights alone require roughly 18 GB. It is therefore
the largest defensible BF16 Qwen3-VL model for one 24 GB RTX 4090 after allowing memory for the
vision encoder activations and CUDA workspaces.

`Qwen3-VL-30B-A3B` only activates about 3B language parameters per token, but all approximately
31B parameters must still reside in memory. Its BF16 weights require roughly 62 GB. The 32B dense
model similarly requires roughly 66 GB. Both are multi-GPU jobs, not single-4090 jobs.

The often-mentioned 72B vision-language checkpoint is
[`Qwen2.5-VL-72B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct), a different model
generation with about 73B parameters. Its BF16 weights alone require roughly 146 GB, so it fits
neither one 24 GB 4090 nor four 24 GB 4090s without quantization or CPU offload. It is not included
in this Qwen3-VL comparison.

## Configs

| Config | Hardware | Purpose |
| --- | --- | --- |
| `mlp_4b_server_1x4090.json` | 1 x RTX 4090 | Intermediate scaling point |
| `mlp_8b_server_1x4090.json` | 1 x RTX 4090 | Largest recommended single-4090 BF16 run |
| `mlp_30b_a3b_server_4x4090.json` | 4 x RTX 4090 | Multi-GPU MoE scaling point |
| `mlp_32b_server_4x4090.json` | 4 x RTX 4090 | Largest dense BF16 run expected to fit |

The 30B/32B estimates leave much less headroom than the 8B run. Run their smoke tests before the
full dataset and ensure all selected GPUs are idle.

## Commands

Single 4090 (replace `3` with an idle physical index from `nvidia-smi`):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python experiments/qwen_vl_cifar10/main.py \
  --config experiments/qwen_vl_cifar10/configs/mlp_8b_server_1x4090.json \
  --smoke-test
```

Remove `--smoke-test` after the 32/32-sample run succeeds.

Four 4090s (replace the list with four idle 4090 indices):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,3,4 \
python experiments/qwen_vl_cifar10/main.py \
  --config experiments/qwen_vl_cifar10/configs/mlp_30b_a3b_server_4x4090.json \
  --smoke-test
```

The multi-GPU configs set `device_map=auto`; the single-GPU configs keep `device_map=none`.

## Timing protocol and outputs

Every run now writes:

- `metrics.json`: accuracy, macro-F1, throughput, memory, runtime metadata, and epoch history;
- `timing.json`: UTC start/end timestamps, monotonic stage durations, benchmark protocol, and
  explicit inclusion/exclusion rules;
- `run_summary.png` and `run_summary.pdf`: loss, accuracy, per-class accuracy, and stage time;
- feature caches and prediction/checkpoint artifacts as before.

GPU stages call `torch.cuda.synchronize()` around measured regions. Model loading, feature
extraction, head/adapter training, evaluation, dataset loading, cold latency, steady-state latency,
and total wall time are kept as separate quantities. Do not report feature-cache-hit timing as
fresh feature-extraction timing.

Generate the cross-model paper figure and CSV after multiple runs:

```bash
python -m experiments.qwen_vl_cifar10.visualize_results
```

Outputs are written to `experiments/qwen_vl_cifar10/runs/summary/` as PNG, PDF, and CSV.
