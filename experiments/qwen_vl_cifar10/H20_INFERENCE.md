# Qwen3-VL-32B CIFAR-10 inference on one H20

## Capacity conclusion

A full NVIDIA H20 exposes 96 GB of HBM3. Qwen3-VL-32B is listed as approximately 33B parameters;
BF16 weights therefore occupy roughly 61.5 GiB before CUDA workspaces, activations, and KV cache.
Batch-size-1, short-output inference should fit on a full 96 GB H20.

This conclusion does **not** apply to a 48/24/16 GB vGPU or MIG slice. The benchmark checks visible
memory before downloading/loading the model and requires at least 80 GiB by default.

Check the actual allocation:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
```

## Timing definitions

The inference benchmark uses mutually attributable stages:

| Metric | Boundary |
| --- | --- |
| `dataset_fetch_sec` | `DataLoader.next()` and collation only |
| `prompt_build_sec` | Construct per-image CIFAR-10 chat messages |
| `image_preprocess_sec` | Actual Qwen image-processor calls |
| `tokenizer_sec` | Actual tokenizer calls |
| `processor_framework_sec` | Chat template, visual placeholders, processor overhead |
| `processor_total_sec` | Whole `apply_chat_template(..., tokenize=True)` call |
| `host_to_device_sec` | Tensor transfer to the model input GPU |
| `model_generate_sec` | `model.generate()` only; tokenizer and decode excluded |
| `decode_postprocess_sec` | New-token slicing, GPU-to-CPU conversion, text decode |
| `complete_inference_sec` | Prompt construction through decoded class string |
| `end_to_end_sec` | Dataset fetch through decoded class string |

All GPU stages are bounded by synchronization of every visible CUDA device. Warm-up batches,
model loading, feature-shape probing, result serialization, and plotting are reported separately
and excluded from the measured latency distribution.

The shape probe additionally reports one pure vision-encoder forward and one multimodal prefill
forward (`use_cache=False`). Those probe values describe operator-level forward latency; the main
`model_generate_sec` metric describes the actual autoregressive zero-shot classification workload.

Use `num_workers=0` for the paper's primary timing table. Multiple workers prefetch data, so
`DataLoader.next()` no longer represents complete sample-loading work.

## Run

Install the updated requirements, then run the 32-image smoke benchmark:

```bash
python -m pip install -r experiments/qwen_vl_cifar10/requirements.txt

CUDA_VISIBLE_DEVICES=0 python experiments/qwen_vl_cifar10/inference_benchmark.py \
  --config experiments/qwen_vl_cifar10/configs/inference_32b_h20_smoke.json
```

If it succeeds, run all 10,000 CIFAR-10 test images:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/qwen_vl_cifar10/inference_benchmark.py \
  --config experiments/qwen_vl_cifar10/configs/inference_32b_h20.json
```

The benchmark is zero-shot generation: Qwen receives each image and must output exactly one
CIFAR-10 class name. No classifier head is trained.

## Outputs

- `inference_metrics.json`: accuracy, timing distributions, runtime, parameters, and peak memory;
- `batch_timings.csv`: raw per-batch values for statistical analysis;
- `feature_shapes.json`: input, vision Transformer, merger, language Transformer, DeepStack, and
  pooled-feature shapes with logical byte sizes;
- `predictions.csv` and `confusion_matrix.csv`;
- `inference_summary.png` and `.pdf`: timing, latency trace, per-class accuracy, and layer widths.

The shape probe records the actual checkpoint rather than assuming a width. For Qwen3-VL-32B,
the expected distinction is a 1152-wide vision Transformer followed by a 5120-wide visual merger
and language model, but `feature_shapes.json` is the authoritative record for the installed model.
