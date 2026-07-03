# Qwen3-VL-8B Vision-only Frozen Feature + MLP Classifier

This experiment uses only the Qwen3-VL visual feature path. It calls
`model.get_image_features()` and trains an MLP classifier on pooled visual features.

```text
image
 -> Qwen image_processor
 -> model.get_image_features()
 -> mean over merged visual tokens
 -> Linear(visual_dim, 1024) -> GELU -> Dropout -> Linear(num_classes)
```

The complete Qwen3-VL backbone is frozen. This baseline does not call the text tokenizer and does
not execute the language decoder.

The default baseline uses FP32 weights and eager attention (`dtype=float32`,
`attn_implementation=eager`) to avoid reduced-precision or fused/SDPA attention acceleration.
An 8B checkpoint in FP32 normally does not fit in a single 24 GB RTX 4090.

## Datasets

Supported datasets are CIFAR-10, CIFAR-100, STL-10, SVHN, Fashion-MNIST, and a custom
ImageFolder layout. The default configuration is `configs/cifar100.json`.

## Run

All stage-specific commands are in [RUN_COMMANDS.md](RUN_COMMANDS.md).

The checkpoint source may be overridden without editing a config:

```bash
python -m experiments.qwen3_vl_8b_vision_mlp \
  --config experiments/qwen3_vl_8b_vision_mlp/configs/cifar100.json \
  --model-id /path/to/local/Qwen3-VL-8B-Instruct \
  --local-files-only \
  --phase all
```

Local relative model paths are resolved relative to the config file. `data_root`, `output_dir`,
and `cache_dir` accept `~`, `$NAME`, and `${NAME}`. An unset referenced environment variable is
reported as an error.

Cache selection is portable across servers:

1. Omit `--cache-dir` to use the Hugging Face default.
2. Set `HF_HOME=/path/to/cache` and omit `--cache-dir`.
3. Pass `--cache-dir /path/to/cache` explicitly.

If `--cache-dir` points to an `HF_HOME` root containing a `hub/` subdirectory, the experiment
automatically selects that nested Hub cache.

No server-specific absolute cache path is stored in the default configs.

## Timing and outputs

Warmup batches are excluded. CUDA is synchronized around GPU stages. The benchmark records data
loading, image preprocessing, host-to-device transfer, visual forward, token pooling, MLP,
postprocessing, pipeline, and end-to-end latency.

Each run writes resolved config, environment, dataset/model metadata, frozen features, the best
MLP checkpoint, training history, predictions, metrics, per-batch timing, confusion matrix, and
PNG/PDF figures beneath its configured output directory.
