# Qwen3-VL-8B Full Multimodal Frozen Feature + MLP Classifier

This experiment uses the full Qwen3-VL vision-language forward path. Each image is paired with a
fixed classification prompt. The model runs the original multimodal forward pass, and the hidden
state at the answer position is fed into the same MLP classifier.

```text
image + classification prompt
 -> Qwen3-VL chat template and original processor
 -> image tokens + prompt tokens
 -> full frozen Qwen3-VL vision-language forward
 -> final-layer hidden state at the answer position [4096]
 -> Linear(4096, 1024) -> GELU -> Dropout -> Linear(num_classes)
```

The backbone is frozen with `model.requires_grad_(False)` and remains in evaluation mode. Only the
MLP parameters are trained. This implementation does not manually concatenate visual and textual
features.

The default CIFAR-100 prompt is configured as:

```text
Classify this image into one of the CIFAR-100 categories. Answer:
```

It can be changed with the config field `classification_prompt` for another dataset.

## Datasets and run commands

Supported datasets are CIFAR-10, CIFAR-100, STL-10, SVHN, Fashion-MNIST, and custom ImageFolder.
All stage-specific commands are in [RUN_COMMANDS.md](RUN_COMMANDS.md).

The checkpoint source may be overridden without editing a config:

```bash
python -m experiments.qwen3_vl_8b_multimodal_mlp \
  --config experiments/qwen3_vl_8b_multimodal_mlp/configs/cifar100.json \
  --model-id /path/to/local/Qwen3-VL-8B-Instruct \
  --local-files-only \
  --phase all
```

Local relative model paths are resolved relative to the config file. `data_root`, `output_dir`,
and `cache_dir` accept `~`, `$NAME`, and `${NAME}`. An unset referenced environment variable is
reported as an error.

Cache selection is portable across servers: use the Hugging Face default, set `HF_HOME`, or pass
`--cache-dir`. If the explicit path is an `HF_HOME` root containing `hub/`, that nested Hub cache
is selected automatically. No server-specific absolute cache path is stored in the configs.

## Timing and outputs

Warmup batches are excluded. CUDA is synchronized around GPU stages. The benchmark records:

- data loading;
- multimodal preprocessing, including image processing, tokenizer, and input construction;
- host-to-device transfer;
- the complete Qwen3-VL multimodal forward;
- answer-position hidden-state selection;
- MLP forward, postprocessing, pipeline, and end-to-end latency.

Each run writes resolved config, environment, dataset/model metadata, frozen features, the best
MLP checkpoint, training history, predictions, metrics, per-batch timing, confusion matrix, and
PNG/PDF figures beneath its configured output directory.
