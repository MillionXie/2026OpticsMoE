# Qwen3-VL-2B CIFAR-10 Vision Homogeneous Optical MoE

This experiment replaces the complete electronic Qwen3-VL-2B **vision Transformer stack** with the verified homogeneous optical MoE. It does not replace or execute the language decoder. CIFAR-10 images remain RGB and are processed by the native Qwen image processor and frozen vision patch embedding.

The default configs reuse the repository-level `opticalmoe_experiments/data` CIFAR download instead of creating another 170 MB copy inside the new experiment.

## Teacher

```text
RGB image
-> Qwen image processor
-> frozen Qwen vision patch embedding
-> complete electronic vision Transformer stack
-> valid pre-merger visual tokens [T,1024]
-> mean over T
-> LayerNorm(1024) + Linear(1024,10)
```

## Student

```text
RGB image
-> Qwen image processor
-> frozen Qwen vision patch embedding
-> Linear(1024,120) + LayerNorm(120) + Softplus
-> strict zero-row padding to [120,120]
-> homogeneous top-3 optical MoE (9 experts, 5 phase planes)
-> global phase + 20 cm propagation
-> full [480,480] square-law detector
-> AvgPool2d(4,4) -> [120,120]
-> non-affine LayerNorm + ReLU
-> read first T rows [T,120]
-> Linear(120,1024)
-> mean over T
-> the same LayerNorm(1024) + Linear(1024,10) head
```

There are no ten class-specific detector regions and no detector auxiliary loss. The detector is a latent feature plane. If the processor produces more than 120 pre-merger visual tokens for any image, execution stops and asks for a lower `processor_max_pixels`; tokens are never cropped, resized, pooled, or mixed across images.

The field has 120 rows, but `T` is measured rather than hard-coded. For square CIFAR-10 and the default 25,600-pixel budget, `T` is normally about 100, followed by 20 strict zero-padding rows.

## Training

The Qwen backbone remains frozen. Trainable components are the input/output adapters, top-k router, 45 expert phase masks, global phase mask, and student head. The student head starts from the teacher head checkpoint. The student never runs the teacher online; teacher stack outputs and logits are cached first.

```text
L = 1.0 L_hidden + 0.5 L_KD + 0.5 L_CE + 0.1 L_router_balance
```

`L_hidden` compares LayerNorm-normalized student and teacher token states. The default classification head has only 12,298 parameters for a 1024-dimensional Qwen vision hidden and 10 classes.

The configuration is grouped by experiment, dataset, Qwen runtime, batching, teacher cache, vision adapter, optical MoE, loss, optimizer, training, regularization and visualization. Terminal status refresh has two independent limits: `training.logging.interval_batches` and `training.logging.interval_seconds`; whichever is reached first prints the next update. Per-epoch data reduction uses a rotating class-balanced window rather than permanently throwing away samples.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the exact optical data flow and parameter boundaries. See [CONFIGURATION.md](CONFIGURATION.md) for every configuration group and command-line override.
