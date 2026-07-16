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
-> after every OEO stage: independent LN + activation for selected experts
-> restore each selected expert's router weight as a field-amplitude coefficient
-> strict zero for unselected experts, then zero-phase optical reload
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

The prompt still applies the sparse top-3 router weights before the first propagation, matching the all-optical routing input. Per-expert LayerNorm would otherwise erase those relative amplitude scales, so `reapply_routing_weights=true` restores the same weights after every interlayer normalization and activation. This is amplitude weighting (`field *= weight`), not square-root power weighting. The following LayerNorm prevents the coefficient from simply accumulating as `weight**5`.

Before the router's trainable linear gate, its pooled 10x10 input is normalized with a non-affine LayerNorm. This prevents a shared positive DC level in the Softplus optical input fields from overwhelming the smaller image-dependent routing features. No random routing jitter is used.

The router gate uses its own Adam/AdamW parameter group with `learning_rate=0.001`; the optical surrogate, adapters, and classification head keep the main `0.008` learning rate. This prevents the small `Linear(100,9)` softmax gate from saturating much faster than the optical system. The balance coefficient is `0.2`: in the observed collapsed run, raw `L_balance` was approximately 3.0, so the weighted term is about 0.6 versus about 3.04 for hidden/KD/CE supervision.

The field has 120 rows, but `T` is measured rather than hard-coded. For square CIFAR-10 and the default 25,600-pixel budget, `T` is normally about 100, followed by 20 strict zero-padding rows.

## Training

The Qwen backbone remains frozen. Trainable components are the input/output adapters, top-k router, 45 expert phase masks, global phase mask, and student head. The student head starts from the teacher head checkpoint. The student never runs the teacher online; teacher stack outputs and logits are cached first.

```text
L = 1.0 L_hidden + 0.5 L_KD + 0.5 L_CE + 0.2 L_router_balance
```

`L_hidden` compares LayerNorm-normalized student and teacher token states. The default classification head has only 12,298 parameters for a 1024-dimensional Qwen vision hidden and 10 classes.

The configuration is grouped by experiment, dataset, Qwen runtime, batching, teacher cache, vision adapter, optical MoE, loss, optimizer, training, regularization and visualization. Terminal status is refreshed only every `training.logging.interval_batches` batches. Each update reports all nine experts' cumulative selection rates and their mean weights when selected. Per-epoch data reduction uses a rotating class-balanced window rather than permanently throwing away samples.

The student uses the complete retained CIFAR-10 training split. To match the legacy homogeneous optical-MoE experiment, the test split is evaluated after every epoch and selects `best`. This is convenient for direct experimental comparison, but `best_test` is selection-biased because the test split is no longer a strictly held-out final evaluation. Every visualization interval also saves random per-sample optical and hidden-state diagnostics under `figures/debug_examples/epoch_XXXX/`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the exact optical data flow and parameter boundaries. See [CONFIGURATION.md](CONFIGURATION.md) for every configuration group and command-line override.

