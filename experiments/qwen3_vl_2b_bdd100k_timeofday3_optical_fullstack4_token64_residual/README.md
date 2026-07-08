# Qwen3-VL-2B TimeOfDay-3 token64 optical4 residual

This is an independent BDD100K TimeOfDay-3 experiment copied from the full-stack optical4 baseline. It keeps the Qwen processor, tokenizer, patch embedding, vision merger, multimodal injection, final language norm, answer-position extraction, and MLP classifier. The original vision and language Transformer stacks are each replaced by one four-conversion optical surrogate.

## Token64 adapter

The Qwen processor pixel budget is fixed to 16,384 pixels by default to reduce the pre-merge visual token count to approximately 64. Actual token counts still depend on image aspect ratio and Qwen grid alignment, so both adapters enforce a strict upper bound.

The optical adapter no longer interpolates token-channel matrices. Each visual token is projected from 1,024 hidden features to 64 non-negative optical channels. The resulting `[T_v, 64]` tensor is written directly into the first `T_v` rows of a zero-initialized `[64, 64]` optical field. After four optical conversions, only those valid rows are read and projected back to 1,024. The frozen Qwen vision merger then performs the original 1,024-to-2,048 mapping.

Language tokens follow the same mapping: `[S, 2048] -> [S, 64] -> zero-padded [64, 64] -> optical4 -> [S, 64] -> [S, 2048]`. Padding positions remain zero.

If `T_v > 64`, execution stops and asks for a lower `processor_max_pixels`. If `S > 64`, execution stops and asks for a shorter prompt or a lower visual pixel budget. There is no crop, truncation, pooling, multi-field fallback, or resize fallback.

## Residual branch

Each surrogate has independent scales:

```text
vision:   Y_v = beta_v * X_v + alpha_v * Delta_v
language: Y_l = beta_l * X_l + alpha_l * Delta_l
```

By default, `beta_v` and `beta_l` start at 1.0 and are fixed buffers. `alpha_v` and `alpha_l` start at 0.1 and are trainable parameters. Disabling `optical_residual_enabled` returns only `Delta`.

The four values are written to `model.json`, every row of `metrics/student_training_history.csv`, `metrics/student_training_latest.json`, `metrics/best_validation.json`, `metrics/student_training.json`, final `metrics/student_inference.json`, and checkpoint metadata sidecars. The scale tensors themselves are included in each vision/language surrogate state dict, including fixed buffers.

Teacher caches are stored under this experiment's independent output directory. Cache metadata includes both processor pixel budgets and both hidden sizes; incompatible caches cause a hard error.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete shape flow and [RUN_COMMANDS.md](RUN_COMMANDS.md) for commands.

## Configurable classification head

The head supports `mlp`, `linear`, `bottleneck`, and `normalized_linear`. The TimeOfDay-3 main configuration now uses `LayerNorm(2048) -> Linear(2048,64) -> GELU -> Dropout -> Linear(64,3)`, reducing the head from about 2.1 million parameters to about 135 thousand. Dedicated bottleneck-64 and linear configs provide isolated ablations. Head metadata and separate optical/electronic parameter counts are written to `model.json` and checkpoints.
