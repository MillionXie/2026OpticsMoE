# Qwen3-VL-2B KADID-10k Quality-3 token64 optical4 residual

This independent experiment performs three-class image quality assessment on KADID-10k:

```text
0 high_quality
1 medium_quality
2 low_quality
```

It keeps the frozen Qwen3-VL processor, tokenizer, vision patch embedding, vision merger, multimodal injection, final RMSNorm, and answer-position classifier. One token64 optical4 surrogate replaces the complete vision Transformer stack and another replaces the complete language stack.

## Labels

The default `score_tertile` mode divides MOS or DMOS scores at the one-third and two-third quantiles. DMOS is interpreted as lower-is-better; MOS is interpreted as higher-is-better. An ambiguous column such as `score` requires an explicit `quality_score_higher_is_better` value.

The optional `distortion_level_3class` mode maps levels 1-2 to high quality, level 3 to medium quality, and levels 4-5 to low quality.

## Reference-disjoint split

Rows are first grouped by reference-image identity. Twenty percent of references are assigned to test. Validation references are then selected from the remaining training references. The train, validation, and test reference sets are mutually disjoint, preventing distorted variants of the same source image from leaking across evaluation boundaries. Reference IDs are also cached so teacher-MLP validation uses the same policy.

## Token64 optical adapter

The processor pixel budget defaults to 16,384 pixels. For each sample:

```text
hidden -> Linear(64) -> LayerNorm(64) -> Softplus
       -> direct row write and zero padding to [64,64]
       -> OpticalConversion x 4
       -> valid rows only -> Linear(original hidden size)
       -> beta * input + alpha * optical delta
```

Vision returns 1,024 features to the frozen Qwen vision merger. Language returns 2,048 features to the frozen final RMSNorm. There is no token-field interpolation. A visual or language sequence longer than 64 raises an error; no crop, pooling, truncation, or resize fallback is used.

The short prompt is:

```text
Rate image quality: high_quality, medium_quality, or low_quality. Answer:
```

`alpha_v`, `beta_v`, `alpha_l`, and `beta_l` are recorded in model, epoch, checkpoint-metadata, and inference reports.

`prepare_data` now downloads KADID-10k automatically from the OSF alternative linked by the official database page. The approximately 3.1 GB archive is stored under `data_root/_downloads`, safely extracted under `data_root/_raw`, and its `dmos.csv` plus `image`/`images` directory are located automatically. Existing manually prepared data is detected first and is never downloaded again. Set `download=false` only when intentionally using a manually configured dataset.
