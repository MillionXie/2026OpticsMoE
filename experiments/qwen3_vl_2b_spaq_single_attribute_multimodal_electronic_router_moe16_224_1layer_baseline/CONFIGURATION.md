# Configuration guide

The formal base config is `configs/spaq_mos.json`. Brightness, Colorfulness, and
Contrast configs inherit it and change only task name, prompt, and output path.

## Baseline invariants

| Setting | Value | Meaning |
|---|---:|---|
| `layers_per_expert` | `1` | one 16-expert phase plane |
| global phase | retained | second phase plane in each stack |
| `tap_stages` | `[1]` | one pre-global auxiliary vision tap |
| `top_k` | `4` | four selected experts per sample |
| native attention | `false` | no trainable attention prelude |
| residual | `true` | fixed `X + optical_delta` |
| SAM | `false` | one forward/backward pass per batch |
| weight decay | `0` | both electronic and phase parameters |
| phase dropout | `false` | masks are never bypassed/dropped |
| ranking loss | `0` | disabled |
| Norm-in-Norm loss | `0` | disabled |
| historical initialization | `null` | fresh student |

These are validated in code; incompatible settings fail before model loading.

## Formal batching

```json
{
  "feature_batch_size": 4,
  "student_batch_size": 8,
  "inference_batch_size": 8,
  "head_batch_size": 512,
  "num_workers": 8,
  "cpu_threads": 4,
  "cpu_interop_threads": 1
}
```

Teacher/processor precompute uses `num_workers=8`. Student cached-data loading is
kept at `num_workers=0` to avoid duplicating shard caches across worker
processes. Batch size 8 is the effective formal GPU batch; there is no gradient
accumulation.

Smoke configs deliberately use student/inference batch 1 and tiny datasets so
that interface tests fit smaller GPUs.

## Optical geometry

```json
{
  "canvas_size": 1026,
  "active_size": 986,
  "num_experts": 16,
  "expert_size": 224,
  "expert_pitch": 254,
  "grid_rows": 4,
  "grid_cols": 4,
  "layers_per_expert": 1
}
```

The expert gap is `254 - 224 = 30` pixels. The 986-pixel active footprint is
zero-padded by 20 pixels per side to the propagation canvas. Global phase and
CCD effective ROI both align to the full 986 × 986 footprint.

## Physical settings

```json
{
  "wavelength_nm": 532.0,
  "pixel_pitch_um": 8.0,
  "inter_layer": 0.1,
  "last_expert_to_global": 0.1,
  "global_to_detector": 0.1,
  "phase_parameterization": "sigmoid",
  "phase_init": "zeros",
  "k_space_enabled": false
}
```

In the one-stage model, expert phase propagation reaches the OEO reload/global
plane after 10 cm; the global phase then propagates 10 cm to the final CCD.
All phase masks initialize from raw phase zero and use the configured sigmoid
phase parameterization.

## Optimization

```json
{
  "type": "adamw",
  "learning_rate": 0.008,
  "router_learning_rate": 0.001,
  "student_head_learning_rate": 0.001,
  "weight_decay": 0.0,
  "phase_weight_decay": 0.0,
  "sam": {
    "enabled": false
  },
  "scheduler": "cosine"
}
```

AdamW with zero decay is retained to preserve the optimizer interface used by
the source experiment. The formal run uses 100 epochs. Phase dropout is also
off, so every batch updates the full selected phase masks.

## Loss weights

```json
{
  "vision_hidden_weight": 1.0,
  "answer_hidden_weight": 1.0,
  "prediction_distill_weight": 0.5,
  "regression_weight": 1.0,
  "router_balance_weight": 0.03,
  "router_importance_weight": 0.0
}
```

Ranking and Norm-in-Norm code remains available in the copied trainer for
compatibility, but this baseline fixes their weights to zero.

## Cache layout

Frozen teacher outputs and processor tensors are stored under:

```text
experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/cache/
```

The directory is ignored by Git. Cache identity includes dataset split, task,
prompt, model, processor pixel budget, and dtype. Optical layer count is
deliberately excluded because these are frozen-teacher inputs/targets.

Each task has a different prompt and therefore a different cache identity. An
existing compatible cache may be copied into this cache root; metadata
validation rejects incompatible data rather than silently reusing it.

## Formal configs

- `spaq_mos.json`
- `spaq_brightness.json`
- `spaq_colorfulness.json`
- `spaq_contrast.json`

Diagnostics:

- `spaq_mos_vision_electronic_language.json`: Vision optical, Language frozen
  electronic.

Smoke:

- `spaq_mos_smoke.json`
- `spaq_brightness_smoke.json`
- `spaq_colorfulness_smoke.json`
- `spaq_contrast_smoke.json`
