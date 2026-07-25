# Configuration notes

The primary configuration is `configs/spaq_mos.json`; the remaining task and
smoke files inherit it through `base_config`.

## Fixed architecture fields

```json
{
  "vision_adapter": {
    "optical_channels": 224,
    "max_visual_tokens": 224,
    "tap_stages": [1, 2, 3]
  },
  "language_adapter": {
    "max_language_tokens": 224
  },
  "moe": {
    "geometry": {
      "canvas_size": 1026,
      "active_size": 986,
      "expert_size": 224,
      "expert_pitch": 254,
      "num_experts": 16,
      "grid_rows": 4,
      "grid_cols": 4,
      "layers_per_expert": 4
    },
    "router": {
      "top_k": 4
    },
    "final_detector_readout": {
      "pool_type": "adaptive_avg",
      "output_size": 224,
      "crop_to_active": true
    }
  }
}
```

The final detector first crops the active ROI from 1026×1026 to 986×986 and
only then applies adaptive pooling. This is an electronic detector readout,
not a field interpolation used inside the optical path.

## Attention and residual

All supplied configs set:

```json
{
  "native_pre_attention_enabled": false,
  "initialize_attention_from_teacher": false,
  "native_pre_attention_trainable": false,
  "residual_enabled": true
}
```

The validator rejects attention-enabled configurations for this experiment.

## Shared cache

```json
{
  "teacher_cache": {
    "precompute_cache_dir": "../cache"
  }
}
```

The path is resolved relative to `configs/`, so it points at the new
experiment's `cache/` directory. Do not manually put it under a run. Separate
cache identities are generated for different tasks, split limits, prompts and
processor settings.

## Batch size

The formal config uses `student_batch_size=1` and
`inference_batch_size=1`. A 1026×1026 complex FFT canvas is much larger than
the previous 480×480 canvas, and both vision and language cores retain four
stage activations for backpropagation. Increase the batch only after measuring
GPU memory. `feature_batch_size=4` is independent and applies to frozen Qwen
precompute.

## LayerNorm placement

- Input adapter: `Linear -> LayerNorm -> Softplus`.
- OEO stages: `square detection -> per-expert non-affine LayerNorm -> ReLU`.
- Final CCD: `crop -> pool -> per-token non-affine LayerNorm -> ReLU`.
- Residual: fixed identity addition after optical hidden restore.

OEO `elementwise_affine=false` is deliberate. A separate affine tensor for
every pixel, expert and stage would add millions of electronic parameters and
obscure the optical parameter budget.
