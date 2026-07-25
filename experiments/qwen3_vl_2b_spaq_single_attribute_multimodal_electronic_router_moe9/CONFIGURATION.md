# Configuration notes

The complete base configuration is `configs/spaq_mos.json`. Brightness, Colorfulness, and Contrast inherit it and change only the task, prompt, experiment name, and output directory.

Important switches:

```json
"student": {
  "language_stack_mode": "optical_moe",
  "transformer_block_alignment": {
    "native_pre_attention_enabled": false,
    "initialize_attention_from_teacher": false,
    "native_pre_attention_trainable": false,
    "residual_enabled": true,
    "vision_attention_source_layer": 0,
    "language_attention_source_layer": 0
  }
}
```

- `native_pre_attention_enabled=false`: bypass both vision and language attention preludes. This is the default for every inherited SPAQ task config.
- `initialize_attention_from_teacher` and `native_pre_attention_trainable` only matter when the prelude is explicitly re-enabled.
- `residual_enabled=true`: fixed Transformer-style identity residual with coefficient 1.

Trainable attention uses its own optimizer group:

```json
"optimizer": {
  "learning_rate": 0.008,
  "attention_learning_rate": 0.0001,
  "student_head_learning_rate": 0.001
}
```

The lower default avoids applying the comparatively aggressive optical-mask learning rate to newly initialized Qwen attention projections.

Physical router:

```json
"router": {
  "implementation": "electronic_amplitude_topk",
  "amplitude_slm": {
    "weight_domain": "amplitude",
    "input_normalization": "none",
    "relay": "ideal_4f_identity"
  }
}
```

`weight_domain=amplitude` places `w_i A`; `weight_domain=power` places `sqrt(w_i) A`. No phase prompt is generated in either mode.

All propagation distances are in metres:

```json
"distances_m": {
  "inter_layer": 0.1,
  "last_expert_to_global": 0.1,
  "global_to_detector": 0.1
}
```

Keep `final_detector_readout.layernorm_scope="per_token"` unless deliberately reproducing the old full-field ablation.

## CPU workers, thread pools, and cache residency

```json
"batching": {
  "num_workers": 8,
  "cpu_threads": 4,
  "cpu_interop_threads": 1
},
"teacher_cache": {
  "shard_size": 128,
  "lru_shards": 128
}
```

- `num_workers` is used while decoding source images for teacher and processor
  precomputation. Student training consumes already-preprocessed cache tensors
  and deliberately uses `student_cache_workers=0`; multiprocessing workers
  would each duplicate the multi-gigabyte cache LRU.
- `cpu_threads` and `cpu_interop_threads` bound PyTorch's process-local CPU
  pools. They do not change the model, batches, sampling order, or losses.
- Cached image patches and teacher targets remain fp16 on the CPU and are
  promoted only after transfer to the GPU. Qwen already casts image patches to
  its visual dtype on-device.
- The multimodal forward requests `logits_to_keep=1`. The final hidden-state
  hook is unchanged, while Qwen no longer projects every prompt position to
  the full language vocabulary when those logits are not used.
- `lru_shards=128` is large enough for the current 10,013-image SPAQ caches.
  It uses roughly the combined cache footprint (about 9.5 GB for MOS) but
  avoids deserializing evicted shards again in later epochs. Reduce this value
  on RAM-constrained hosts; doing so affects performance, not model results.
- Epoch metrics record cache hit rates, shard loads, resident shard counts,
  and separate train/test wall times.
