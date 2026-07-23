# Configuration notes

The complete base configuration is `configs/spaq_mos.json`. Brightness, Colorfulness, and Contrast inherit it and change only the task, prompt, experiment name, and output directory.

Important switches:

```json
"student": {
  "language_stack_mode": "optical_moe",
  "transformer_block_alignment": {
    "native_pre_attention_enabled": true,
    "initialize_attention_from_teacher": false,
    "native_pre_attention_trainable": true,
    "residual_enabled": true,
    "vision_attention_source_layer": 0,
    "language_attention_source_layer": 0
  }
}
```

- `initialize_attention_from_teacher=false`: copy architecture, independently initialize attention projections.
- `initialize_attention_from_teacher=true`: initialize from the selected original Qwen block.
- `native_pre_attention_trainable=false`: freeze the prelude.
- `residual_enabled=true`: fixed Transformer-style identity residual with coefficient 1.

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
