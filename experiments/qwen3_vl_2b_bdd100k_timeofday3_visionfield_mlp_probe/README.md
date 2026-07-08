# BDD100K TimeOfDay-3 Vision Optical Input Field Probe

This experiment evaluates how much task-relevant information is already encoded in the optical input field before physical propagation. It skips the optical propagation and language optical surrogate, flattens the trained vision optical input field into a 4096-dimensional vector, and trains a lightweight electronic readout.

该实验用于评估真实光路不理想时，光学输入前端编码特征本身的可分类性。它不是新的光学传播模型，而是 optical input feature 的电子读出上限 baseline。

## Actual extraction path

```text
RGB image
-> Qwen image processor
-> frozen Qwen vision patch embedding and position embedding
-> trained Linear(1024 -> 64)
-> trained LayerNorm(64)
-> Softplus
-> strict zero padding from [T_v,64] to [64,64]
-> flatten [4096]
```

The extractor does not execute the Qwen vision transformer blocks, any `OpticalConversion`, the vision merger, multimodal injection, the language decoder, the language optical surrogate, final RMSNorm, or answer-hidden extraction. `VisionOpticalStackSurrogate.forward()` is disabled in this project; extraction can only use `encode_groups_to_input_fields()`.

The source `input_adapter` and `adapter_norm` are loaded from `vision_optical_stack_best.pt` and frozen. The default trainable probe is `4096 -> 512 -> 3`, containing 2,099,203 parameters. The linear probe contains 12,291 parameters.

The source and probe must use identical `processor_min_pixels=16384`, `processor_max_pixels=16384`, `optical_dim=64`, and `optical_field_size=64`. More than 64 visual tokens is a hard error; no crop, pooling, truncation, or fallback resize is used.

## Outputs

- `features/train_vision_input_field.pt`
- `features/test_vision_input_field.pt`
- `metrics/feature_extraction_{train,test}.json`
- `metrics/probe_training_history.csv`
- `metrics/probe_inference.json`
- `metrics/probe_predictions.csv`
- `metrics/probe_vs_student_comparison.json`, when source metrics exist
- `figures/vision_input_fields/`
- `figures/probe_training_curves.png`
- `figures/probe_confusion_matrix.png`

`finetune_vision_input_adapter` is reserved for a future online-feature experiment. This fixed-feature baseline requires it to remain `false`.

