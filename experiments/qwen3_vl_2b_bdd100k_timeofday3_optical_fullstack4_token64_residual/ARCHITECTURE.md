# Architecture

## Preserved Qwen3-VL path

```text
image + fixed TimeOfDay-3 prompt
  -> Qwen processor / tokenizer / chat template
  -> frozen vision patch embedding
  -> VisionOpticalStackSurrogate
  -> frozen Qwen vision merger (1024 -> 2048)
  -> original multimodal token injection
  -> LanguageOpticalStackSurrogate
  -> frozen final RMSNorm
  -> answer-position hidden [B, 2048]
  -> trainable MLP classifier
  -> logits [B, 3]
```

The remaining original vision and language blocks are bypass modules. No original Transformer block is restored.

## Vision surrogate

For each packed-image boundary obtained from `cu_seqlens`:

```text
X_v [T_v, 1024]
  -> Linear(1024, 64)
  -> LayerNorm(64)
  -> Softplus
  -> direct row write + zero padding [64, 64]
  -> OpticalConversion x 4
  -> first T_v rows [T_v, 64]
  -> Linear(64, 1024) = Delta_v
  -> beta_v * X_v + alpha_v * Delta_v
  -> Y_v [T_v, 1024]
```

`T_v` must be at most 64. A larger value raises an error that reports the measured token count and requests a lower `processor_max_pixels`.

## Language surrogate

The original 2D attention mask identifies valid tokens independently for every sample:

```text
X_l_i [S_i, 2048]
  -> Linear(2048, 64)
  -> LayerNorm(64)
  -> Softplus
  -> direct row write + zero padding [64, 64]
  -> OpticalConversion x 4
  -> first S_i rows [S_i, 64]
  -> Linear(64, 2048) = Delta_l
  -> beta_l * X_l_i + alpha_l * Delta_l
  -> Y_l_i [S_i, 2048]
```

`S_i` must be at most 64. Larger sequences raise an error; the implementation never truncates the prompt or resizes token matrices.

## Optical conversion

The existing conversion remains unchanged: padded angular-spectrum propagation on a 128-by-128 grid, trainable phase mask, optional amplitude mask, square-law detection, and mean intensity normalization. The active field is 64-by-64 and the default pixel pitch is 8 micrometres.

## Residual controls

The configuration keys are in `settings.py` and every JSON config:

```text
optical_residual_enabled
optical_identity_scale_init
optical_modulated_scale_init
optical_identity_scale_trainable
optical_modulated_scale_trainable
```

Vision and language modules own separate scale tensors. Non-trainable scales are registered buffers, so they move with the model and remain in checkpoints.

## Classification head

Teacher and student heads are built from the same config. Available structures are MLP, linear, bottleneck MLP with optional LayerNorm, and normalized linear. The default three-class head is bottleneck-64 with LayerNorm. Old configs without the new keys still resolve to the original `2048 -> hidden_dim -> 3` MLP.
