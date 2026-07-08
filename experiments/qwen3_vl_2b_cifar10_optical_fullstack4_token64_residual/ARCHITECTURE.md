# Architecture

## Preserved Qwen3-VL path

```text
CIFAR-10 RGB image + fixed 10-class prompt
  -> Qwen processor, tokenizer, and chat template
  -> frozen vision patch embedding
  -> VisionOpticalStackSurrogate
  -> frozen vision merger: 1024 -> 2048
  -> original multimodal token injection
  -> LanguageOpticalStackSurrogate
  -> frozen final RMSNorm
  -> answer-position hidden [B, 2048]
  -> trainable MLP: 2048 -> 1024 -> 10
```

The original vision and language Transformer layers are bypassed. Patch embedding, vision merger, token embeddings, multimodal injection, and final norm remain Qwen modules and are frozen.

## Vision surrogate

For each image boundary recovered from `cu_seqlens`:

```text
X_v [T_v, 1024]
  -> Linear(1024, 64)
  -> LayerNorm(64)
  -> Softplus
  -> direct row write and zero padding [64, 64]
  -> OpticalConversion x 4
  -> first T_v rows [T_v, 64]
  -> Linear(64, 1024) = Delta_v
  -> beta_v * X_v + alpha_v * Delta_v
  -> Y_v [T_v, 1024]
```

`T_v` must not exceed 64. Batch size affects only the number of independently processed fields, never the contents of one image's field.

## Language surrogate

The attention mask determines each sample's valid sequence length:

```text
X_l_i [S_i, 2048]
  -> Linear(2048, 64)
  -> LayerNorm(64)
  -> Softplus
  -> direct row write and zero padding [64, 64]
  -> OpticalConversion x 4
  -> first S_i rows [S_i, 64]
  -> Linear(64, 2048) = Delta_l
  -> beta_l * X_l_i + alpha_l * Delta_l
  -> Y_l_i [S_i, 2048]
```

`S_i` must not exceed 64. Padding positions remain zero.

## Distillation

The electronic teacher cache stores only stack-level outputs required for training: `teacher_vision_stack_output` and `teacher_answer_hidden`, plus labels, indices, token counts, and grids. After teacher MLP training, `teacher_logits` are generated from cached answer features without rerunning Qwen.

Student loss is:

```text
L = 0.4 * L_vision + 0.4 * L_answer + 1.0 * L_KD + 1.0 * L_CE
```

- `L_vision`: normalized MSE between student and teacher full vision-stack outputs.
- `L_answer`: normalized MSE between answer-position hidden features.
- `L_KD`: temperature-scaled KL divergence between teacher and student logits.
- `L_CE`: hard-label CIFAR-10 cross entropy.

## Optical conversion

Each stack contains four conversions on a 64-by-64 active field with 128-by-128 padded angular-spectrum propagation. A conversion applies propagation, trainable phase modulation, optional amplitude modulation, square-law detection, and mean-intensity normalization. Detected intensity continues to the next conversion; the training path does not take `sqrt(intensity)`.

The default pixel pitch is 8 micrometres, wavelength is 532 nm, phase initialization is zeros, and amplitude masks are disabled.

## Classification head ablation

`build_head()` supports the original MLP, a direct linear classifier, a configurable bottleneck MLP with optional input LayerNorm, and LayerNorm plus linear classification. Teacher and student always use the same resolved head configuration. Checkpoints record the resolved type and dimensions; incompatible head checkpoints fail explicitly. The model report separates head parameters from optical phases, optional amplitude masks, electronic adapters, and residual scales.

The debug path is read-only and runs under `torch.no_grad()`. It records `Linear -> LayerNorm -> Softplus` input fields, detector intensity after every conversion, output-adapter delta, residual-combined vision hidden, language sequence hidden, and answer hidden against teacher cache targets. Raw intensity must be nonnegative; any negative detector count triggers a warning. Hidden/delta/signed-difference plots may contain negative values and use centered diverging colormaps.
