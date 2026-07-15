# Architecture

## Scope

- Dataset: CIFAR-10, native RGB input.
- Backbone: Qwen3-VL-2B vision stem only.
- Replaced: all Qwen vision Transformer blocks.
- Not used: tokenizer, text prompt, multimodal injection, language decoder, final language RMSNorm.
- Classification feature: mean of valid pre-merger vision token states.

## Token boundary

The vision block input is packed as `[sum(T_i),1024]`. Qwen's `cu_seqlens` is mandatory and splits it into one tensor per image. Each image is independently mapped to one optical field:

```text
[T_i,1024]
-> Linear(1024,120)
-> LayerNorm(120)
-> Softplus
-> [T_i,120]
-> zero rows only
-> [120,120]
```

`T_i > 120` is an error. Batch size never changes an individual image's field.

With the default 25,600-pixel budget and square CIFAR-10 images, the current Qwen3-VL processor normally produces a 160x160 processor image and about 10x10 = 100 pre-merger token rows. Those 100 rows occupy the first 100 rows of the 120x120 field; the remaining 20 rows are exactly zero. The code still uses the runtime `cu_seqlens` rather than assuming `T=100`.

## Homogeneous MoE

The 120x120 field is centered on a 480x480 canvas. The prompt uses one uniform routing amplitude per 150x150 cell and one continuous global quadratic phase. A learned router selects top-3 of nine 120x120 experts.

Each of five stages performs:

```text
9 independent phase-only 120x120 masks
-> 20 cm angular-spectrum propagation on the 480x480 canvas
-> square-law detection
-> per-expert non-affine LayerNorm
-> ReLU
-> zero-phase amplitude reload
```

The fifth stage is followed by the trainable 450x450 global phase and 20 cm propagation to the final detector.

## Full-plane detector readout

The final detector has no class boxes:

```text
complex detector field [B,480,480]
-> intensity = |E|^2
-> AvgPool2d(kernel=4,stride=4)
-> [B,120,120]
-> LayerNorm([120,120], elementwise_affine=False)
-> ReLU
-> first T_i rows [T_i,120]
-> Linear(120,1024)
-> student token hidden [T_i,1024]
```

The electronic output adapter restores the vision hidden size, not the language hidden size. There is no residual electronic bypass.

## Cached teacher targets

`teacher_precompute` runs the complete electronic Qwen vision stack once and stores variable-length `[T_i,1024]` outputs in shards. Cache metadata includes RGB mode, model ID, processor pixel budget, vision depth/hidden size, and replacement mode. A mismatched cache is rejected.
