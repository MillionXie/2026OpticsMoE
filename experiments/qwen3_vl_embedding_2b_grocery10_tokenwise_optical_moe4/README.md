# Qwen3-VL-Embedding-2B token-wise Optical MoE4

This independent experiment tests token-level routing for Grocery-10 product
retrieval. It does not modify the existing Grocery retrieval experiment.

The central difference is that routing is no longer performed once for an
entire 224×224 feature field. Qwen produces up to 196 packed vision tokens with
hidden width 1024. Each token is routed independently and reshaped exactly from
`[1024]` to `[32,32]`; there is no learned `1024→224` input adapter and no
`224→1024` output adapter.

The frozen Qwen patch/position embedding runs before optics. After the optical
vision replacement, the native frozen Qwen vision merger, DeepStack injection,
language model, final RMSNorm, and official 64-D MRL pooling all remain active.
The language model is electronic in this first experiment because a 2048-D
language token cannot be mapped to 32×32 without an adapter.

## Default physical layout

- token field: 32×32;
- experts per token: 2×2 = 4;
- expert gap inside one token group: 2 pixels;
- token group: `32×2 + 2 = 66` pixels per side;
- token groups: 14×14 = 196;
- gap between token groups: 2 pixels;
- active phase area: `66×14 + 2×13 = 950` pixels per side;
- propagation padding: 20 pixels per side;
- FFT canvas: 990×990;
- router: one shared `LayerNorm(1024) → Linear(1024,4)` evaluated for every token;
- routing: top-2 per token;
- propagation distance after each plane: 5 cm at 532 nm and 16 µm pixels.

The four expert phase tensors are shared over all 196 token positions by
default. The SLM therefore displays 196 physical copies of the same four expert
functions, while the router assigns different experts and weights to each
token. Setting `share_expert_phase_across_tokens: false` is supported as a
high-parameter positional ablation, but it is no longer a standard shared MoE.

## Two second-plane modes

`mode: global` is the main run:

```text
Qwen token [1024]
→ fixed LayerNorm + Softplus
→ reshape [32,32]
→ per-token top-2 routing
→ first shared expert phase bank
→ 5 cm propagation
→ square detection + per-token/per-expert LayerNorm + ReLU
→ reapply the original routing weights; unselected experts stay zero
→ 950×950 global phase
→ 5 cm propagation
→ square detection
→ signed per-expert readout and routing-weighted aggregation
→ reshape back to [1024]
```

`mode: expert` replaces the global phase with another four-expert phase bank.
It does not call the router again; the first routing indices and weights are
reused exactly.

## Important switches

- `k_space.enabled`: implemented, default `false`.
- `residual.enabled`: Qwen-style input residual around the optical core,
  default `false` for this experiment.
- `phase.dropout_*`: phase bypass/block bypass regularization, default off.
- `regularization.phase_dc`: phasor DC penalty, default off.
- `router.amplitude_weight_domain`: interpret routing weights in amplitude or
  power domain.
- `oeo.reapply_routing_weights`: restore the original token router weight after
  OEO normalization, default on.
- `oeo.hard_route_mask`: force unselected token-expert regions to zero, default on.
- `detector.aggregation`: routing-weighted signed readout or selected-expert mean.
- `input.amplitude_normalization`: optional per-token max/RMS normalization;
  default none because it alters relative optical energy.
- `share_expert_phase_across_tokens`: shared experts (default) versus positional
  independent phases.

Training uses frozen 64-D Qwen teacher embeddings, cosine embedding KD,
supervised contrastive retrieval loss, and router balance regularization. The
test set may be printed each epoch for observation, but the saved best model is
selected only by training loss.

See [RUN_COMMANDS.md](RUN_COMMANDS.md) for complete commands and
[ARCHITECTURE.md](ARCHITECTURE.md) for tensor-level details.
