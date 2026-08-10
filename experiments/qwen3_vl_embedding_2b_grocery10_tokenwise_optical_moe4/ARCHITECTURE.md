# Architecture and tensor contract

## Optical-language variant (recommended current experiment)

```text
Qwen patch/position embedding [Nv,1024]
-> Vision token-wise Optical MoE4 [Nv,1024]
-> frozen Qwen vision merger
-> one normal visual-token/text-token merge (DeepStack disabled)
-> multimodal hidden [B,S,2048]
-> trainable LayerNorm + Linear(2048,1024)
-> Language token-wise Optical MoE4 [valid S,1024]
-> trainable Linear(1024,2048)
-> frozen Qwen final RMSNorm
-> official 64-D MRL embedding + L2 normalization
```

The `1024->2048` bridge is required because the frozen final RMSNorm and the
official embedding readout operate in Qwen's 2048-D language space. It is not
a classifier. Padding language tokens never enter the optical panel, and a
sequence longer than 196 valid tokens raises an error instead of truncation.

After first-plane detection, per-token/per-expert LayerNorm would erase all
relative response magnitude. With response-amplitude preservation enabled, the
implementation measures pre-LN RMS field response, removes the already-applied
router scale, normalizes response across the selected experts of that token,
clips only for numerical stability, then reapplies the original routing scale
once. Unselected experts remain exactly zero.

The shared configuration uses four expert identities repeatedly at every token
position. The nonshared ablation instead allocates independent phase masks for
all `196 x 4` token-position/expert pairs; this tests position capacity but is
not the canonical parameter-sharing MoE interpretation.

## Boundary with Qwen

The replacement is installed at `visual.blocks[0]`. Native Qwen vision blocks
are bypassed, while patch embedding and positional encoding stay frozen and
native. Input is packed `[sum(T_i),1024]` plus Qwen `cu_seqlens`.

Every `T_i` is checked. The default hardware panel supports at most 196 tokens;
larger images raise an explicit error asking for a lower processor pixel budget.
No token is cropped, pooled, or silently truncated.

The optical output has exactly the same packed shape and dtype. Since there is
one optical vision result, bypassed visual blocks expose that same result at
Qwen's native DeepStack tap indexes. Native frozen mergers then inject it into
the native frozen language stack.

## Per-token routing

For padded batch representation `[B,196,1024]`:

```text
router_input = optional LayerNorm(token)
logits       = shared Linear(1024, 4)
probability  = softmax(logits / temperature)
top-2        = deterministic sparse selection (noise is train-only and off by default)
weight       = selected probability renormalized over top-2
```

The balance loss aggregates importance and hard load over all valid tokens,
not over images. Padding token positions contribute neither routing load nor
feature loss.

## Optical mapping

The 1024 hidden values are normalized/nonnegative-encoded without changing
dimension, then reshaped to 32×32. Each token has a dedicated 66×66 spatial
group containing four expert apertures. Token index is mapped row-major into
the 14×14 token-group grid. Aperture linear indices are validated as unique.

The shared-expert default has four trainable 32×32 phase tensors per expert
plane. These four tensors are repeated physically across all token groups.
With raw phase zero and sigmoid phase parameterization, every physical phase
starts at π, not at phase zero.

## OEO and second plane

After the first 5 cm propagation, CCD intensity is cropped at every
token/expert aperture. Each crop receives its own spatial LayerNorm, followed
by ReLU or Softplus. The original routing decision is reused. No second router
forward occurs.

The main `global` mode uses a trainable 950×950 phase-only plane. The `expert`
ablation instead uses a second four-expert 32×32 bank. Both then propagate 5 cm
to the final detector.

Final detector crops are LayerNormed as signed features before aggregation.
No ReLU is applied to the Qwen-bound feature. Aggregating the four crops yields
one 32×32 signed tensor per token, flattened back to 1024.

## Default parameter counts

Counts that do not depend on Qwen version:

- router Linear: `1024×4 + 4 = 4,100` parameters (router LN is non-affine);
- first shared expert bank: `4×32×32 = 4,096` phase parameters;
- global second plane: `950×950 = 902,500` phase parameters;
- total core in the default non-affine configuration: 910,696 parameters.

The `expert` second-plane ablation replaces 902,500 parameters with 4,096,
giving 12,292 parameters. Runtime writes exact counts to
`metrics/model_parameters.json`.
