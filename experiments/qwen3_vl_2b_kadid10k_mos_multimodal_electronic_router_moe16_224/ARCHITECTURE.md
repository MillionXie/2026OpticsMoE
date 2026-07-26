# Architecture

## End-to-end data flow

Input is the original RGB distorted KADID image. No grayscale conversion is
performed.

```text
KADID distorted RGB image
  + "Predict the human-rated perceptual quality ... on a 1-5 scale. Score:"
→ Qwen chat template, add_generation_prompt=True
→ Qwen processor (min_pixels=max_pixels=37,632)
```

The teacher is a completely electronic, frozen Qwen3-VL-2B backbone. Only its
small normalized linear regression head is trained. The student keeps Qwen
embeddings, merger, native DeepStack injection and final RMSNorm frozen, while
replacing both Transformer stacks with independent optical MoE16 cores.

## Optical tensor mapping

Vision pre-merge hidden has shape `[T_v, 1024]`; language hidden has shape
`[B, S, 2048]`.

Each stack performs:

```text
hidden
→ trainable Linear(D, 224)
→ LayerNorm(224)
→ Softplus
→ zero-pad token rows to [224, 224]
→ electronic top-4 routing
→ 4-stage homogeneous optical MoE16
→ square-law CCD detection over the 986×986 active ROI
→ adaptive-average pool to [224, 224]
→ non-affine LayerNorm + ReLU
→ read valid token rows
→ trainable Linear(224, D)
→ fixed identity residual + optical delta
```

No visual or language token is silently truncated. `T_v > 224` or `S > 224`
raises a runtime error.

## Optical geometry

```text
expert size                 224×224
expert grid                 4×4
expert pitch                254
gap                         30
active/global phase size    986×986
outer propagation canvas    1026×1026
outer zero padding          20 per side
top-k                       4
phase stages per expert     4
pixel pitch                 8 μm
wavelength                  532 nm
all propagation distances   0.10 m
```

The electronic router runs once from the encoded input. The same selected
experts and routing weights are preserved through the optical stages. At each
OEO interlayer:

```text
selected expert field
→ |E|²
→ per-expert non-affine LayerNorm
→ ReLU
→ reapply the original routing weight
→ unselected expert regions are exactly zero
→ reload as zero-phase amplitude
```

The amplitude-loading SLM and phase SLM are modeled as coplanar; the configured
ideal 4f relay therefore does not add a separate propagation operation.

## Supervision

Teacher cache stores four vision targets (three Qwen DeepStack taps plus final
vision stack output) and the final answer-position hidden. Student loss uses
token-wise normalized MSE for hidden matching. The prediction terms use the
same normalized-linear head structure for teacher and student.

KADID DMOS is mapped from `[1,5]` to `[0,1]` only for optimization:

```text
normalized = (DMOS - 1) / 4
DMOS       = normalized × 4 + 1
```

Norm-in-Norm is deliberately a small correlation-shape regularizer. It does not
provide absolute calibration, so the ground-truth and teacher SmoothL1 terms
remain active.

## Split and cache invariants

- Train/test split is reference-disjoint.
- The split is persisted and hashed.
- Teacher and processor cache roots are derived from that hash.
- Pixel budget 37,632 is part of cache identity.
- KADID target scale/direction is part of cache identity.
- The test set is never used for gradient updates.
- In this experiment's inherited protocol, test SRCC is nevertheless evaluated
  each epoch and used to save `best`; report this as selection-biased.
