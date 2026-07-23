# Architecture specification

## Fixed tensor contract

```text
image                         [B,3,224,224] RGB
patch embedding               [B,196,224]
token optical field           [B,224,224]
channel optical field         [B,224,224]
MoE canvas                    [B,792,792] complex64
central detector ROI          [B,224,224]
final pooled hidden           [B,224]
student CLIP embedding        [B,512]
ImageNet logits               [B,1000]
```

The 28 padded positions are exact zeros at optical loading. They are not
additional learned tokens and are not retained after detector readout.

## Folded block

For block input `X`:

```text
A_t = Softplus(transpose(LN_t(X)))                 [B,224,196]
F_t = zero_pad_columns(A_t, 28)                    [B,224,224]
K,w = Router(F_t)                                  one call
E_t = DirectAmplitudeLoad(F_t, K, w)               [B,792,792] complex
E_t = ExpertStage2(ExpertStage1(E_t, K, w), K, w)
D_t = Detector(GlobalPhase(E_t))                   [B,224,224]
Delta_t = transpose(D_t[:, :, :196])               [B,196,224]
U = X + Delta_t

A_c = Softplus(LN_c(U))                            [B,196,224]
F_c = zero_pad_rows(A_c, 28)                       [B,224,224]
E_c = DirectAmplitudeLoad(F_c, SAME K, SAME w)
E_c = ExpertStage5(ExpertStage4(ExpertStage3(E_c)))
D_c = Detector(SAME GlobalPhase(E_c))              [B,224,224]
Delta_c = D_c[:, :196, :]                          [B,196,224]
Y = U + Delta_c
```

The same global phase parameter is used for both half-block readouts. This
preserves two residual updates without doubling the global phase parameter
count.

## Optical stage

```text
selected expert complex fields
-> independent 224x224 phase-only masks
-> full 792x792 angular-spectrum propagation
-> physical intensity |E|^2
-> crop nine expert apertures
-> independent non-affine LayerNorm
-> ReLU
-> multiply the original sparse routing weights
-> hard-zero unselected expert regions
-> zero-phase complex reload
```

The router is electronic and sample-dependent. It is recomputed once at every
Mixer block because each block receives a different representation, but it is
not recomputed between that block's five optical stages.

## Physical planes

All configurable distances default to 10 cm:

```text
expert phase -> next OEO detector/reload
readout field -> shared global phase
shared global phase -> central CCD detector
```

The amplitude SLM and phase SLM are treated as co-planar through an ideal 4f
relay. The ideal relay itself is not simulated as an extra propagation.

## Trainable/non-trainable components

Trainable:

```text
35 nine-expert phase planes
7 global phase planes
7 electronic top-3 routers
RGB patch embedding
pre-LayerNorm parameters
CLIP projection
ImageNet classifier
```

Non-trainable:

```text
CLIP ViT-B/16 teacher
CLIP ImageNet text prototypes
angular-spectrum transfer functions
OEO LayerNorm affine parameters (disabled)
residual scales (fixed at 1)
```

No trainable amplitude mask, attention, language model, Qwen block, SAM, or
fine-tuning stage is included.
