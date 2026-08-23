# P11 architecture

## Optical operator

For canonical token-feature field `X in C^(196x224)`, each macro block applies:

```text
X -> Qwen-to-row-major permutation
  -> phase mask -> FFT_y -> H_y -> IFFT_y
  -> inverse permutation -> CCD/reload/fusion
  -> phase mask -> FFT_x -> H_x -> IFFT_x
  -> CCD/reload/fusion
```

The token-axis transfer function is constant along `f_x`, so different feature
columns cannot exchange energy in that stage. The channel-axis transfer is
constant along `f_y`, so different token rows cannot exchange energy there.
Four token/channel pairs use eight independently learned 3x224x224 phases.

## Layout and feedback contract

The trainable phase, random phase and fixed-feedback phase of a token stage all
live in its physical row-major plane. Only the signal is permuted at the OEO
optical/electronic boundary. Consequently exact BP, matching fixed feedback,
random feedback, phase shifts and phase errors all see one consistent layout.

The exported state has a unique P11 architecture signature. Strict loading into
P09/P10 fails instead of silently treating an axis transfer buffer as ordinary
2-D propagation. A fixed-feedback source for P11 must itself come from P11 with
the same axis schedule and propagation kernels.

## Controlled quantities

- Optical phase parameters: 1,204,224.
- Residual electronic parameters: 733,472.
- Reusable-backbone optical parameter fraction: 55.51%.
- Same P09 electronic mixer in every stage and the same optical fusion minimum
  of 0.5.
- Prepared formal config matches P09's ImageNet recipe, global batch 192,
  initialization seed, learning rates and 90 epochs.

## Physical interpretation and limits

P11 assumes an anisotropic/cylindrical optical system that propagates one axis
while ideally imaging the other; it is not ordinary isotropic free space. The
row-major permutation is feasible at the existing CCD/reload boundary but is a
fixed routing operation, not a learned optical computation.

The 196 tokens are propagated along one rasterized 1-D line together with 28
padding modes. This is a semantic token-axis mixer, not a true local 2-D 14x14
convolution. Also, a full 2-D phase means different feature columns/token rows
can learn different one-dimensional transforms; weights are not shared exactly
like an electronic MLP-Mixer.
