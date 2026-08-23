# P10 architecture

## Hypothesis

P09 applies the same broad 50 mm angular-spectrum propagation in all eight
stages. P10 tests whether serial local-to-global optics preserve fine structure
before broad mixing without changing trainable capacity.

Each macro block is:

```text
input
  -> phase mask -> 5 mm 2-D ASM -> CCD/nonlinearity -> electronic fusion
  -> phase mask -> 50 mm 2-D ASM -> CCD/nonlinearity -> electronic fusion
```

There are four macro blocks and eight independently learned 3x224x224 phase
planes. The P09 width-96 spatial/channel electronic mixer remains in every
stage, and the outer optical fusion gate remains constrained to at least 0.5.

## Controlled quantities

- Optical phase parameters: 1,204,224.
- Residual electronic parameters: 733,472.
- Reusable-backbone optical parameter fraction: 55.51%.
- Same frozen stem, adapter, task head and trainable initialization as P09.
- Same seed, ImageNet data, global batch 192, learning rates, augmentation,
  warmup, cosine schedule and 90-epoch horizon in the prepared formal config.

The local/global transfer functions are persistent checkpoint buffers but not
trainable parameters. A unique P10 architecture-signature buffer prevents a
P10 checkpoint from being silently loaded into P09 or P11 with `strict=True`.

## Physical limitation

The current simulator uses an FFT-sized periodic field without zero padding.
Its propagation is therefore circular at the boundary, and the 5 mm operator
should be described as a reduced-receptive-field diffractive operator rather
than a literal finite-support convolution. A deployment study should repeat the
comparison with measured PSFs or padded/cropped propagation.
