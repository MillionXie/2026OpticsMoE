# P08 locked architecture

## Scientific boundary

The Qwen component is a frozen input stem, not an electronic backbone. Its
output is the uncontextualized 1024-wide patch/position token sequence. All 24
Qwen vision Transformer blocks, every merger MLP, the language stack and the
embedding head are absent from both training and deployment.

This distinction is important: the model tests whether a pretrained visual
tokenizer can make a million-phase optical network trainable on ImageNet. It
does not claim that the optical network replaces the tokenizer itself.

## Static stem

- input: RGB 224x224 after ImageNet augmentation;
- Qwen normalization: mean/std 0.5;
- frozen patch operator: equivalent static Conv2D, kernel/stride 16;
- patch grid: 14x14 = 196 tokens;
- token width: 1024;
- learned position table: bilinear 48x48 -> 14x14 with `align_corners=True`;
- token order: Qwen native 2x2 spatial-merger order;
- frozen deployment parameters: 988,160;
- full Qwen loaded during training/inference: no.

## Optical input

Each 1024-wide token passes through one shared trainable adapter:

```text
LayerNorm(1024) -> Linear(1024,224) -> Softplus -> token RMS normalization
```

The 196x224 tensor is padded with 28 zero token rows to 224x224, then copied
without parameters into three latent banks. Copying retains the earlier
million-phase capacity without interpreting the banks as RGB wavelengths.

## Optical trunk

- stages: 8;
- banks per stage: 3;
- phase values: 8 x 3 x 224 x 224 = 1,204,224;
- wavelength: 532 nm;
- simulated pitch: 16 um;
- propagation distance: 5 cm;
- detector: square-law intensity, spatial standardization, ReLU and RMS reload;
- residual: low-resolution 3->64->64->3 convolution at 32x32;
- fusion: constrained optical coefficient alpha >= 0.5;
- electronic Transformer/attention/token mixer: none;
- global electronic bypass around the optical trunk: none.

## Readout and accounting

The final three banks are fused by three softmax weights. Only the first 196
rows are retained. Token mean and max pooling produce 448 values, followed by a
448-hidden two-layer classifier.

The runtime writes exact accounting to `manifest.json`. The locked design must
pass both conditions before training starts:

- optical trainable parameters / all trainable parameters >= 0.5;
- minimum per-stage optical fusion coefficient >= 0.5.

The frozen Qwen stem is reported separately. No claim based only on trainable
parameters should hide its fixed electronic cost.
