# Qwen3-VL static stem + eight-stage optical ImageNet backbone

This P08 experiment uses Qwen3-VL only as a frozen static-image tokenizer. The
deployed model does not load Qwen3-VL-Embedding-2B and contains no electronic
Transformer, attention block, language model, teacher loss or hidden-state
cache.

The one-time extraction reads only three tensors from the original safetensors
checkpoint. For a still image, the two temporal Conv3D kernels are summed into
one exactly equivalent Conv2D kernel. The 48x48 learned position table is
interpolated once to the fixed 14x14 patch grid and stored in native Qwen 2x2
merge order. The resulting frozen deployment checkpoint is about one million
parameters.

The online path is:

```text
224x224 RGB
-> frozen Qwen patch Conv2D + position table
-> 196x1024 tokens
-> shared LN + Linear(1024,224) + Softplus
-> zero-pad to 224x224
-> parameter-free copy into three non-RGB latent optical banks
-> eight optical/OEO stages with alpha >= 0.5
-> convex bank fusion + token mean/max pooling
-> 448-hidden ImageNet head
```

The three optical banks are latent paths, not RGB wavelengths. They start from
the same amplitude and immediately diverge through independent phase masks.
Their eight phase planes contain 1,204,224 trainable optical values.

See `commands/COMMANDS.md` for the reproducible server commands and
`OPTIMIZATION_LOG.md` for implementation and run decisions.

At completion, `checkpoints/backbone.pt` excludes the disposable ImageNet
readout and retains the frozen stem, token adapter and all eight optical/OEO
feature stages. `forward_features` is the downstream feature contract.
