# P10: dual-scale local/global optical backbone

P10 is a controlled optical-operator variant of P09. It keeps the frozen Qwen
patch/position stem, trainable 1024-to-224 adapter, three optical banks, eight
phase planes, per-stage width-96 electronic mixer, ImageNet readout, parameter
counts and training recipe. Only the fixed propagation distance changes:

```text
stage 1 local 5 mm  -> stage 2 global 50 mm
stage 3 local 5 mm  -> stage 4 global 50 mm
stage 5 local 5 mm  -> stage 6 global 50 mm
stage 7 local 5 mm  -> stage 8 global 50 mm
```

The short and long propagators have measured impulse-response r90 radii of
approximately 5.8 and 58 pixels under the current discrete optical settings.
"Local" therefore means a substantially smaller receptive field than P09; it
does not claim exact equivalence to a digital 3x3 convolution.

The formal 90-epoch config and guarded launcher are prepared but have not been
run. See [ARCHITECTURE.md](ARCHITECTURE.md),
[OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md) and
[commands/COMMANDS.md](commands/COMMANDS.md).
