# P09: width-96 slim spatial-token-mixer optical backbone

P09 is the controlled successor to P08. It keeps the frozen extracted Qwen3-VL
patch/position stem, the 1024-to-224 adapter, three 224x224 latent optical banks,
eight optical OEO stages and the temporary ImageNet readout. The only intended
architecture change is the electronic bypass inside every optical stage:
P08's 32x32 low-resolution CNN is replaced by an attention-free width-96 mixer.

Each mixer restores the 196 Qwen tokens to the true 14x14 patch grid, applies a
gated depthwise 3x3 spatial update, applies a separately gated channel MLP
update, and returns the tokens to Qwen block-major order. The same mixer weights
are shared across the three latent banks within one stage; the eight stages own
independent mixer weights. There is no electronic Transformer, attention layer,
MoE or language model.

The formal run is a strict architecture ablation against P08: ImageNet-1K,
90 epochs, seed 2026, global batch 192, 6,672 optimizer updates per epoch, the
same augmentation/loss recipe, and the same four learning rates and schedules.
P08 was stopped after its complete epoch 9 at 18.436% validation Top-1, and its
checkpoint/history are retained for matched-epoch comparison.

See [ARCHITECTURE.md](ARCHITECTURE.md), [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)
and [commands/COMMANDS.md](commands/COMMANDS.md).
