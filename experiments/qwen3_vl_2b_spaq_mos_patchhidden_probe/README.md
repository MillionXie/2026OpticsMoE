# Qwen3-VL-2B SPAQ MOS Patch-Hidden Probe

This experiment removes the complete optical path and the complete electronic vision transformer stack. It measures how much SPAQ MOS information is already readable from the frozen Qwen visual patch embedding.

```text
cached Qwen processor pixel_values
-> frozen Qwen visual patch embedding
-> capture hidden before visual.blocks[0] [T,1024]
-> valid-token mean pooling [1024]
-> LayerNorm(1024) -> Linear(1024,1)
-> MOS regression
```

It does not execute any Qwen vision transformer block, optical expert, prompt phase, global phase, detector, or language model. Only the 3,073-parameter regression head is trainable. The head uses the same structure and SmoothL1 objective as the optical student.

该实验用于回答：不经过光学传播，仅使用冻结 Qwen patch embedding 的原始 hidden，直接接相同回归头时能够达到多少 SPAQ MOS 性能。它不是 teacher：teacher 使用完整电子 vision transformer；这里使用的是 transformer 之前的 patch hidden。

The experiment reuses the source run's persistent `processor_cache`, so feature extraction does not reopen high-resolution SPAQ JPEG files. Patch-hidden mean features are cached under `features/`; head training then requires no Qwen model.

Reported outputs include fixed-epoch `last` metrics and a separately marked `best_test_srcc` checkpoint. The latter is selection-biased because it uses the test split for checkpoint selection.

