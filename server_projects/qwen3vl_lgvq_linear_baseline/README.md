# Qwen3-VL-2B LGVQ strict linear baseline

This is an isolated baseline project. It does not import or modify the optical
network. The formal contract is deliberately narrow:

- Backbone: local `Qwen3-VL-2B-Instruct`, full native Vision tower, native
  merger, native video temporal position encoding, and full language model.
- Backbone parameters: all frozen.
- Added neural network: exactly one shared `nn.Linear(2048, 1)` with 2,049
  trainable parameters.
- Tasks: spatial quality and temporal quality only. Alignment is excluded.
- Task switching: the same video and the same shared scalar head are used with
  two target-specific prompts. Metrics are reported independently.
- Frames: a four-frame baseline and a sixteen-frame baseline.
- Training: 50 epochs on the fixed 2,250-video training split.
- Selection: the fixed 558-video test split is evaluated every epoch and the
  highest mean of spatial/temporal SRCC is kept. No validation split is used.

The Qwen language-model vocabulary projection is not executed during feature
extraction. The scalar head consumes the final valid hidden state at the
assistant-generation prefix after Qwen's final normalization. This is still the
unmodified Qwen multimodal backbone output; avoiding the unused 151,936-way
language logits saves memory without adding a model component.

Large caches and results are written to:

```text
/root/autodl-tmp/qwen3vl_lgvq_linear_baseline_artifacts
```

See `COMMANDS.md` for the exact execution order and `EXPERIMENT_RECORD.md` for
the completed 2026-09-03 results, timing protocol, and evidence paths.
