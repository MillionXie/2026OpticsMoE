# SPAQ Qwen3-VL-2B DeepStack-aware optical MoE distillation

This independent experiment evaluates single-attribute SPAQ image-quality regression with the complete Qwen3-VL image-plus-prompt path. It supports two controlled students:

- `electronic`: the complete vision Transformer is replaced by a homogeneous optical MoE9x5, while the original Qwen language decoder remains frozen and electronic.
- `optical_moe`: both the complete vision Transformer and complete language decoder are replaced by separate homogeneous optical MoE9x5 modules.

The default formal task is MOS. Images remain RGB and the prompt is processed by the native tokenizer, chat template, and multimodal processor.

## Correct Qwen3-VL DeepStack handling

Qwen3-VL-2B exposes intermediate visual features after vision blocks 5, 11, and 17, then merges them from 1024 to 2048 dimensions and injects them after language layers 0, 1, and 2. The replacement preserves those native merger and injection locations.

The vision MoE exposes optical stages 1, 3, and 4 at the three intermediate vision positions. Its stage-5 detector readout replaces the final vision output. The frozen Qwen deepstack mergers and final vision merger remain in place.

In `optical_moe` mode, five language optical stages occupy decoder-layer positions 0 through 4. Native DeepStack additions after layers 0, 1, and 2 are detected as hidden-space deltas, projected through the shared language input adapter, fanned out with the same sample router, and added to the current expert field before the next optical stage. This prevents DeepStack features from being injected only after a completed optical stack.

## Optical MoE

Vision and language use independent parameters but the same verified geometry: 480×480 canvas, 450×450 active area, nine 120×120 experts on a 3×3 grid, input-dependent top-3 routing, five phase planes per expert, optoelectronic inter-layer detection/normalization/nonlinearity, one global phase, and a full-plane detector readout. Token matrices are projected to 120 channels and zero-row padded; there is no token-field interpolation.

## Distillation

The electronic teacher cache contains four pre-merger vision targets (blocks 5, 11, 17, and 23), the final answer-position hidden state, the target score, token counts, and grid metadata. Student training combines mean LayerNorm-MSE over the four vision targets, answer-hidden LayerNorm-MSE, teacher-score SmoothL1 distillation, ground-truth SmoothL1 regression, and router balance losses. The teacher never runs online during student epochs.

The two modes use separate output directories. Processor pixels, prompt, task, split digest, architecture dimensions, and DeepStack indexes are cache-validated.
