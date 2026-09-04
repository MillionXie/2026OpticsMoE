# Qwen3-VL-2B LGVQ strict baseline record

- Date: 2026-09-03
- Server GPU: NVIDIA GeForce RTX 5090 D, 32,607 MiB
- PyTorch: 2.8.0+cu128
- Transformers: 4.57.6

## Baseline boundary

This experiment intentionally contains no optical network and no learned
temporal/readout network beyond one scalar linear layer.

- Frozen backbone: `Qwen3-VL-2B-Instruct`, 2,127,532,032 parameters.
- Trainable backbone parameters: 0.
- Only trainable module: one shared `nn.Linear(2048,1)`.
- Trainable parameters: 2,049 (`linear.weight` and `linear.bias`).
- The same scalar head is shared by both tasks; the prompt selects spatial or
  temporal quality.
- No alignment target, validation split, attention module, MLP, LayerNorm,
  temporal convolution, or task-specific head is added.
- The original Qwen Vision tower, learned merger, native video tokens,
  multimodal language stack, and video temporal positional encoding all run.
- The unused vocabulary projection is skipped; the head reads Qwen's final
  2,048-dimensional assistant-prefix hidden state.

## Data and training

- LGVQ fixed split: 2,250 train videos and 558 test videos.
- Every frame is center-cropped to 65% of the short side and resized to
  448x448, matching the comparison input convention.
- Four-frame positions: 10%, 37%, 63%, and 90% of each clip.
- Sixteen-frame positions: uniform interval midpoints from 3.125% to 96.875%.
- Spatial prompt: `Please evaluate the spatial quality of this video and rate
  it using one of the following five levels: Excellent, Good, Fair, Poor, or
  Bad.`
- Temporal prompt: identical except `spatial` is replaced by `temporal`.
- Targets are standardized independently using training-set statistics.
- Loss: plain mean squared error. There are no auxiliary or ranking losses.
- Optimizer: AdamW, learning rate 1e-3, zero weight decay.
- Epochs: exactly 50.
- Test is evaluated every epoch. The checkpoint with the highest mean of
  spatial SRCC and temporal SRCC is selected. No validation split is used, so
  test leakage is explicit and accepted for this experiment.

## Final test results

| Frames | Target | Best epoch | SRCC | KRCC | PLCC | RMSE | MAE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | Spatial | 50 | 0.6440 | 0.4632 | 0.6806 | 8.3308 | 6.6750 |
| 4 | Temporal | 50 | 0.7440 | 0.5487 | 0.7549 | 9.1132 | 7.4183 |
| 16 | Spatial | 49 | 0.6612 | 0.4783 | 0.6933 | 8.1788 | 6.5214 |
| 16 | Temporal | 49 | 0.7574 | 0.5569 | 0.7663 | 8.8638 | 7.1900 |

Sixteen frames improve all ten reported quantities. Relative to four frames:

- Spatial: SRCC +0.0172, KRCC +0.0151, PLCC +0.0126, RMSE -0.1520,
  MAE -0.1536.
- Temporal: SRCC +0.0134, KRCC +0.0081, PLCC +0.0114, RMSE -0.2494,
  MAE -0.2283.
- Mean spatial/temporal SRCC improves from 0.6940 to 0.7093.

The improvement is consistent but modest. The result is a deliberately strict
linear-probe baseline and should not be conflated with a Qwen baseline that
uses a task-specific MLP, Transformer temporal head, frame-statistics branch,
or separate spatial/temporal heads.

## Controlled processing-time comparison

The timing below uses the same 128 videos and batch size 8 for both frame
counts. It includes two prompt-conditioned Qwen evaluations per video. Model
loading and linear-head training are excluded from the inference timing.

| Frames | Qwen forward / prompt-video | Full pipeline / video, both prompts | Peak GPU memory |
|---:|---:|---:|---:|
| 4 | 23.310 ms | 158.538 ms | 5.220 GiB |
| 16 | 83.491 ms | 317.502 ms | 8.943 GiB |

Increasing from four to sixteen frames therefore costs approximately:

- 3.58x Qwen forward time per prompt;
- 2.00x end-to-end paired-prompt pipeline time, because video decoding is
  shared across the two prompts;
- 3.72 GiB additional peak GPU memory.

For a single prompt, the estimated pipeline time from the same component
measurements is approximately 126.1 ms/video for four frames and 209.5 ms/video
for sixteen frames. A batch-one call to the scalar linear head alone is about
0.027 ms; it is negligible beside Qwen.

The 50-epoch head training takes approximately 2.1-2.4 seconds after the frozen
Qwen features have been cached.

## Evidence locations

Project and commands:

```text
/root/qwen3vl_lgvq_linear_baseline/
```

Final table and machine-readable comparison:

```text
/root/autodl-tmp/qwen3vl_lgvq_linear_baseline_artifacts/RESULTS.md
/root/autodl-tmp/qwen3vl_lgvq_linear_baseline_artifacts/comparison.json
```

Per-frame-count evidence:

```text
/root/autodl-tmp/qwen3vl_lgvq_linear_baseline_artifacts/frames4/
/root/autodl-tmp/qwen3vl_lgvq_linear_baseline_artifacts/frames16/
```

Each directory contains the frozen-Qwen feature cache and extraction timing;
`linear_head/` contains the selected checkpoint, complete 50-epoch history,
test metrics, and all 558 test predictions.

Controlled timing evidence:

```text
/root/autodl-tmp/qwen3vl_lgvq_linear_baseline_artifacts/frames4/smoke_128.report.json
/root/autodl-tmp/qwen3vl_lgvq_linear_baseline_artifacts/frames16/smoke_128.report.json
```
