# Fixed-feedback optical fine-tuning: main-result record

## Executive conclusion

At the prespecified epoch-50 endpoint, no fine-tuning reached 42.33% test accuracy, above BP (37.61 ± 0.35%). This is not evidence that fine-tuning is intrinsically harmful: the downstream set is a 10-class CIFAR-100-C subset whose classes and 100-way output coordinates were already learned during CIFAR-100 pretraining, while the 50-epoch full-parameter adaptation strongly overfits the small corrupted-domain split.

The geometric result is nevertheless internally consistent with the fixed-feedback hypothesis: FA-pretrained ends with 0.997× the BP drift and cosine 0.972 to the matched BP update, whereas FA-random moves 1.236× as far with cosine 0.364. The separation is qualitatively similar to the reference paper, but substantially weaker for the random-feedback baseline.

## What each metric means

- `relative_parameter_drift = ||θ_T − θ_pre||₂ / ||θ_pre||₂`. It is measured from the shared pretrained checkpoint.
- `endpoint_cosine_to_matched_bp` is the cosine between `θ_method,T − θ_pre` and the same-seed BP update `θ_BP,T − θ_pre` at the same epoch.
- BP is therefore exactly 1 up to floating-point precision. No fine-tuning has a zero update, so its cosine is undefined (N/A), not 0.
- The task-performance checkpoint policy and the geometric matched-endpoint policy answer different questions and must not be mixed.

## Checkpoint policy and fairness

The original comparison used `last.pt` at epoch 50 for BP/FA and the fixed pretrained-best model for No Fine-Tuning. This is a valid fixed-budget endpoint comparison because all trained methods receive the same budget, but it is not an “each method's best result” comparison.

For reporting, use both of the following without using test data for model selection:

1. **Primary fixed-budget endpoint:** epoch 50 for BP, FA-pretrained and FA-random; fixed pretrained model for NoFT.
2. **Secondary validation-selected result:** select the epoch with highest validation accuracy independently per seed/method, then report that checkpoint's test accuracy. NoFT remains fixed.

Do not compare geometry at independently selected best epochs. The endpoint vectors would correspond to different training times. The geometric analysis uses matched epoch 10 and epoch 50 checkpoints.

The reference paper itself uses mixed rules: GSM8K reports the best scheduled test checkpoint; FA on SAMSum uses validation selection; BP on SAMSum reports the best scheduled test checkpoint. It explicitly labels the test-selected values as best-observed scheduled checkpoints rather than independent held-out estimates. Its geometric analysis instead uses matched epoch-3 checkpoints for every method and seed. Our validation-selected secondary table is statistically cleaner than selecting on test.

## Task performance

| Policy | Method | Test accuracy, mean ± sample SD | Mean selected epoch | Train−test gap |
|---|---|---:|---:|---:|
| fixed epoch | BP | 37.61 ± 0.35% | 50.0 | 35.28 pp |
| fixed epoch | FA-pretrained | 37.17 ± 0.50% | 50.0 | 35.85 pp |
| fixed epoch | FA-random | 37.00 ± 1.04% | 50.0 | 31.91 pp |
| validation selected | BP | 41.33 ± 1.32% | 7.3 | 11.59 pp |
| validation selected | FA-pretrained | 40.72 ± 2.11% | 10.7 | 14.09 pp |
| validation selected | FA-random | 39.11 ± 3.53% | 6.7 | 11.37 pp |
| fixed model | No fine-tuning | 42.33% | 0 | N/A |

Even after validation selection, BP reaches 41.33 ± 1.32%, still slightly below NoFT (42.33%). Thus the result is not only a last-epoch artifact, although the late-epoch overfitting makes the fixed-endpoint gap larger.

## Matched-endpoint geometry

| Epoch | Method | Relative drift | Drift / BP | Cosine to BP |
|---:|---|---:|---:|---:|
| 10 | BP | 0.1804 ± 0.0031 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 10 | FA-pretrained | 0.1803 ± 0.0032 | 0.999 ± 0.001 | 0.997 ± 0.000 |
| 10 | FA-random | 0.2197 ± 0.0049 | 1.218 ± 0.025 | 0.433 ± 0.045 |
| 50 | BP | 0.4033 ± 0.0014 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 50 | FA-pretrained | 0.4019 ± 0.0012 | 0.997 ± 0.001 | 0.972 ± 0.001 |
| 50 | FA-random | 0.4984 ± 0.0131 | 1.236 ± 0.035 | 0.364 ± 0.009 |

## Interpretation relative to the paper

The trend agrees qualitatively with the paper: FA-pretrained stays close to BP in both update magnitude and direction, while random fixed feedback deviates more. However, this run does **not** reproduce the paper's limited-drift regime quantitatively:

- BP relative drift is 0.403, far larger than the approximately 0.004 scale reported for language-model SFT.
- FA-random moves only 1.24× as far as BP rather than approximately 2×.
- FA-random cosine is 0.364, lower than FA-pretrained but not near orthogonal.

Therefore the present experiment supports a preliminary geometric analogy, not a close quantitative replication.

## Why No Fine-Tuning is higher

1. **The downstream classes are not unseen.** They are selected CIFAR-100 classes under CIFAR-100-C corruptions, and the same 100-way head coordinates are retained.
2. **The downstream set is small.** Fine-tuning uses 1,800 training images against millions of trainable optical phase values plus electronic parameters.
3. **The phase learning rate and horizon are aggressive.** Validation-selected epochs occur early, while training accuracy keeps rising and test accuracy falls.
4. **The experiment violates the intended small-drift premise.** A relative drift near 0.40 means the frozen pretrained feedback is being tested far from its initialization.
5. **NoFT already solves much of the task.** Fine-tuning has little headroom and can easily overwrite robust pretrained features.

## Recommended next experiment

- Keep these completed runs unchanged as the fixed-budget baseline.
- Add a genuinely distinct downstream task (actual CIFAR-10 with a newly initialized 10-way head, or a disjoint CIFAR-100 class split). This makes NoFT an honest transfer baseline.
- Use a shorter 10–15 epoch adaptation window, validation selection, and a lower phase learning rate or warmup/cosine decay to target a much smaller drift regime.
- Report drift during training and predefine a drift budget; do not select a checkpoint using test accuracy.
- If the goal is to test feedback rather than readout mismatch, first train the new readout with the optical backbone frozen, then jointly fine-tune all methods from that same checkpoint.
- For the current same-class corruption task, consider partial freezing or an explicit penalty to the pretrained optical operator; the objective should preserve robust features rather than relearn the labels.

## Figures and source data

- `figures/task_performance.*`: test trajectories and validation-selected test performance.
- `figures/endpoint_geometry.*`: drift trajectories and matched-endpoint update geometry.
- `source_data/training_trajectories.csv`: all epochs, methods and seeds.
- `source_data/checkpoint_performance.csv`: fixed endpoint and validation-selected checkpoints.
- `source_data/endpoint_geometry.csv`: matched epoch-10 and epoch-50 geometry.

All reported dispersion values are sample standard deviations over three matched seeds. NoFT is one deterministic checkpoint and therefore has no seed SD.
