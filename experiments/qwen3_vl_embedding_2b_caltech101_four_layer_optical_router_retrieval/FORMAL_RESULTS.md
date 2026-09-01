# Formal Router Results

> Status: **COMPLETE**. All pre-registered E1/E2/E4/O2 seeds and the legacy anchor passed artifact checks.

## Reporting contract

- E1/E2/E4/O2 each use optimization seeds 42, 43, and 44; the Caltech split and PK batch order remain fixed.
- Each test evaluation is performed on the pre-selected `ema_best_train_loss_checkpoint.pt`; the checkpoint is selected only by minimum training total loss.
- The training log must contain no finite per-epoch test metric. The held-out test is evaluated once after selection under this run protocol.
- The historical Caltech test has been examined in earlier project work, so it is **not globally unseen**. The defensible statement is: this adaptation run did not use test results for epoch selection.
- Mean ± sample SD is across three optimization seeds. Brackets are a 95% percentile bootstrap CI of the seed mean (10,000 resamples; fixed seed 20260902). With only three seeds, the CI is descriptive and coarse.

## Aggregate retrieval results

All values below are percentage points. `Δ legacy` compares the three-seed mean with the single fixed warmstart5 anchor; it is not a paired three-seed estimate.

| Variant | Router | k | Seeds | Top-1 mean ± SD [95% CI] | Top-3 mean ± SD [95% CI] | MRR mean ± SD [95% CI] | Top-1 Δ legacy |
|---|---|---:|---|---:|---:|---:|---:|
| E1 | electronic | 1 | 42,43,44 | 82.83 ± 0.76 [82.00, 83.50] | 93.50 ± 0.00 [93.50, 93.50] | 89.04 ± 0.47 [88.53, 89.46] | +1.83 |
| E2 | electronic | 2 | 42,43,44 | 82.50 ± 1.00 [81.50, 83.50] | 93.50 ± 0.50 [93.00, 94.00] | 88.90 ± 0.53 [88.33, 89.37] | +1.50 |
| E4 | electronic | 4 | 42,43,44 | 81.83 ± 0.29 [81.50, 82.00] | 93.50 ± 0.00 [93.50, 93.50] | 88.64 ± 0.14 [88.48, 88.73] | +0.83 |
| O2 | optical | 2 | 42,43,44 | 83.00 ± 0.00 [83.00, 83.00] | 93.50 ± 0.00 [93.50, 93.50] | 89.22 ± 0.02 [89.20, 89.24] | +2.00 |

## Legacy anchor

Fixed warmstart5 anchor (one run, no optimizer adaptation): Top-1 81.00%, Top-3 93.00%, MRR 87.63%.

## Query-level paired tests

McNemar tests are computed separately for each paired optimization seed on identical `sample_id` values. Anchor comparisons reuse the same fixed anchor predictions and are labeled accordingly. No rows are pooled across seeds, avoiding pseudo-replication. `p_Holm` adjusts across all tests emitted in this report.

| Comparison | Seed | n | left-only correct | right-only correct | Δ Top-1 | exact p | p_Holm |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 vs E2 | 42 | 200 | 0 | 0 | +0.00 | 1 | 1 |
| E1 vs E2 | 43 | 200 | 1 | 0 | +0.50 | 1 | 1 |
| E1 vs E2 | 44 | 200 | 1 | 0 | +0.50 | 1 | 1 |
| E4 vs E2 | 42 | 200 | 0 | 3 | -1.50 | 0.25 | 1 |
| E4 vs E2 | 43 | 200 | 2 | 1 | +0.50 | 1 | 1 |
| E4 vs E2 | 44 | 200 | 1 | 3 | -1.00 | 0.625 | 1 |
| O2 vs E2 | 42 | 200 | 1 | 2 | -0.50 | 1 | 1 |
| O2 vs E2 | 43 | 200 | 5 | 2 | +1.50 | 0.4531 | 1 |
| O2 vs E2 | 44 | 200 | 3 | 2 | +0.50 | 1 | 1 |
| E1 vs legacy | 42 | 200 | 8 | 3 | +2.50 | 0.2266 | 1 |
| E1 vs legacy | 43 | 200 | 4 | 2 | +1.00 | 0.6875 | 1 |
| E1 vs legacy | 44 | 200 | 6 | 2 | +2.00 | 0.2891 | 1 |
| E2 vs legacy | 42 | 200 | 8 | 3 | +2.50 | 0.2266 | 1 |
| E2 vs legacy | 43 | 200 | 3 | 2 | +0.50 | 1 | 1 |
| E2 vs legacy | 44 | 200 | 5 | 2 | +1.50 | 0.4531 | 1 |
| E4 vs legacy | 42 | 200 | 6 | 4 | +1.00 | 0.7539 | 1 |
| E4 vs legacy | 43 | 200 | 5 | 3 | +1.00 | 0.7266 | 1 |
| E4 vs legacy | 44 | 200 | 5 | 4 | +0.50 | 1 | 1 |
| O2 vs legacy | 42 | 200 | 8 | 4 | +2.00 | 0.3877 | 1 |
| O2 vs legacy | 43 | 200 | 8 | 4 | +2.00 | 0.3877 | 1 |
| O2 vs legacy | 44 | 200 | 8 | 4 | +2.00 | 0.3877 | 1 |

## Run inventory

| Run | Variant | Seed | Status | Selected epoch | Missing / error |
|---|---|---:|---|---:|---|
| `electronic_legacy_topk2_anchor` | legacy | 42 | complete | 8 | — |
| `electronic_power_topk1` | E1 | 42 | complete | 9 | — |
| `electronic_power_topk1_seed43` | E1 | 43 | complete | 6 | — |
| `electronic_power_topk1_seed44` | E1 | 44 | complete | 9 | — |
| `electronic_power_topk2` | E2 | 42 | complete | 9 | — |
| `electronic_power_topk2_seed43` | E2 | 43 | complete | 6 | — |
| `electronic_power_topk2_seed44` | E2 | 44 | complete | 9 | — |
| `electronic_power_topk4` | E4 | 42 | complete | 9 | — |
| `electronic_power_topk4_seed43` | E4 | 43 | complete | 9 | — |
| `electronic_power_topk4_seed44` | E4 | 44 | complete | 8 | — |
| `optical_power_topk2` | O2 | 42 | complete | 9 | — |
| `optical_power_topk2_seed43` | O2 | 43 | complete | 9 | — |
| `optical_power_topk2_seed44` | O2 | 44 | complete | 9 | — |

## Machine-readable handoff

- `formal_results/aggregate_results.json`: complete provenance, run audit, aggregate metrics, per-class metrics, and paired tests.
- `formal_results/aggregate_metrics.csv`: one row per variant and metric.
- `formal_results/per_class_metrics.csv`: class-resolved mean, sample SD, and bootstrap CI.
- `formal_results/paired_mcnemar.csv`: per-seed exact paired tests.
- `formal_results/run_inventory.csv`: checkpoint and training-selection audit.

Regenerate from the repository root:

```text
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.formal_results
```
