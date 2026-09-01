# Formal Router Results

> Status: **PENDING**. Run the aggregation command below after the legacy anchor
> and all E1/E2/E4/O2 seeds have completed their explicit test evaluation.

This file is generated from run artifacts. It must not be filled by copying log
lines or manually choosing a favorable epoch.

The reporting contract is:

- E1/E2/E4/O2 use optimization seeds 42, 43, and 44.
- Each run evaluates `ema_best_train_loss_checkpoint.pt`, selected by minimum
  training total loss without per-epoch test observations.
- The test is used once after checkpoint pre-selection within each adaptation
  run. The historical Caltech test was viewed in earlier project work and is
  therefore not globally unseen.
- Results report mean ± sample standard deviation and a fixed-seed percentile
  bootstrap 95% confidence interval over the three optimization seeds.
- Query-level McNemar tests are emitted only when matching
  `retrieval_results.csv` files contain the same `sample_id` values.

From the repository root, run:

```text
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.formal_results
```

For CI/automation, require all artifacts to be complete:

```text
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.formal_results --require-complete
```

The command rewrites this file and creates the machine-readable files under
`formal_results/`.
