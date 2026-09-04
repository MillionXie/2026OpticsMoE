# Commands

Run every command from the `2026OpticsMoE` repository root:

```bash
cd /path/to/2026OpticsMoE
export LGVQ_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct
export LGVQ_MANIFEST_PATH=/path/to/lgvq_split.csv
```

## 1. Contract check

```bash
python -m experiments.qwen3_vl_2b_lgvq_linear_baseline.baseline preflight
```

The report must state:

```text
qwen_frozen: true
only_trainable_module: nn.Linear(2048,1)
only_trainable_parameter_count: 2049
alignment_target_used: false
```

## 2. Optional two-video smoke checks

```bash
python -m experiments.qwen3_vl_2b_lgvq_linear_baseline.baseline cache --frames 4 --limit 2
python -m experiments.qwen3_vl_2b_lgvq_linear_baseline.baseline cache --frames 16 --limit 2
```

Smoke caches cannot be used for formal training.

## 3. Formal four-frame baseline

```bash
python -m experiments.qwen3_vl_2b_lgvq_linear_baseline.baseline cache --frames 4
python -m experiments.qwen3_vl_2b_lgvq_linear_baseline.baseline train --frames 4
```

## 4. Formal sixteen-frame baseline

```bash
python -m experiments.qwen3_vl_2b_lgvq_linear_baseline.baseline cache --frames 16
python -m experiments.qwen3_vl_2b_lgvq_linear_baseline.baseline train --frames 16
```

## 5. Combine the result table

```bash
python -m experiments.qwen3_vl_2b_lgvq_linear_baseline.baseline report
```

The final human-readable table is:

```text
experiments/qwen3_vl_2b_lgvq_linear_baseline/artifacts/RESULTS.md
```

Machine-readable evidence, checkpoints, prediction CSV files and timings are
under the corresponding `frames4/` and `frames16/` directories.

## One-command formal run

The cache is sharded and resumable, so this command can be restarted safely:

```bash
python -m experiments.qwen3_vl_2b_lgvq_linear_baseline.baseline all 2>&1 | tee run_all.log
```
