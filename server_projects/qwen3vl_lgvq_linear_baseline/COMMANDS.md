# Commands

Run every command from the project root:

```bash
cd /root/qwen3vl_lgvq_linear_baseline
```

## 1. Contract check

```bash
python baseline.py preflight
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
python baseline.py cache --frames 4 --limit 2
python baseline.py cache --frames 16 --limit 2
```

Smoke caches cannot be used for formal training.

## 3. Formal four-frame baseline

```bash
python baseline.py cache --frames 4
python baseline.py train --frames 4
```

## 4. Formal sixteen-frame baseline

```bash
python baseline.py cache --frames 16
python baseline.py train --frames 16
```

## 5. Combine the result table

```bash
python baseline.py report
```

The final human-readable table is:

```text
/root/autodl-tmp/qwen3vl_lgvq_linear_baseline_artifacts/RESULTS.md
```

Machine-readable evidence, checkpoints, prediction CSV files and timings are
under the corresponding `frames4/` and `frames16/` directories.

## One-command formal run

The cache is sharded and resumable, so this command can be restarted safely:

```bash
python baseline.py all 2>&1 | tee run_all.log
```
