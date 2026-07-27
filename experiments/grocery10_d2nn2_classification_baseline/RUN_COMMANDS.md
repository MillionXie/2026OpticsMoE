# Commands

Run every command from the `2026OpticsMoE` repository root. Commands are kept
on one line and contain no continuation backslashes.

## Prepare data

```bash
python -m experiments.grocery10_d2nn2_classification_baseline --config experiments/grocery10_d2nn2_classification_baseline/configs/grocery10_d2nn2.yaml --phase prepare_data
```

## Smoke

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_d2nn2_classification_baseline --config experiments/grocery10_d2nn2_classification_baseline/configs/grocery10_d2nn2_smoke.yaml --phase all
```

## Full training

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_d2nn2_classification_baseline --config experiments/grocery10_d2nn2_classification_baseline/configs/grocery10_d2nn2.yaml --phase all
```

## Test an existing checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_d2nn2_classification_baseline --config experiments/grocery10_d2nn2_classification_baseline/configs/grocery10_d2nn2.yaml --phase test
```

## Unit tests

```bash
pytest experiments/grocery10_d2nn2_classification_baseline/tests -q
```
