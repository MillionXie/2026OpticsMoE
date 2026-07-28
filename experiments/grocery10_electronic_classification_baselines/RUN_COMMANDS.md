# Commands

Run every command from the `2026OpticsMoE` repository root. Commands are one
line and contain no continuation backslashes.

## Smoke tests

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_electronic_classification_baselines --config experiments/grocery10_electronic_classification_baselines/configs/resnet18_smoke.yaml --phase all
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_electronic_classification_baselines --config experiments/grocery10_electronic_classification_baselines/configs/efficientnet_b0_smoke.yaml --phase all
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_electronic_classification_baselines --config experiments/grocery10_electronic_classification_baselines/configs/mobilenet_v3_small_smoke.yaml --phase all
```

## Formal direct-classification runs

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_electronic_classification_baselines --config experiments/grocery10_electronic_classification_baselines/configs/resnet18_classification.yaml --phase all
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_electronic_classification_baselines --config experiments/grocery10_electronic_classification_baselines/configs/efficientnet_b0_classification.yaml --phase all
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_electronic_classification_baselines --config experiments/grocery10_electronic_classification_baselines/configs/mobilenet_v3_small_classification.yaml --phase all
```

## Evaluate an existing checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_electronic_classification_baselines --config experiments/grocery10_electronic_classification_baselines/configs/resnet18_classification.yaml --phase evaluate
```

## Unit tests

```bash
pytest experiments/grocery10_electronic_classification_baselines/tests -q
```
