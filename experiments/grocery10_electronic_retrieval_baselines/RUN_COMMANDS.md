# Commands

Run all commands from the `2026OpticsMoE` repository root. Every command is a
single line and contains no shell continuation backslashes.

## Smoke

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_electronic_retrieval_baselines --config experiments/grocery10_electronic_retrieval_baselines/configs/resnet18_smoke.yaml --phase all
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_electronic_retrieval_baselines --config experiments/grocery10_electronic_retrieval_baselines/configs/efficientnet_b0_smoke.yaml --phase all
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_electronic_retrieval_baselines --config experiments/grocery10_electronic_retrieval_baselines/configs/mobilenet_v3_small_smoke.yaml --phase all
```

## Formal runs

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_electronic_retrieval_baselines --config experiments/grocery10_electronic_retrieval_baselines/configs/resnet18_imagenet.yaml --phase all
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_electronic_retrieval_baselines --config experiments/grocery10_electronic_retrieval_baselines/configs/efficientnet_b0_imagenet.yaml --phase all
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.grocery10_electronic_retrieval_baselines --config experiments/grocery10_electronic_retrieval_baselines/configs/mobilenet_v3_small_imagenet.yaml --phase all
```

## Aggregate comparison

```bash
python -m experiments.grocery10_electronic_retrieval_baselines.compare_runs --optical-metrics experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_replaced_finetune/student_metrics.json --baseline-metrics experiments/grocery10_electronic_retrieval_baselines/runs/resnet18_imagenet/metrics/test_metrics.json --baseline-metrics experiments/grocery10_electronic_retrieval_baselines/runs/efficientnet_b0_imagenet/metrics/test_metrics.json --baseline-metrics experiments/grocery10_electronic_retrieval_baselines/runs/mobilenet_v3_small_imagenet/metrics/test_metrics.json --output-dir experiments/grocery10_electronic_retrieval_baselines/runs/comparison
```

## Tests

```bash
pytest experiments/grocery10_electronic_retrieval_baselines/tests -q
```
