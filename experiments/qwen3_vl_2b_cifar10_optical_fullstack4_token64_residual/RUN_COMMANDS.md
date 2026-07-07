# Run commands

All commands are deliberately single-line commands without shell continuation backslashes.

## Prepare/download CIFAR-10

```bash
python -m experiments.qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual/configs/cifar10.json --phase prepare_data
```

## Smoke teacher cache

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual/configs/cifar10_smoke.json --phase teacher_precompute --device cuda
```

## Run phases separately

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual/configs/cifar10.json --phase teacher_precompute --device cuda
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual/configs/cifar10.json --phase teacher_train --device cuda
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual/configs/cifar10.json --phase teacher_logits --device cuda
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual/configs/cifar10.json --phase student_train --device cuda
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual/configs/cifar10.json --phase student_inference --device cuda
```

## Full run

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual/configs/cifar10.json --phase all --device cuda
```

## Tests

```bash
pytest experiments/qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual/tests -q
```
