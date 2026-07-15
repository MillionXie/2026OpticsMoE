# Commands

Run these commands from the repository root. They intentionally contain no shell continuation backslashes.

```bash
python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/configs/cifar10.json --phase download
```

```bash
python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/configs/cifar10.json --phase prepare_data
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/configs/cifar10.json --phase teacher_precompute --device cuda
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/configs/cifar10.json --phase teacher_train --device cuda
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/configs/cifar10.json --phase teacher_logits --device cuda
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/configs/cifar10.json --phase student_train --device cuda
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/configs/cifar10.json --phase student_inference --device cuda
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/configs/cifar10.json --phase all --device cuda
```

```bash
pytest experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/tests -q
```

