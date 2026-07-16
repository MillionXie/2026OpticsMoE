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
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/configs/cifar10.json --phase student_train --device cuda
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/configs/cifar10.json --phase student_inference --device cuda
```

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/configs/cifar10.json --phase all --device cuda
```

Override the batch-only terminal refresh interval without editing JSON:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/configs/cifar10.json --phase student_train --device cuda --log-interval-batches 20
```

Use a rotating 500-sample window from every class in each epoch, while allowing later epochs to cover the remaining cached samples:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/configs/cifar10.json --phase student_train --device cuda --train-samples-per-class-per-epoch 500
```

```bash
pytest experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/tests -q
```
