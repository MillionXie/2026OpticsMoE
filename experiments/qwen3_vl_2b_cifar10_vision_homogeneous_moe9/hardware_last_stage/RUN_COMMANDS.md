# Export the latest main experiment best student hardware package

```bash
CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9.hardware_last_stage.export_oneshot --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/hardware_last_stage/configs/oneshot_main_best.json --device cuda
```

# Export the historical batch-8 comparison package

```bash
CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9.hardware_last_stage.export_oneshot --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/hardware_last_stage/configs/oneshot_batch8_best.json --device cuda
```

# Fine-tune the last electronic readout on captured CCD training frames

```bash
CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9.hardware_last_stage.ccd_readout --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/hardware_last_stage/configs/ccd_readout.json --phase finetune --device cuda
```

# Evaluate captured CCD test frames

```bash
CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9.hardware_last_stage.ccd_readout --config experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/hardware_last_stage/configs/ccd_readout.json --phase inference --device cuda
```
