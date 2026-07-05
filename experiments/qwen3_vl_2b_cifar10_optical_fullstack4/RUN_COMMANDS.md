# Run commands

```bash
# Smoke test
python -m experiments.qwen3_vl_2b_cifar10_optical_fullstack4.run \
  --config experiments/qwen3_vl_2b_cifar10_optical_fullstack4/configs/cifar10_smoke.json \
  --phase all

# Full run
python -m experiments.qwen3_vl_2b_cifar10_optical_fullstack4.run \
  --config experiments/qwen3_vl_2b_cifar10_optical_fullstack4/configs/cifar10.json \
  --phase all

# Run phases separately
python -m experiments.qwen3_vl_2b_cifar10_optical_fullstack4.run \
  --config experiments/qwen3_vl_2b_cifar10_optical_fullstack4/configs/cifar10.json \
  --phase teacher_precompute

python -m experiments.qwen3_vl_2b_cifar10_optical_fullstack4.run \
  --config experiments/qwen3_vl_2b_cifar10_optical_fullstack4/configs/cifar10.json \
  --phase student_train
```
