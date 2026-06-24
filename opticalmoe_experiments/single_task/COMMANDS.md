# Single-Task Commands

Run these commands from `opticalmoe_experiments/`.

All commands are written as complete one-line commands for easy copy/paste.

## Smoke Tests

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/mnist_learnable_moe_E9_complex.yaml --run_name mnist_learnable_moe_E9_complex_smoke --epochs 1 --smoke_test --disable_visualization
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/mnist_d2nn_matched_params.yaml --run_name mnist_d2nn_matched_smoke --epochs 1 --smoke_test --disable_visualization
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/mnist_lenet5.yaml --run_name mnist_lenet5_smoke --epochs 1 --smoke_test --disable_visualization
```

## MNIST

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/mnist_learnable_moe_E9_complex.yaml --run_name mnist_learnable_moe_E9_complex_seed7
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/mnist_fixed_moe_E9_complex.yaml --run_name mnist_fixed_moe_E9_complex_seed7
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/mnist_d2nn_matched_params.yaml --run_name mnist_d2nn_matched_seed7
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/mnist_lenet5.yaml --run_name mnist_lenet5_seed7
```

## Fashion-MNIST

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/fashionmnist_learnable_moe_E9_complex.yaml --run_name fashionmnist_learnable_moe_E9_complex_seed7
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/fashionmnist_fixed_moe_E9_complex.yaml --run_name fashionmnist_fixed_moe_E9_complex_seed7
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/fashionmnist_d2nn_matched_params.yaml --run_name fashionmnist_d2nn_matched_seed7
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/fashionmnist_lenet5.yaml --run_name fashionmnist_lenet5_seed7
```

## KMNIST

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/kmnist_learnable_moe_E9_complex.yaml --run_name kmnist_learnable_moe_E9_complex_seed7
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/kmnist_fixed_moe_E9_complex.yaml --run_name kmnist_fixed_moe_E9_complex_seed7
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/kmnist_d2nn_matched_params.yaml --run_name kmnist_d2nn_matched_seed7
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/kmnist_lenet5.yaml --run_name kmnist_lenet5_seed7
```

## EMNIST Letters

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/emnist_letters_learnable_moe_E9_complex.yaml --run_name emnist_letters_learnable_moe_E9_complex_seed7
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/emnist_letters_fixed_moe_E9_complex.yaml --run_name emnist_letters_fixed_moe_E9_complex_seed7
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/emnist_letters_d2nn_matched_params.yaml --run_name emnist_letters_d2nn_matched_seed7
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/emnist_letters_lenet5.yaml --run_name emnist_letters_lenet5_seed7
```

## CIFAR10 Grayscale

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/cifar10_gray_learnable_moe_E9_complex.yaml --run_name cifar10_gray_learnable_moe_E9_complex_seed7
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/cifar10_gray_fixed_moe_E9_complex.yaml --run_name cifar10_gray_fixed_moe_E9_complex_seed7
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/cifar10_gray_d2nn_matched_params.yaml --run_name cifar10_gray_d2nn_matched_seed7
```

```powershell
python single_task/scripts/train_single_task.py --config single_task/configs/cifar10_gray_lenet5.yaml --run_name cifar10_gray_lenet5_seed7
```

## Rebuild All Master Tables

```powershell
python single_task/scripts/build_single_task_tables.py --runs_dir single_task/runs --out_dir single_task/results
```

## Notes

- `readout.dropout` is electronic readout dropout.
- `regularization.phase_dropout` is optical phase-layer dropout.
- `fixed_route_moe` freezes prompt amplitude and prompt phase-bias parameters.
- `learnable_route_moe` trains prompt amplitude logits and phase biases.
- `general_d2nn` has no prompt and no expert routing.
- `lenet5` is an electronic baseline and does not save optical light-field propagation figures.


## MultiGPU

```
CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=1

watch -n 1 nvidia-smi
```
