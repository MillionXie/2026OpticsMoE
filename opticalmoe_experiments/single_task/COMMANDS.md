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

## Audit Dataset Config Fields

After editing YAML files, run:

```powershell
python scripts/audit_dataset_config_fields.py
```

## Dataset Size Quick Edits

Run full official data:

```yaml
sampling_protocol:
  enabled: false
  total_size: null
```

Run an approximate 10k-sample experiment:

```yaml
val_split: 0.1
sampling_protocol:
  enabled: true
  total_size: 10000
  train_test_ratio: [4, 1]
  class_balanced: true
```

This gives about `train=7200`, `val=800`, `test=2000`.

Directly cap split sizes:

```yaml
max_train_samples: 5000
max_val_samples: 1000
max_test_samples: 1000
```

## Notes

- `readout.dropout` is electronic readout dropout.
- `regularization.phase_dropout` is optical phase-layer dropout.
- `fixed_route_moe` freezes prompt amplitude and prompt phase-bias parameters.
- `learnable_route_moe` trains prompt amplitude logits and phase biases.
- `general_d2nn` has no prompt and no expert routing.
- `general_d2nn` is 5 center-window D2NN phase masks plus one center-window global FC phase mask.
- `canvas_size=1000` is the propagation canvas; the default active trainable optical window is center `600 x 600`, with transparent non-trainable padding outside.
- In D2NN configs, `target_local_phase_param_count` counts only the local D2NN masks; `expected_total_optical_param_count` also includes the `600 x 600` global FC mask.
- D2NN phase masks are saved under `figures/phase_masks/<epoch>/` as `d2nn_phase_layer_*.png`, `d2nn_all_phase_layers.png`, `global_fc_phase_window.png`, `global_fc_phase_region_on_canvas.png`, and `global_fc_phase.png`.
- MoE expert usage rows include fixed-validation-batch `expert_entrance_energy_ratio` and `expert_output_energy_ratio`.
- `lenet5` is an electronic baseline and does not save optical light-field propagation figures.
- `lenet5` adapts to the configured dataset `input_size` and does not save optical phase masks or optical energy rows.
- Linux servers should start with `num_workers=16`, `pin_memory=auto`, `persistent_workers=true`, `prefetch_factor=4`; reduce to `8` or `4` if CPU/RAM is saturated.
- On Windows or when debugging, use `num_workers=0`. All `--smoke_test` runs force `num_workers=0`, `persistent_workers=false`, and `prefetch_factor=null`.
- `sampling_protocol.enabled=false` means full official split; `enabled=true` means `total_size` is train+val+test for this run.
- `pin_memory=auto` enables pinned memory only when CUDA is available.
- MNIST/Fashion/KMNIST/EMNIST are already grayscale; setting `grayscale=false` does not reduce optical compute because the transform still emits `[1,H,W]`.

## Multi-GPU Notes

```bash
watch -n 1 nvidia-smi
htop #cpu
nproc
conda activate xml
cd xml_code/2026OpticsMoE/opticalmoe_experiments/
```

Run one experiment on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python single_task/scripts/train_single_task.py --config single_task/configs/mnist_learnable_moe_E9_complex.yaml --run_name mnist_moe_gpu0
```

Run another experiment on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python single_task/scripts/train_single_task.py --config single_task/configs/fashionmnist_learnable_moe_E9_complex.yaml --run_name fashion_moe_gpu1
```

## Visualization Commands

Rebuild master tables:

```powershell
python single_task/scripts/build_single_task_tables.py --runs_dir single_task/runs --out_dir single_task/results
```

Compare MNIST baseline training curves:

```powershell
python single_task/visualization/plot_training_curves.py --run_dirs single_task/runs/mnist_learnable_moe_E9_complex_seed7 single_task/runs/mnist_fixed_moe_E9_complex_seed7 single_task/runs/mnist_d2nn_matched_seed7 single_task/runs/mnist_lenet5_seed7 --metrics acc loss --show train val --mode overlay --out_dir single_task/figures/mnist_baselines --name mnist_baseline_training
```

Plot final accuracy from master tables:

```powershell
python single_task/visualization/plot_final_comparison.py --master_dir single_task/results --dataset mnist --x model_type --metric final_test_acc --out_dir single_task/figures/mnist_baselines --name mnist_final_accuracy
```

Plot training time:

```powershell
python single_task/visualization/plot_time_comparison.py --master_dir single_task/results --dataset mnist --unit min --out_dir single_task/figures/mnist_baselines --name mnist_training_time
```

Plot expert usage heatmap:

```powershell
python single_task/visualization/plot_expert_usage.py --master_dir single_task/results --dataset mnist --model_type learnable_route_moe --value normalized_prompt_power --out_dir single_task/figures/mnist_learnable_moe --name mnist_expert_usage
```

Generate a compact report:

```powershell
python single_task/visualization/make_single_task_report.py --master_dir single_task/results --dataset mnist --out_dir single_task/figures/reports/mnist_baselines --name mnist_baselines
```
