# Dataset Switching Commands

Run these commands from `opticalmoe_experiments/`.

## Runtime Notes

- Formal configs default to `num_workers=16`, `pin_memory=auto`, `persistent_workers=true`, and `prefetch_factor=4`.
- On Linux servers this is the recommended starting point. If CPU or RAM is saturated, reduce `num_workers` to `8` or `4`.
- On Windows or while debugging, set `num_workers=0`.
- `--smoke_test` automatically forces `num_workers=0`, `persistent_workers=false`, and `prefetch_factor=null`.
- New runs use `fast120_520`: canvas `520 x 520`, input/expert `120`, and center `450 x 450` prompt/global-FC window; padding is transparent and not trainable.
- Explicit `fair134_1000` configs remain available for legacy reproduction.
- `sampling_protocol.enabled=false` runs each official dataset split. `enabled=true` makes `total_size` mean train+val+test for each task dataset.
- For example, `total_size=10000`, `train_test_ratio=[4,1]`, `val_split=0.1` gives about `train=7200`, `val=800`, `test=2000`.
- Use `max_train_samples`, `max_val_samples`, and `max_test_samples` when you want exact split caps.
- `sequential_backward=true` means one update forwards/backwards each task sequentially, then runs one shared optimizer step. This saves GPU memory.
- `balanced_sampling=true` keeps task sampling balanced when datasets have different sizes.
- `loss_reduction=mean` averages task losses after weighting.

## Audit Dataset Config Fields

```powershell
python scripts/audit_dataset_config_fields.py
```

Audit task-specific head fields:

```powershell
python scripts/audit_head_config_fields.py
```

## Three-Task Learnable MoE Smoke

```powershell
python dataset_switching/scripts/train_dataset_switching.py --config dataset_switching/configs/mnist_fashion_emnist_letters_learnable_moe_E9_complex.yaml --run_name dswitch_mnist_fashion_emnist_letters_learnable_smoke --epochs 1 --smoke_test --device cuda
```

## Three-Task Learnable MoE

```powershell
python dataset_switching/scripts/train_dataset_switching.py --config dataset_switching/configs/mnist_fashion_emnist_letters_learnable_moe_E9_complex.yaml --run_name dswitch_mnist_fashion_emnist_letters_learnable_E9_seed7 --device cuda
```

## Three-Task Fixed Uniform MoE

```powershell
python dataset_switching/scripts/train_dataset_switching.py --config dataset_switching/configs/mnist_fashion_emnist_letters_fixed_uniform_moe_E9_complex.yaml --run_name dswitch_mnist_fashion_emnist_letters_fixed_uniform_E9_seed7 --device cuda
```

## Three-Task Shared D2NN

```powershell
python dataset_switching/scripts/train_dataset_switching.py --config dataset_switching/configs/mnist_fashion_emnist_letters_shared_d2nn.yaml --run_name dswitch_mnist_fashion_emnist_letters_shared_d2nn_seed7 --device cuda
```

## Three-Task Independent D2NN

```powershell
python dataset_switching/scripts/run_independent_baseline.py --config dataset_switching/configs/mnist_fashion_emnist_letters_independent_d2nn.yaml --run_name independent_d2nn_mnist_fashion_emnist_letters_seed7 --device cuda
```

The combined command trains three separate D2NN models sequentially. It does
not share optical or electronic parameters across datasets.

## Independent D2NN, One Dataset Per Process

MNIST:

```powershell
python dataset_switching/scripts/run_independent_baseline.py --config dataset_switching/configs/mnist_independent_d2nn_canvas400_grid220.yaml --device cuda
```

Fashion-MNIST:

```powershell
python dataset_switching/scripts/run_independent_baseline.py --config dataset_switching/configs/fashionmnist_independent_d2nn_canvas400_grid220.yaml --device cuda
```

EMNIST-letters:

```powershell
python dataset_switching/scripts/run_independent_baseline.py --config dataset_switching/configs/emnist_letters_independent_d2nn_canvas400_grid220.yaml --device cuda
```

The same combined config can also be filtered:

```powershell
python dataset_switching/scripts/run_independent_baseline.py --config dataset_switching/configs/mnist_fashion_emnist_letters_independent_d2nn.yaml --task mnist --run_name independent_mnist_grid220_seed7 --device cuda
```

On a two-GPU Linux server, launch two independent processes:

```bash
CUDA_VISIBLE_DEVICES=0 python dataset_switching/scripts/run_independent_baseline.py --config dataset_switching/configs/mnist_independent_d2nn_canvas400_grid220.yaml --device cuda
CUDA_VISIBLE_DEVICES=1 python dataset_switching/scripts/run_independent_baseline.py --config dataset_switching/configs/fashionmnist_independent_d2nn_canvas400_grid220.yaml --device cuda
```

Run EMNIST after either GPU becomes available. When multiple runs finish
concurrently, rebuild the master tables once at the end.

## Prompt Swap Evaluation

```powershell
python dataset_switching/scripts/run_prompt_swap_eval.py --run_dir dataset_switching/runs/dswitch_mnist_fashion_emnist_letters_learnable_E9_seed7 --checkpoint best.pt --device cuda
```

## Rebuild Master Tables

```powershell
python dataset_switching/scripts/build_dataset_switching_tables.py --runs_dir dataset_switching/runs --out_dir dataset_switching/results
```

## Plot Prompt Swap Matrix

```powershell
python dataset_switching/visualization/plot_prompt_swap_matrix.py --run_dir dataset_switching/runs/dswitch_mnist_fashion_emnist_letters_learnable_E9_seed7 --out_dir dataset_switching/figures/prompt_swap --name prompt_swap
```
