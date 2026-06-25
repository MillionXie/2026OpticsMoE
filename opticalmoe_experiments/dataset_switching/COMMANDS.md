# Dataset Switching Commands

Run these commands from `opticalmoe_experiments/`.

## Runtime Notes

- Formal configs default to `num_workers=16`, `pin_memory=auto`, `persistent_workers=true`, and `prefetch_factor=4`.
- On Linux servers this is the recommended starting point. If CPU or RAM is saturated, reduce `num_workers` to `8` or `4`.
- On Windows or while debugging, set `num_workers=0`.
- `--smoke_test` automatically forces `num_workers=0`, `persistent_workers=false`, and `prefetch_factor=null`.
- The optical canvas is `1000 x 1000`, but the default trainable global FC phase window is center `600 x 600`; the padding is transparent and not trainable.

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
