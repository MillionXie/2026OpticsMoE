# Dataset-Switching Visualization

These scripts read either `dataset_switching/results/master_*.csv` or one run
directory under `dataset_switching/runs/<run_id>/`.

Main plots:

- `plot_task_accuracy.py`: per-task validation/test accuracy.
- `plot_prompt_swap_matrix.py`: prompt swap accuracy matrix.
- `plot_expert_usage_heatmap.py`: task x expert prompt power or entrance energy.
- `plot_prompt_history.py`: prompt amplitude or power over epochs.
- `plot_independent_vs_moe.py`: shared MoE vs independent D2NN comparison.
- `make_dataset_switching_report.py`: compact report wrapper.

Example:

```powershell
python dataset_switching/visualization/plot_prompt_swap_matrix.py --run_dir dataset_switching/runs/dswitch_mnist_fashion_emnist_letters_learnable_E9_seed7 --out_dir dataset_switching/figures/prompt_swap --name prompt_swap
```
