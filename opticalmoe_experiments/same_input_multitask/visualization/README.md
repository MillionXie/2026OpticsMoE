# Same-Input Multitask Visualization

These scripts plot diagnostics from `same_input_multitask/runs/<run_id>/` or
from rebuilt master tables in `same_input_multitask/results/`.

Main plots:

- `plot_task_accuracy.py`: train/val accuracy per task.
- `plot_same_input_predictions.py`: same input task-switching accuracy.
- `plot_prompt_swap_matrix.py`: fixed-readout prompt swap accuracy matrix.
- `plot_expert_usage_heatmap.py`: task-by-expert prompt power or energy.
- `plot_prompt_similarity.py`: task prompt similarity matrix.
- `plot_scaling_results.py`: stage 1/2/3 scaling summary.
- `make_same_input_report.py`: generate a small report folder with common plots.

Example:

```bash
python same_input_multitask/visualization/plot_prompt_swap_matrix.py \
  --master_dir same_input_multitask/results \
  --run_id simt_dsprites_shape_scale_learnable_E9_seed7 \
  --out_dir same_input_multitask/figures/reports/shape_scale
```
