# Single-Task Visualization

This folder contains paper-style plotting scripts for `opticalmoe_experiments/single_task`.

The scripts do not train models and do not modify checkpoints. They only read saved run outputs or master tables and write figures plus the exact CSV data used to draw each figure.

## Data Sources

You can plot from one or more run directories:

```powershell
python single_task/visualization/plot_training_curves.py --run_dirs single_task/runs/runA single_task/runs/runB --metrics acc loss --out_dir single_task/figures/custom_plots --name example
```

Or from rebuilt master tables:

```powershell
python single_task/visualization/plot_final_comparison.py --master_dir single_task/results --dataset mnist --x model_type --metric final_test_acc --out_dir single_task/figures/mnist_baselines --name mnist_final_accuracy
```

The common inputs are:

- `run_dir/metrics/epoch_metrics.csv`
- `run_dir/metrics/final_metrics.json`
- `run_dir/metrics/confusion_matrix.csv`
- `run_dir/summary_for_master/*.json`
- `single_task/results/master_*.csv`

## Scripts

- `plot_training_curves.py`: train/validation accuracy and loss curves.
- `plot_final_comparison.py`: final accuracy, loss, and parameter comparisons.
- `plot_time_comparison.py`: total training time, average epoch time, and accuracy-time scatter.
- `plot_generalization_gap.py`: overfitting gap from final metrics or epoch curves.
- `plot_confusion_matrix.py`: normalized or raw confusion matrix for one or more runs.
- `plot_expert_usage.py`: MoE expert usage heatmaps from prompt power or expert energy.
- `plot_prompt_history.py`: prompt amplitude or normalized prompt power over epochs.
- `plot_expert_ablation.py`: expert ablation heatmaps after running expert ablation diagnostics.
- `plot_optical_energy.py`: optical energy and leakage across propagation stages.
- `make_single_task_report.py`: generate a compact figure set for selected runs.

Each figure is saved as PNG/PDF/SVG by default, and each script also writes `<figure_name>_plot_data.csv`.

## Example Questions

- Training curves: did the model converge, and when did it overfit?
- Final comparison: does learnable MoE outperform fixed MoE and matched D2NN?
- Time comparison: how expensive is each model relative to its accuracy?
- Expert usage: did the router collapse to one expert or distribute power?
- Prompt history: did the learnable route stabilize?
- Optical energy: where does light leak outside the expert apertures?

