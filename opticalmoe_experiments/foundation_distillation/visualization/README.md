# Distillation Visualization

- `plot_distillation_curves.py`: weighted loss, accuracy, and feature cosine over epochs.
- `plot_feature_similarity.py`: train/validation student-teacher cosine similarity.
- `plot_confusion_matrix.py`: confusion matrix from a run CSV.
- `plot_expert_usage.py`: final normalized prompt power by expert.

The training script generates these primary figures automatically under each run's `figures/` directory. Light-field, prompt, and phase-mask images use the shared visualization implementation.

