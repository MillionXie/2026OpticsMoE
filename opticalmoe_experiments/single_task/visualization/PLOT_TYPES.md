# Plot Types

## A. Training Curves

Use `plot_training_curves.py` to inspect convergence, train/validation divergence, and whether phase dropout changes validation behavior.

## B. Final Comparison

Use `plot_final_comparison.py` to compare `general_d2nn`, `fixed_route_moe`, `learnable_route_moe`, and `lenet5` across datasets or seeds.

## C. Time Comparison

Use `plot_time_comparison.py` to compare training cost and accuracy-time trade-offs.

## D. Generalization Gap

Use `plot_generalization_gap.py` to quantify overfitting. A large positive gap means training accuracy is much higher than validation accuracy.

## E. Expert Usage

Use `plot_expert_usage.py` to check whether MoE uses multiple experts or collapses to a small subset.

## F. Prompt History

Use `plot_prompt_history.py` to see how prompt amplitude and normalized prompt power evolve during training.

## G. Expert Ablation

Use `plot_expert_ablation.py` after running `single_task/scripts/run_expert_ablation.py`. It shows which experts matter most for accuracy.

## H. Optical Energy

Use `plot_optical_energy.py` to inspect energy leakage outside expert apertures and concentration into a few experts.

## I. Confusion Matrix

Use `plot_confusion_matrix.py` to inspect class-level mistakes.

