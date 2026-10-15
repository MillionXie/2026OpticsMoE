# Targeted train-only objective and quantization diagnosis (2026-10-15)

Human now explicitly requests active expert diagnosis and strategy changes after
SAM/dropout plateaus. New199 source197+driver/protocol. GPU4 only, non-A100,
2GiB+800MiB spare, no foreign process changes. Old197 sources/claims/models frozen.
Exclusive new smoke run caltech10_targeted_train_diagnosis_20261015, once only.
No Adam updates or test arrays/forward; two fixed TRAIN macro16 groups per depth,
not test-informed class filtering. Use independently audited selected models,
report classification/capture/phase-smooth gradient magnitudes and cosines plus
actual8bit phase-gray change fraction for global SAM vector radii.05/.5/2.0.
Each perturbation is measured then parameters restored exactly; no evaluation
metrics or optimization at perturbed points. Radius sweep is a training-physics
diagnostic, not model/test candidate selection. Save actual source/model/parent
SHA, gradients aggregate, exact restoration, no raw images/IDs Git. This evidence
will determine a separate frozen next strategy, not silently edit old experiments.
All old results and accepted models preserved; frozen100 controls/L3 unchanged.
