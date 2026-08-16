# CIFAR-100-C optical fixed-feedback experiment

This independent experiment tests whether the adjoints of pretrained optical
operators remain useful fixed feedback connectors during small-drift optical
fine-tuning. It implements BP, fixed pretrained feedback, fixed random feedback,
and no fine-tuning from one shared checkpoint.

The formal model uses 20 OEO stages, 400 x 400 fields, 16 um pixels, 532 nm
light, and 5 cm propagation per stage. Each stage contains one CCD, one
non-affine LayerNorm, one ReLU and a trainable two-branch residual reload.

Formal training uses 80 pretraining epochs with a deterministic rotating 15,000
sample epoch (about 26.7 effective passes over the 45,000-image pretraining
split), followed by 50 full downstream fine-tuning epochs for each of three
matched seeds. The phase learning rate is 0.01 and the electronic learning rate
is 0.001. Pretraining uses batch 128 and every fine-tuning method uses batch 64.
All methods use AdamW with zero weight decay.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the exact feedback definition and
[commands/COMMANDS.md](commands/COMMANDS.md) for server commands.

Outputs are kept under this experiment's `runs/main` directory. They include
resolved configuration, data manifests, training CSV files, checkpoints,
gradient-to-BP diagnostics, endpoint drift statistics, phase masks, residual
weights, optical stage examples, confusion matrices, and aggregate comparison
figures.

The comparison phase also writes a publication-oriented, checkpoint-policy-aware
record to [results/main/RESULTS_AND_ANALYSIS.md](results/main/RESULTS_AND_ANALYSIS.md).
It separates fixed-budget task performance, validation-selected task performance,
and matched-epoch endpoint geometry; these quantities must not be mixed.
