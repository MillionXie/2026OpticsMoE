# P12 fixed-feedback mechanism audit commands

This is a zero-training, post-hoc audit of the existing P12 four-group experiment.
It does not add a fifth main-table group and it does not replace the exact BP used
by the electronic residual or downstream head.

## What must exist first

For every requested task/seed, all four formal runs must be complete: the selected
50-epoch NoFT/common checkpoint and the BP, FA-pretrained, and FA-random 50-epoch
adaptation endpoints. Run the audit from the same clean code/config lineage used
for those jobs. The loader rejects mismatched task, seed, split manifest, P11 source
SHA, source phases, common-start SHA, frozen stem, feedback manifest, config digest,
implementation digest, checkpoint shape, or P11 architecture signature.

The audit writes only below a separate
`runs/p12_downstream_fa_50e_mechanism/` tree. It never mutates the formal runs.

## Recommended order

From repository root on an idle GPU:

```bash
P12_MECHANISM_GPU=4 \
  bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_mechanism_audit.sh smoke

P12_MECHANISM_GPU=4 \
  bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_mechanism_audit.sh pilot
```

`smoke` evaluates one test batch per state and is only an integration check.
`pilot` is the preregistered minimum: Caltech101 and ISIC2016, seed 2026, selected
best endpoints. Each task/seed/endpoint has 40 unique fresh-model states, so the
pilot performs 80 full test-set evaluations.

If formal training was executed from a locked worktree but the audit code is in
a separate derived worktree, keep the two roots explicit:

```bash
P12_REPO_ROOT=/path/to/derived-audit-worktree \
P12_FORMAL_REPO_ROOT=/path/to/locked-training-worktree \
P12_MECHANISM_GPU=4 \
  bash /path/to/derived-audit-worktree/experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_mechanism_audit.sh smoke
```

`P12_FORMAL_REPO_ROOT` reconstructs the original absolute Settings and
implementation digest. The formal checkpoint/result tree remains read-only.

Only after the pilot is interpretable:

```bash
# Checks whether validation selection changes the mechanism story.
P12_MECHANISM_GPU=4 \
  bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_mechanism_audit.sh selection

# All tasks and all three seeds, best endpoints (360 evaluations).
P12_MECHANISM_GPU=4 \
  bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_mechanism_audit.sh full

# Optional, expensive selection-sensitivity expansion (720 evaluations).
P12_MECHANISM_GPU=4 \
  bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_mechanism_audit.sh full-selection
```

The script fails closed when its destination already contains a complete result.
Use a new `P12_MECHANISM_OUTPUT_ROOT` for a deliberate rerun; do not overwrite a
reported audit silently.

## Outputs and reading order

Each `<task>/seed_<seed>/<best|last>/` directory contains:

- `mechanism_result.json`: complete identities, state metrics, links, and summaries.
- `states.csv`: one row per unique P/E/H hybrid.
- `shapley.csv`: exact 8-state factorial Shapley values for phase (P), reusable
  electronics (E), and temporary task head (H).
- `directed_swaps.csv`: BP/FAP/FAR phase and electronics donor-to-recipient swaps,
  including deltas from common and each recipient's own endpoint.
- `phase_depth.csv`: stage 1--7 versus stage 8 phase-reset counterfactuals.
- `mechanism_manifest.json`: strict source identity and partition audit.

For accuracy/IoU/PCK, a positive benefit delta is better. For loss, the CSV flips
the sign only in the explicitly named `benefit_*` columns; raw deltas remain
literal. A transport ratio is undefined (`null`) when the recipient's own block
has effectively zero effect. Shapley and swap findings are functional
counterfactual evidence, not proof of a unique causal decomposition in this
nonlinear network.

## Scientific boundary

P12 FA changes only the optical inter-stage backward connector. The local gradient
of the current physical phase, the electronic residual path, and the temporary
task head still use exact backpropagation. This is the scientifically appropriate
hybrid-hardware interpretation: electronics can store activations and compute
ordinary gradients, while the hard-to-invert optical propagation receives fixed
feedback. Results must therefore be described as **optical-path fixed feedback**,
not as global feedback alignment for every model parameter.
