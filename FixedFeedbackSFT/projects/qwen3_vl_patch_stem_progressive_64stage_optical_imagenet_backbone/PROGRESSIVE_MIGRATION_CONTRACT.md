# P13 progressive-growth migration contract

This document defines initialization only. It does not claim that formal
ImageNet training has run, and it does not turn the engineering GPU sweep into
a training result.

## Three independent stage properties

Every `ProgressiveOpticalStageSlot` exposes three explicit properties:

- `is_p11_mixer_anchor`: immutable architecture property. Exactly eight slots
  own the width-96 P11 electronic mixers.
- `is_carried_from_parent`: this target slot contains one complete stage copied
  from the immediately preceding source depth.
- `is_newly_inserted`: this slot was inserted by the current growth transition.

Mixer ownership and growth provenance are not interchangeable. For example, a
16-stage identity-electronic slot becomes parent-carried when growing to 32 and
must remain active at alpha one even though it is not a mixer anchor.

The persistent `growth_parent_stage_indices` vector records this transition.
Each non-negative entry is the ordered source-stage index copied into that
target slot; `-1` marks a newly inserted slot. Non-negative entries must be
exactly `0..source_depth-1`, each once and in order.

Only `new_slots()` participate in the epsilon-to-one alpha schedule. Carried
slots are locked to alpha one. Alpha zero uses an exact Python bypass; alpha one
executes `Stage(x)` directly rather than evaluating
`x + 1 * (Stage(x) - x)`, avoiding numerical round-trip error.

## P11 epoch-88 to P13

Formal continuation initialization must call:

```python
migrate_strict_p11_training_checkpoint(
    target,
    backbone_checkpoint,
    training_checkpoint,
)
```

`backbone_checkpoint` is the official P11 `backbone.pt` and
`training_checkpoint` is the matching epoch-88 `best.pt`. Before mutating the
target, migration verifies:

- P11 architecture signature, model report and frozen-stem SHA-256;
- `best.pt.epoch == backbone.pt.best_epoch`;
- identical config digests;
- identical non-readout key sets and tensor values;
- strict full P11 state loading, including the ImageNet readout.

The frozen stem, adapter, eight complete optical/mixer stages and readout are
then copied. Consequently alpha-zero P13 preserves both final optical features
and ImageNet logits. The older `migrate_strict_p11_checkpoint()` remains for
reusable-backbone/engineering use and deliberately does not migrate the head.

## P13 depth-to-depth growth

`migrate_strict_progressive_checkpoint(target, checkpoint)` accepts only:

```text
16 -> 32 -> 64 -> 100
```

The source payload format is
`p13-progressive-imagenet-training-v1` and must include:

- `checkpoint_role: best_full_depth`;
- complete `model` state, including readout;
- `model_config` and `model_report`;
- `stem_checkpoint_sha256`;
- positive `epoch`, non-empty `config_digest`, and alpha-one `depth_alpha`
  metadata consistent with both the report and serialized model state.

The source must already be at alpha-one full depth. Source token/channel pairs
are embedded monotonically into the target pair sequence. The four P11 mixer
anchors are pinned to their target-depth mixer anchors, while intermediate
pairs are distributed inside the three anchor intervals. Every source stage is
copied once, in order and with matching optical axis/mixer structure. All other
target stages keep deterministic target initialization and alpha zero.

After either migration path, the full-depth feedback source is recaptured from
the target's actual current physical phases. Thus every target stage contributes
one distinct connector; no eight-stage phase sequence is tiled through a deeper
model. Runtime feedback mode is reset explicitly to `bp_current` by the existing
feedback contract.

## Optimizer-facing groups

The model exposes five disjoint groups whose union is every trainable parameter:

1. `carried_phase_parameters()`
2. `new_phase_parameters()`
3. `carried_electronic_parameters()` -- adapter plus all carried stage
   electronics, excluding readout
4. `new_electronic_parameters()` -- electronics of newly inserted slots only
5. `head_parameters()`

`phase_motion()` reports circular physical-phase displacement for training
diagnostics.

## Automated coverage

`tests/test_progressive_migration.py` covers:

- persistent carried/new provenance and disjoint parameter groups;
- alpha scheduling only newly inserted slots;
- P11 backbone/best cross-check and exact feature/logit preservation;
- rejection of a mismatched P11 non-head tensor;
- monotone pair mappings for all three supported transitions;
- strict structural migration for 16->32, 32->64 and 64->100;
- exact feature/logit preservation for all three transitions;
- full-depth feedback recapture;
- rejection of a source checkpoint before alpha reaches one.
