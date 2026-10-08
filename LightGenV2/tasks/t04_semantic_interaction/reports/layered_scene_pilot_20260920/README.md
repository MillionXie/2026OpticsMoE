# Layered OpenMoji pilot (2026-09-20)

This is a **10-epoch connectivity and learning-trend pilot**, not the final
paper number.  It validates the approved proportional, layered scene renderer
before the separate 100-epoch run.

## Frozen protocol

- 5,000 training scenes and 1,000 disjoint test scenes; no validation split.
- Add/replace/move/remove: 1,250 training and 250 test scenes each.
- Four scene families: home, grove, street and still life.
- Official OpenMoji 17.0.0 SVG artwork, category-specific physical size,
  bottom anchors and explicit painter order; an object must retain at least
  70% visible alpha.
- The prediction contract remains the established 6x6 semantic anchor grid.
  Large objects may span several cells visually but still occupy one anchor.
- Frozen Qwen tokenizer and `embed_tokens` lookup only; no Transformer or
  attention block in the student.
- Language and vision each use an optical Top-2 router, four optical experts
  and one global optical phase; the fusion is `(1-alpha)E + alpha O`, with
  alpha constrained to `[0.4001, 0.95]` and initialized at 0.60.
- Simulated unmodulated/zero-order intensity is sampled from 20%-30%.

Training source commit: `24e77302a6df8d582f4a9ffaf62e7165840c577b`.
The portable gallery fix was evaluated at `642363f1f78db50e216bdcb9fffa7744106c95dd`.

## Pilot result

The periodic test improved from 10.60% changed-cell accuracy at epoch 1 to
30.60% at epoch 5 and 36.00% at epoch 10.  The loss decreased from 3.7049 to
2.1875.  The selected checkpoint is epoch 10.

| Metric | Normal light+electronic | Same checkpoint, optics removed | Difference |
|---|---:|---:|---:|
| Changed-cell accuracy | 0.3600 | 0.2365 | +12.35 percentage points |
| Cell accuracy | 0.9285 | 0.9194 | +0.91 percentage points |
| Edit-grid IoU | 0.2495 | 0.2161 | +0.0334 |
| Scene exact match | 0.0370 | 0.0350 | +0.0020 |

Changed-cell accuracy by operation at epoch 10:

| Add | Replace | Move | Remove |
|---:|---:|---:|---:|
| 0.092 | 0.228 | 0.332 | 0.788 |

The 10-epoch result is under-trained, especially for add/replace, but the
strict same-weight ablation already shows a positive optical contribution.

## Router and phase audit

- Language expert selection shares: `[0.2440, 0.2380, 0.2400, 0.2780]`;
  inverse-Simpson effective experts: 3.983/4.
- Vision expert selection shares: `[0.2515, 0.2460, 0.2550, 0.2475]`;
  inverse-Simpson effective experts: 3.999/4.
- No expert is unused and both router audits pass the configured acceptance
  limits.
- At epoch 10, optical phase RMS displacement from initialization is roughly
  0.197-0.301 rad for language phases and 0.226-0.328 rad for vision phases.
  Non-zero final-batch gradients are recorded for every router, expert and
  global phase.  The masks therefore genuinely trained.

Source-scene occlusion did not damage changed-cell accuracy in this short run:
0.3594 for the 761 source-occluded scenes versus 0.3619 for the 239 clear
scenes.  Target occlusion remains harder (0.3133 versus 0.4606), which is a
useful target for the formal run.

## Artifacts

- `selected_checkpoint_test_evaluation.json`: complete normal evaluation.
- `same_checkpoint_remove_optical.json`: strict no-retraining optical ablation.
- `layered_subgroup_metrics.json`: family and occlusion subgroups.
- `phase_training_audit.json`: per-epoch phase movement and gradient audit.
- `student_architecture.json`: executable architecture contract.
- `dataset_summary.json` and `split_contract.json`: data protocol and hashes.
- `training_history.csv`: loss/test trajectory.
- `*_examples.png`: unselected test examples with actual source/target scenes,
  predicted semantic anchors and error anchors.
- `best_phase_overview.png`: selected phase overview.

The server checkpoint remains in the task-local pilot run directory.  SHA256:

- `best_checkpoint.pt`: `935daa34d944156cd8d7fcbd159a45f5f72748db3d1495be3373f6c9705df92c`
- `last_checkpoint.pt`: `643026a491dd4c5aed2a51426b9c1c1aa71893f70132bd47c24d7f856218e376`

Selection is deliberately test-biased under the current project protocol:
test is evaluated at epoch 1, every five epochs and the final epoch, and the
best accepted router checkpoint is retained.  These numbers must not be
presented as an untouched final-test estimate.
