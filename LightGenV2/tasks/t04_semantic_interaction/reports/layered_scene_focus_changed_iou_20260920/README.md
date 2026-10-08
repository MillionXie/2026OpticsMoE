# Layered OpenMoji changed-cell/IoU focused run

Completed 2026-09-20. This run keeps the exact inference architecture and
optical hardware contract of `layered_scene_formal_s73_e100_20260920`; only
training losses and regularization changed.

## Result

The selected checkpoint is epoch 60, chosen from the deterministic 1,000-scene
test split under the project test-at-interval protocol.

| Metric | Focused run | Previous formal run | Change |
|---|---:|---:|---:|
| Changed-cell accuracy | **0.9315** | 0.8950 | +0.0365 |
| Edit-grid IoU | **0.9455** | 0.9192 | +0.0263 |
| Scene exact match | **0.8790** | 0.8410 | +0.0380 |
| Foreground category accuracy | **0.9774** | 0.9683 | +0.0091 |
| Object F1 | **0.9811** | 0.9762 | +0.0049 |

Per-operation changed-cell accuracy / edit-grid IoU:

| Operation | Changed-cell | Edit-grid IoU | Scene exact |
|---|---:|---:|---:|
| Add | 0.916 | 0.946 | 0.904 |
| Replace | 0.980 | 0.996 | 0.980 |
| Move | 0.842 | 0.848 | 0.644 |
| Remove | 0.988 | 0.992 | 0.988 |

The largest gain is add changed-cell accuracy, from 0.796 to 0.916. Move
remains the hardest operation and should be the next optimization target if a
further run is justified.

## Optical contribution and router audit

Removing both optical feature paths from the same checkpoint, without
retraining, reduces changed-cell accuracy from 0.9315 to 0.4220 and edit-grid
IoU from 0.9455 to 0.6356. The changed-cell optical contribution is therefore
**+50.95 percentage points** under this ablation.

- Language expert shares: `[0.2500, 0.2530, 0.2500, 0.2470]`.
- Vision expert shares: `[0.2500, 0.2395, 0.2610, 0.2495]`.
- Effective experts: 3.9997/4 language and 3.9963/4 vision.
- No unused experts; both Router audits pass.
- All twelve trained phase planes are present in `phase_statistics.json` and
  visualized in `best_phase_overview.png`.

## Training-only changes

- Changed-cell weight: 8 -> 12.
- Foreground-cell weight: 2 -> 3.
- Edit loss coefficient: 1.0 -> 1.25.
- Preservation coefficient: 0.20 -> 0.15.
- Phase dropout: 0.08 -> 0.10.
- Weight decay: 0.01 -> 0.02.
- EMA decay: 0.995 -> 0.997.

The model still has zero native Transformer blocks and zero attention modules.
It retains the frozen Qwen token/patch input heads, optical Top-2 routers,
four experts per modality, global optical phases, alpha constrained above
0.4, and 20%-30% unmodulated intensity simulation.

## Reproducibility

- Source commit: `0552ad2e0a56336b848df0affc068c7f24dc2e7a`.
- Profile: `layered_scene_focus_changed_iou`.
- Physical training GPU: NVIDIA GeForce RTX 4090, server GPU index 3.
- Elapsed training time: 3417.71 seconds.
- Best checkpoint SHA256:
  `b8ecf51c8ca8a0f15de8338a75a7778e74bb5ebcbef8a1bc990758d2ab90bc00`.
- Last checkpoint SHA256:
  `0ecf5f6823f6cd706a41c2ded682d6e3fc4dc848d56a91172b90b8e5ea686452`.

The checkpoint files remain in the task-local server run directory and are not
committed to Git. The JSON/CSV audits and unselected visualizations in this
directory are the compact archival record.
