# LGVQ Temporal consistency 0.8044

> Repository consolidation note (2026-10-07): this is a historical internal
> review/export entry, not the maintained training entry. Start daily work from
> [`LightGenV2 T06`](../../../LightGenV2/tasks/t06_video_quality_assessment/README.md).
> The original `teacher_release` and `teacher_release_final` record SRCC
> **0.8043868643**. The later `teacher_release_final_v2` uses 8 µm propagation-grid
> resampling and detector-perturbation code and records **0.8022806420**, despite
> sharing the checkpoint. Do not assign the original score to that changed
> runtime, or infer runtime identity from checkpoint SHA alone. The original
> files and verification reports below remain historical evidence; see the
> [package identity audit](../../../maintenance/storage/T06_REVIEW_PACKAGE_IDENTITY_20261006.json).
> Source consolidation does not re-evaluate either result or replace the sealed
> LightGenV2 model. Public redistribution remains unapproved.

This directory is the standalone review package for the **16 videos × 4 frames**
LightGen optical/electronic Temporal-VQA model.  It is not the single-video
Temporal-36 experiment and it is not the frozen-Qwen baseline.

For the exact local locations of the unpacked checkpoint, sendable ZIP,
inference command, and training entry, start with
[`START_HERE_CN.md`](START_HERE_CN.md).  The unpacked teacher package is under
`teacher_release_final/lgvq_temporal_08044/`; its checkpoint is already present at
`weights/best_checkpoint.pt`.

## Reported result

| item | value |
|---|---:|
| test videos | 558 |
| physical fields | 35 |
| SRCC | **0.8043868643** |
| KRCC | 0.5968244684 |
| PLCC | 0.8180329374 |
| RMSE | 7.9910755 MOS |
| MAE | 5.9920588 MOS |

The last field contains 14 valid videos and two repeated padding slots.  Padding
participates in the coherent field but is excluded from all metrics.  The
packaged result was selected using periodic test-set SRCC at epoch 30; no
validation split was used.  This selection limitation must remain visible in
papers and releases.

## What is independent

- `runtime/` contains every Python module imported by the model.  It never
  imports the surrounding `LightGenV2`, `experiments`, or another task.
- The historical module names below `runtime/` are retained only to preserve
  the checkpoint/runtime identity.
- `simulate.py` adds only this directory's `runtime/` to `sys.path` and can be
  launched with Python isolated mode (`-I`).
- No server absolute path is required for fixed-weight reproduction.
- The teacher delivery ZIP includes the exact checkpoint and all 35 frozen
  test fields, so it does not download Qwen, videos, or datasets at runtime.

## Quick start

Python 3.11 and PyTorch 2.6.0 were used for the reference run.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# Integrity-only check; does not require a GPU.
python verify_release.py

# One-field smoke test. This is not a reportable full-test result.
python -I simulate.py --device cuda --fields 1 --output runs/smoke.json

# Formal 558-video evaluation.
python -I simulate.py --device cuda --fields 0 --output runs/full_test.json
```

The review source tree keeps generated binary assets separate.  This workspace
also contains an unpacked, directly runnable teacher package under
`teacher_release_final/lgvq_temporal_08044/` and the sendable ZIP under `releases/`.
Rebuild details remain documented in [COMMAND.md](COMMAND.md).

## Architecture boundary

Each physical field places 16 unrelated videos in a 4×4 grid and uses four
frames per video.  The student consumes frozen Qwen-front tokens plus a
14-channel quality feature bank.  It performs six shared coherent propagations:

1. frame optical router;
2. frame Top-2 expert propagation;
3. frame global propagation;
4. video optical router;
5. video Top-2 expert propagation;
6. video global propagation.

The trainable student contains no Transformer or attention block.  It does
contain electronic adapters, residual routes, nonlinear CCD readout, four
same-scale optical/electronic fusion gates, and a temporal regression head.
See [MODEL_CARD.md](MODEL_CARD.md) for the scientific/release boundary and
[ARCHITECTURE_AUDIT_CN.md](ARCHITECTURE_AUDIT_CN.md) for the checked 10 cm
physics contract, 17 µm-to-8 µm device mapping, noise-model boundary, and the
proposed source-code architecture.

## Provenance

- training run: `multivideo16x4_rank_s163`
- training-record commit: `ddc0d71abb79185b506e8db979b38695e2dffa79`
- pinned reproduction runtime: `8e869473787f4ffceb2a6a77f4430b94c206f459`
- checkpoint SHA-256:
  `5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c`
- original source archive SHA-256:
  `8421e7418ade4c9ec59f924cdad8e274db4100600c001cf802a8291ab7a67aff`

The rebuilt package was independently evaluated on an RTX 4090 with Python
3.11.15 and PyTorch 2.6.0+cu124.  All 558 predictions and all five metrics
matched the packaged reference exactly; see [verification](verification/README.md).

## Public-release status

This is an internal, open-source-ready review package.  A public repository
still requires the copyright holder to choose a license and approve the
redistribution of the checkpoint and frozen Qwen-derived inputs.  No license is
silently assigned here; see [OPEN_SOURCE_CHECKLIST.md](OPEN_SOURCE_CHECKLIST.md).
