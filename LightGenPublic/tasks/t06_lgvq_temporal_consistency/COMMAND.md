# Commands

Run commands from this directory.  Do not put the parent repository on
`PYTHONPATH`; isolated mode is part of the independence check.

## Verify an extracted teacher package

```bash
python verify_release.py
python -I simulate.py --help
```

## Reproduce the fixed result

```bash
# One of 35 fields, 16 videos. Smoke only.
python -I simulate.py --device cuda --fields 1 --output runs/smoke.json

# All 35 fields / 558 valid videos. This is the formal command.
python -I simulate.py --device cuda --fields 0 --output runs/full_test.json
```

`simulate.py` checks every packaged file before importing the model.  It then
strictly loads the state dictionary, compares every prediction with the
packaged reference, and gates SRCC/KRCC/PLCC/RMSE/MAE for a full run.

## Build the corrected teacher ZIP

The builder consumes the older pinned archive, verifies its SHA-256, removes
cached bytecode and hardware-only wrappers, adds the missing simulation entry
point and regenerates the file manifest.

```bash
python tools/build_teacher_release.py \
  --source-archive /path/to/20260914_shs_temporal08044.zip \
  --output releases/lgvq_temporal_08044_teacher_20260922.zip
```

The source archive must have SHA-256
`8421e7418ade4c9ec59f924cdad8e274db4100600c001cf802a8291ab7a67aff`.
The builder refuses an existing output instead of overwriting it.

## Training-source entry

The training/evaluation implementation is included under `runtime/` and can be
invoked without the parent repository:

```bash
PYTHONPATH=runtime python -m \
  LightGenV2.tasks.t06_video_quality_assessment.multivideo \
  --config configs/temporal_16x4_s163.yaml --phase smoke
```

Full retraining additionally requires the files listed in
`TRAINING_ASSETS.md`.  The published 0.8044 claim is a fixed-checkpoint
reproduction claim; retraining is not promised to reproduce the exact epoch or
floating-point trajectory.
