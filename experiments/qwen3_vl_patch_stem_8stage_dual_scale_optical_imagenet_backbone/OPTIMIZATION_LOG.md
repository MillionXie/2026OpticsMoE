# P10 implementation and verification log

## 2026-08-24 implementation

- Created an independent experiment; the active P09 process and run directory
  were not modified or stopped.
- Subclassed the locked P09 backbone and replaced only the fixed propagators.
- Assigned stages `[local 5 mm, global 50 mm] x 4` while retaining eight phase
  planes and all P09 electronics exactly.
- Added runtime schedule and parameter reporting with zero trainable-parameter
  increase over P09.
- Added a unique persistent P10 architecture signature and model metadata in
  exported backbone checkpoints to reject cross-variant checkpoint mistakes.
- Prepared a P09-matched 90-epoch config and a formal launcher that refuses to
  start unless two GPU indices are explicitly supplied. It was not launched.

## Verification

- Python bytecode compilation: passed locally.
- Git synchronization: commit `422c8bcb` was fast-forwarded on the server and
  pushed to `origin/main` before testing.
- Shell syntax checks for every command script: passed on the server.
- Combined P10/P11 unit suite: `14 passed in 5.80s` on the server. P10 checks
  the alternating schedule, parameter budget, distinct transfer functions,
  P09-matched trainable initialization and strict cross-variant checkpoint
  rejection.
- Measured impulse-response r90: `5.83095 px` at 5 mm and `58.18075 px` at
  50 mm under the configured periodic 224x224 angular-spectrum model.
- GPU smoke: completed on physical GPU 1 with batch size 4, three training
  batches and two validation batches. This is a functional test, not a
  performance result.
- Smoke throughput: `7.862 samples/s`; peak allocated/reserved memory:
  `423.1/442.0 MiB` for this process.
- All eight phase-gradient norms were finite and nonzero (range
  `0.04443-0.32028`), and mean absolute phase motion after three updates was
  `0.01268 rad`.
- Optimizer-state audit after the smoke confirmed finite nonzero updates for
  phase `8/8`, adapter `4/4`, residual/mixer `184/184` and head `7/7`
  parameter tensors.
- Smoke artifacts:
  `runs/gpu_smoke_20260824_014518/`; use `metrics/latest.json` for the
  post-update diagnostics. The tiny 12-train/16-validation sample metrics have
  no accuracy interpretation.
- Formal training: intentionally not started.
