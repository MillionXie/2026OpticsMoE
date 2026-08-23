# P11 implementation and verification log

## 2026-08-24 implementation

- Created an independent experiment without modifying or stopping active P09.
- Implemented axis-selective angular-spectrum propagation: token/y-only and
  channel/x-only operators whose orthogonal axes are mathematically relayed.
- Organized eight stages as four token-to-channel optical macro blocks.
- Added exact Qwen block-major to row-major permutation only around token-axis
  optics and an exact inverse before canonical electronic processing.
- Kept phase/random/feedback tensors in physical layout so the inherited BP,
  fixed-feedback and deployment-perturbation paths remain consistent.
- Reused P09's initialized residual gate and width-96 electronic mixer modules;
  no trainable parameters were added.
- Added a unique P11 architecture signature, explicit layout/schedule metadata
  and cross-variant strict-load rejection.
- Prepared a P09-matched formal config and an explicit-GPU guarded launcher. It
  was not launched.

## Verification

- Python bytecode compilation: passed locally.
- Git synchronization: commit `422c8bcb` was fast-forwarded on the server and
  pushed to `origin/main` before testing.
- Shell syntax checks for every command script: passed on the server.
- Combined P10/P11 unit suite: `14 passed in 5.80s` on the server. P11 checks
  exact Qwen-order round trips, no orthogonal-axis leakage, direct 1-D FFT
  equivalence, alternating axis schedule, parameter budget, P09-matched
  initialization, fixed-feedback gradients, matching-feedback BP equivalence
  and strict cross-variant checkpoint rejection.
- GPU smoke: completed on physical GPU 1 with batch size 4, three training
  batches and two validation batches. This is a functional test, not a
  performance result.
- Smoke throughput: `7.251 samples/s`; peak allocated/reserved memory:
  `432.3/462.0 MiB` for this process.
- All eight phase-gradient norms were finite and nonzero (range
  `0.03061-0.24514`), and mean absolute phase motion after three updates was
  `0.01093 rad`.
- Optimizer-state audit after the smoke confirmed finite nonzero updates for
  phase `8/8`, adapter `4/4`, residual/mixer `184/184` and head `7/7`
  parameter tensors.
- Smoke artifacts:
  `runs/gpu_smoke_20260824_014543/`; use `metrics/latest.json` for the
  post-update diagnostics. The tiny 12-train/16-validation sample metrics have
  no accuracy interpretation.
- Formal training: intentionally not started.
