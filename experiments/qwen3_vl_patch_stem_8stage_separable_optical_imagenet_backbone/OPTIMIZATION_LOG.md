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
- At implementation/smoke time, formal training was intentionally not started;
  the later user-authorized launch is recorded below.

## 2026-08-24 formal ImageNet launch

- The user assigned physical GPU 1 (RTX 4090) and GPU 2 (RTX 3090). Immediately
  before launch they used approximately 1,523 MiB/0% and 13 MiB/0%,
  respectively.
- Launched only through
  `commands/03_launch_imagenet_90e_bs96.sh` with
  `PHYSICAL_GPU_INDICES=1,2`; launcher PID: `3395957`.
- The locked run uses two DDP ranks, batch 96 per rank/global batch 192, the
  full ImageNet-1K train/validation splits and 90 epochs. Output and log are:
  `runs/p11_imagenet1k_pretrain_bs96_90e/` and
  `logs/p11_imagenet1k_pretrain_bs96_90e.log`.
- Startup validation completed with the expected random-initialization
  baseline Top-1/Top-5 of `0.0008/0.0053`. An early utilization check measured
  roughly 10,852/9,170 MiB and 95%/96% utilization on physical GPU 1/2.
- Epoch 1 completed and the run automatically entered epoch 2. Validation
  Top-1/Top-5 were `0.06712/0.18348` with loss `5.42346`; training throughput
  was `783.48 samples/s`, with peak allocated/reserved memory
  `8,482.15/8,706 MiB` per reported rank.
- All eight phase-gradient norms were finite and nonzero (range
  `0.003623-0.041196`). Mean absolute phase motion was `0.304789 rad`, and
  `74.04%` of phase parameters moved by more than `0.1 rad`. All eight optical
  gates remained above the enforced `0.5` lower bound (`0.568-0.597`).
- Recoverable `best.pt` and `last.pt` checkpoints were both written after
  epoch 1 (approximately 44 MiB each), and `metrics/latest.json` plus
  `metrics/history.json` were present.
- No OOM, NCCL failure or traceback was observed. PIL emitted isolated corrupt
  EXIF warnings while still decoding the affected images; training continued.
- The existing P09 controlled run was not stopped and continued independently
  on its assigned devices.
- This remains an early launch/health result. Scientific P09--P11 comparison
  must use matched completed epochs and ultimately the locked 90-epoch budget.
