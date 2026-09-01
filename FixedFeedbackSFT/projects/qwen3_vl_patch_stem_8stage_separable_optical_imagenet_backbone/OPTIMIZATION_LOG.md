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

## 2026-08-24 interruption checkpoint

- The first formal run completed epochs 1--15 and was then stopped at the
  user's request so the GPUs could be released. A partial epoch 16 reached
  approximately batch 800 but had no epoch checkpoint and was intentionally
  discarded.
- The complete epoch-15 result was Top-1/Top-5 `37.392%/62.168%`, validation
  loss `3.004432`, and training throughput `792.76 images/s`. Epoch 15 was also
  the best checkpoint at interruption time.
- `best.pt`, `last.pt` and `epoch_015.pt` all load successfully, contain the
  same 225 model tensors and record epoch 15, 15 history rows and best Top-1
  `0.37392`. Optimizer state (203 entries), scheduler state, AMP scaler,
  initial phases and the matching config digest are present.
- At the matched epoch 15, P09 was `36.430%/61.064%`; P11 led by
  `+0.962/+1.104 pp` Top-1/Top-5. This is an encouraging controlled partial
  result, not a replacement for the full 90-epoch comparison.

## 2026-08-26 resume on physical GPUs 2 and 3

- Immediately before resume, physical GPU 2 (RTX 3090) and GPU 3 (RTX 4090)
  used `13/17 MiB` with `0%/0%` utilization. No P11 process remained; the old
  `launch.pid=3395957` was stale.
- The original log was preserved as
  `logs/p11_imagenet1k_pretrain_bs96_90e.log.through_epoch15_20260824` before
  the launcher opened a fresh active log. No checkpoint or metric file was
  deleted or renamed.
- Resumed only through `commands/03_launch_imagenet_90e_bs96.sh` with
  `PHYSICAL_GPU_INDICES=2,3`. New torchrun PID is `1133915`; physical GPU 2 is
  DDP rank 0 and physical GPU 3 is rank 1. Per-rank batch remains 96 and global
  batch remains 192.
- The trainer explicitly reported `[resume] epoch=16 best=0.3739`, followed by
  epoch-16 batches 1, 100 and 200. A health snapshot measured approximately
  `9,206/9,378 MiB` and `96%/95%` utilization on GPUs 2/3, with no OOM, NCCL,
  strict-load or config-digest error.
- Resume restores model, optimizer, LR scheduler, AMP scaler, best score,
  history and initial phases. It restarts epoch 16 from batch 1, so the earlier
  uncheckpointed partial epoch cannot contaminate the metrics. The epoch-aware
  sampler reproduces the epoch-16 sample order.
- Checkpoints do not store Python/NumPy/Torch/CUDA RNG states. Consequently
  stochastic augmentations after process restart are statistically valid but
  not bitwise identical to an uninterrupted run. This limitation must be kept
  in the reproducibility record; optimizer and learning-rate continuation are
  exact.

The active job is intentionally left running through epoch 90 and final
normal/optical-off/random-phase/electronic-skip-off evaluation. Completion is
defined by `result.json` with `status=complete` and creation of
`checkpoints/backbone.pt`, not merely by the absence of the torchrun process.

## 2026-08-27 completed 90-epoch ImageNet pretraining

- The resumed run completed all epochs and all final evaluations.
  `result.json` records `status=complete`, `history.json` contains epochs
  1--90, and `checkpoints/backbone.pt` was exported successfully.
- The Top-1-selected checkpoint is epoch 88: validation Top-1/Top-5
  `51.348%/75.552%`, loss `2.209665`, over all 50,000 ImageNet-1K validation
  images. Epoch 90 ended at `51.224%/75.582%`; the small difference confirms a
  late plateau rather than a failed ending.
- Selected Top-1 trajectory was `6.712%` at epoch 1, `37.392%` at epoch 15,
  `44.234%` at epoch 30, `49.832%` at epoch 60, `51.198%` at epoch 80 and
  `51.348%` at epoch 88. Epochs 80--88 added only `0.150 pp`.
- Against the controlled P09 checkpoint selected by the same Top-1 rule, P11
  improves Top-1 from `49.812%` to `51.348%` (`+1.536 pp`) and Top-5 from
  `74.224%` to `75.552%` (`+1.328 pp`). P09 and P11 have the same optical,
  adapter, residual and task-head parameter counts and the same 90-epoch
  recipe; the controlled change is the optical propagation operator.
- At the selected checkpoint, circular mean absolute phase motion is
  `1.66190 rad`, and `92.721%` of phases moved more than `0.1 rad`. The
  per-stage mean motion is
  `[1.6962, 1.8388, 1.9966, 1.9440, 1.9437, 1.7907, 1.4500, 0.6352] rad`.
  The last channel-axis stage moved substantially less than the earlier
  stages, but randomizing all phases still collapses performance.
- Final inference diagnostics at the selected checkpoint are: optical-off
  Top-1 `4.556%`, random-phase `0.086%`, and electronic-skip-off `0.098%`.
  Learned optics and the electronic residual are therefore both necessary for
  the trained model. P11 retains more information in the destructive
  optical-off path than P09 (`4.556%` versus `0.262%`), so the accuracy gain
  must not be described as proof that every layer became more optically
  dominant.
- Epoch-90 optical fusion weights are
  `[0.5018, 0.5001, 0.5034, 0.5048, 0.5505, 0.5068, 0.5364, 0.7137]`
  (mean `0.5397`). The mean is close to P09's `0.5394`, but P11 concentrates a
  much stronger optical weight in the final channel-axis stage while several
  earlier stages remain close to the hard `0.5` lower bound.
- Epoch-90 training throughput was `773.80 images/s` with approximately
  `8,504/8,744 MiB` peak allocated/reserved PyTorch memory per reported rank.
  P11 is slower than P09 under the matched mixed 4090/3090 setting because the
  axis-specific layout and propagation add overhead.

Scientific interpretation: the full result upgrades the epoch-15 indication
to a meaningful controlled single-seed result. Explicit token/channel-axis
optical mixing improves the source-backbone objective by 1.536 percentage
points without increasing parameters. It is not yet a multi-seed generalization
claim, and the optical-off/gate pattern shows that the gain still comes from a
co-adapted hybrid system rather than a standalone optical branch.
