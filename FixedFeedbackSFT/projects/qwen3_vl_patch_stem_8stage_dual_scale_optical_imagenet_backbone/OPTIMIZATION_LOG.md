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

## 2026-08-27 formal ImageNet launch

- P11 had completed before launch, and no P09/P11 training process remained.
  Physical GPU 0 and GPU 3 were both idle RTX 4090 devices at `12 MiB/0%`.
- Launched only through `commands/03_launch_imagenet_90e_bs96.sh` with
  `PHYSICAL_GPU_INDICES=0,3`; torchrun PID is `3804958`.
- The controlled run uses the full ImageNet-1K splits, two DDP ranks, batch 96
  per rank/global batch 192 and the same 90-epoch optimization recipe as P09
  and P11. P10 changes only the propagation schedule to
  `[local 5 mm -> global 50 mm] x 4` and adds no trainable parameters.
- Full initial validation completed at Top-1/Top-5 `0.11%/0.52%`. Training
  entered epoch 1 and reached batch 100/6,672; logged loss moved from `7.0776`
  at batch 1 to `7.0021` at batch 100 during warmup.
- A post-start health check measured approximately `9,122/9,124 MiB` device
  memory and `97%/98%` utilization on physical GPUs 0/3. Both DDP workers and
  their data-loader workers were alive, with no baseline restart error, OOM,
  NCCL failure, config mismatch or traceback.

The job is intentionally left active through epoch 90 and the same final
normal/optical-off/random-phase/electronic-skip-off evaluation used by P09 and
P11. Completion requires `result.json` with `status=complete` and an exported
`checkpoints/backbone.pt`.

## 2026-08-29 completed 90-epoch ImageNet pretraining

- The formal run completed all epochs and all final evaluations.
  `result.json` records `status=complete`, the history is exactly epochs 1--90,
  and `checkpoints/backbone.pt` was exported.
- `best.pt`, `last.pt` and `epoch_090.pt` all load successfully, record epoch
  90 with 90 history rows and best Top-1 `0.50888`; all 225 model tensors are
  identical across the three files.
- Epoch 90 is the Top-1-selected checkpoint: ImageNet-1K validation Top-1/Top-5
  `50.888%/74.956%`, loss `2.240395`, over all 50,000 validation images.
- Selected Top-1 trajectory was `7.186%` at epoch 1, `32.484%` at epoch 10,
  `43.442%` at epoch 30, `49.142%` at epoch 60, `50.640%` at epoch 80 and
  `50.888%` at epoch 90. Epochs 80--90 added only `0.248 pp`, so the run was
  close to saturation but still ended at its best Top-1.
- Against matched P09, P10 improves Top-1 by `1.076 pp` and Top-5 by
  `0.732 pp` without adding parameters. P11 remains higher than P10 by
  `0.460 pp` Top-1 and `0.596 pp` Top-5 at their selected checkpoints.
- Final circular mean absolute phase motion is `1.81177 rad`; `93.743%` of
  phases moved more than `0.1 rad`. Per-stage mean motion is
  `[1.7421, 1.7869, 1.9797, 2.0374, 1.8945, 2.0750, 1.0150, 1.9635] rad`.
- Best-epoch optical fusion weights are
  `[0.5000, 0.5009, 0.5028, 0.6355, 0.5527, 0.6088, 0.6457, 0.5013]`
  (mean `0.5560`). All satisfy the `0.5` lower bound, but the contribution is
  distributed unevenly across local/global stages.
- Final diagnostics are: optical-off Top-1 `0.344%`, random-phase `0.088%`,
  and electronic-skip-off `0.080%`. P10 therefore depends on both learned
  phase and the electronic residual; these destructive interventions are not
  standalone retrained-branch accuracies.
- Epoch-90 training throughput was `1081.17 images/s`. P10 used two RTX 4090s,
  whereas P09/P11 used a mixed RTX 4090/3090 pair, so this number cannot be
  used as a controlled architecture-speed comparison.
- No NaN, OOM, NCCL, config mismatch or runtime failure was found. The process
  exited normally only after writing the final result and backbone checkpoint.

Scientific interpretation: alternating local/global propagation gives a real
single-seed accuracy improvement over repeated global propagation, validating
the broader idea that a physically structured optical schedule matters. The
token/channel semantics of P11 are nevertheless more effective under the same
budget, so P10 should be retained as a supporting architecture rather than the
primary downstream source.
