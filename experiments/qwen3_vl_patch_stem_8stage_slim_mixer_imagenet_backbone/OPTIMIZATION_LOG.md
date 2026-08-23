# P09 optimization log

This file records decisions, mutations, tests and run evidence so the experiment
can be reconstructed without relying on chat history.

## 2026-08-24: architecture decision

- User selected mixer width 96 rather than 48 and allowed the temporary
  ImageNet head to be excluded from the reusable-backbone optical ratio.
- Locked eight mixer instances, one in every OEO stage. Weights are independent
  between stages and shared across the three latent optical banks within a stage.
- Implemented two learned sigmoid-gated residuals inside each mixer: true-grid
  depthwise 3x3 spatial mixing and a 96-to-192-to-96 channel MLP.
- Retained P08's constrained stage fusion, so the optical branch weight cannot
  fall below 0.5. Retained a zero-initialized final bypass projection and a
  bounded 0.10/0.25 output scale for stable identity initialization.
- Did not add an electronic Transformer, attention, MoE, language tower, long
  skip, or extra distillation teacher. This isolates the mixer hypothesis.

## 2026-08-24: fairness and budget controls

- Added explicit parameter-budget scopes to the shared trainer. P09 enforces
  at least 50% optics in `backbone_excluding_task_head` and still records the
  stricter all-trainable fraction.
- Deterministic count: 1,204,224 optical phases, 231,648 input-adapter
  electronics, 733,472 residual/fusion electronics and 650,603 task-head
  electronics. Reusable-backbone optical share is 55.51%; task-head-inclusive
  optical share is 42.70%.
- Copied P08's full 90-epoch ImageNet recipe: seed 2026, global batch 192,
  per-rank batch 96 on two ranks, 6,672 optimizer updates per epoch, phase LR
  7e-3, adapter LR 5e-4, residual LR 3.5e-4, head LR 9e-4, one warmup epoch,
  cosine decay, label smoothing, mixup, CutMix and identical clipping.

## 2026-08-24: P08 handoff

- At user direction, stopped only the P08 torchrun job and retained its run
  directory and checkpoints.
- P08 completed 9 full epochs. Its epoch-9 train Top-1 was 9.6122% and validation
  Top-1 was 18.4360%. This becomes the latest matched-epoch reference, not a
  final-model baseline.
- Added a comparison command that prints every epoch completed by both P08 and
  P09, including validation delta in percentage points.

## Verification record

- Local Python bytecode compilation: passed.
- Local PyTorch tests: not executable because the workstation's PyTorch DLL
  (`c10.dll`) fails to initialize. This is an environment failure also affecting
  the unchanged P08 tests; authoritative tests are run in the server `xml`
  environment before launch.
- Git synchronization: commit `a72cbdb0` reached both local and server checkouts
  and was pushed to `origin/main`. The server checkout was advanced without
  touching unrelated dirty files.
- Server unit/integration tests: 7 passed in the `xml` environment, covering the
  Qwen-order grid round trip, dual gates, identity initialization, parameter
  budget, backbone export contract and end-to-end phase/mixer backpropagation.
- Server batch-96 GPU forward/backward/optimizer smoke: passed on physical GPU 5
  for four real ImageNet updates and two validation batches. Peak PyTorch memory
  was 8,250.9 MiB allocated / 8,466.0 MiB reserved, and measured throughput was
  78.4 images/s. All eight phase-gradient norms were finite and nonzero; mean
  phase motion reached 0.01583 rad. All three gate types changed from their exact
  0.10 initialization, confirming mixer learning rather than a forward-only run.
- Runtime parameter audit exactly matched the deterministic counts: reusable
  backbone optical share 0.5551097 and all-trainable share 0.4270378.
- Formal P09 launched under `torchrun` as PID 1980929 on physical GPUs 3 and 5,
  with batch 96 per rank / global batch 192. Both ranks and their data-loader
  workers were verified alive after startup.
- The full 50k-image initial validation completed at Top-1 0.12% / Top-5 0.54%.
  Formal optimization then reached epoch 1 batch 100/6,672 without OOM, NaN or
  DDP failure; loss moved from 7.2646 on batch 1 to 7.0362 at the batch-100 log.
  GPU 3/5 utilization was 97%/98% with 9,124/8,961 MiB total device memory in
  use. The run is intentionally left active for all 90 epochs.
