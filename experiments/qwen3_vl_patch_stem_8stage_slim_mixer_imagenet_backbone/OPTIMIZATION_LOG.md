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
- Server unit/integration tests: pending Git synchronization.
- Server batch-96 GPU forward/backward/optimizer smoke: pending tests.
- Formal P09 launch and live-process/GPU verification: pending smoke success.
