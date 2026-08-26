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

## 2026-08-25: completed 90-epoch ImageNet pretraining

- The formal run completed all 90 epochs. `result.json` records
  `status=complete`, and `history.json` contains the uninterrupted sequence
  1--90. `best.pt`, `last.pt` and `epoch_090.pt` all contain epoch 90 and their
  224 model tensors are identical.
- Epoch 90 is the best Top-1 checkpoint: ImageNet-1K validation Top-1/Top-5 are
  `49.812%/74.224%` over all 50,000 validation images, with loss `2.304278`.
  The minimum validation loss was `2.297755` at epoch 85, but checkpoint
  selection was correctly locked to Top-1.
- Selected validation trajectory (Top-1/Top-5) was:
  `6.932/19.066` at epoch 1, `32.198/56.932` at epoch 10,
  `42.580/67.534` at epoch 30, `48.044/72.682` at epoch 60,
  `49.544/73.886` at epoch 80 and `49.812/74.224` at epoch 90.
  The final ten epochs added only `0.268 pp` Top-1, so the locked recipe was
  close to saturation; simply extending the same cosine schedule is unlikely
  to yield a large gain.
- Epoch-90 training Top-1 was `32.112%`, which is not directly comparable to
  validation accuracy because the training metric includes label smoothing,
  Mixup/CutMix and stronger data augmentation.
- All optical phase gradients remained finite and nonzero. Final circular
  phase motion was `2.00962 rad` mean absolute, with `96.437%` of phases moving
  by more than `0.1 rad`. Phase motion had essentially saturated by epochs
  50--70, ruling out the explanation that the phase learning rate was too
  small for the optical parameters to learn.
- Final optical fusion weights were
  `[0.5029, 0.5000, 0.5157, 0.5190, 0.5605, 0.5880, 0.6265, 0.5025]`
  (mean `0.5394`). They respect the `0.5` constraint, but stages 1, 2 and 8
  lie close to it, exposing a real accuracy-versus-optical-fusion tension.
- Final destructive inference diagnostics were: optical-off Top-1 `0.262%`,
  random-phase `0.078%`, and electronic-skip-off `0.032%`. These results show
  that the trained representation depends on both the learned optical phases
  and the electronic residual. They do not measure either branch's standalone
  capacity, because removing a co-adapted branch creates a large distribution
  shift and is not a retrained architecture comparison.
- The reusable backbone has 1,204,224 optical phases and 965,120 trainable
  electronic adapter/residual parameters. Its optical parameter fraction is
  `55.511%` only when the temporary ImageNet task head is excluded; including
  the task head gives `42.704%`.
- Weighted training throughput was `853.77 images/s`. Pure training took about
  `37.51 h`; validation and final diagnostics brought wall time to about
  `38 h 20 min`. No NaN, OOM, NCCL or runtime failure occurred. A small number
  of PIL EXIF/truncated-metadata warnings did not prevent image decoding.

Scientific interpretation: P09 is a valid and stable source backbone rather
than a failed optimization run. Its main remaining limitation is representation
and fusion quality: the curve and phase motion have saturated, several optical
gates press against their lower bound, and the two branches are strongly
co-adapted. P11 therefore tests a relevant operator change while keeping the
same parameter and training budget.
