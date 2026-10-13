# Screened Caltech20 common200 continuation (2026-10-13)

Human requests all eight current arms extended to200 epochs, four GPUs, an ETA,
and overfitting control. Human explicitly chose unchanged regularization with
validation NLL selecting best weights. This supersedes the old100-epoch ceiling
only for this new continuation; the sealed100 runs, tests and source files remain.

Depth2/4/12/16, rho0/.3, seed17, same frozen3719 photos and20 categories.
CacheSHA a81e78bbff06ea65e7ceba3771b688178276464e33fdcdedf3796fa2930fa6e1,
manifestSHA91f7aa36a97713e300699367904c8866551996de199b9630c0545bb24dc5f60e.
No screening, new candidate, extra class, test-driven change, or epoch extension
beyond200. Previous tests have been observed: exploratory development, not an
independent test or causal proof of depth/residual generalization.

Each arm restores the100th raw model, EMA, AdamW moments/step counters and
ReduceLROnPlateau state. Start at101; deterministic order, affine parameters and
MixUp use global epoch101-200 rather than replaying1-100. Preserve full histories
and identity of initial parameters from the parent. Do not resume the selected
EMA as the raw model, reset Adam/EMA, or reset an already-reduced learning rate.

Keep MixUp.2, label smoothing.08, phase smoothness.05, affine20deg/5pixels/10%,
capture.2, effective16, EMA.95 and the same optical graph. Passed pair modes
2/4micro4,12/16micro1; restored E100 raw states receive a real E101-objective gate
against serial micro1: probability<=1e-6, gradrelative<=.02, cosine>=.999, all
phase gradients finite/nonzero. No thresholds relaxed or failed automatic retry.

Complete100 additional epochs per arm, max36,000 additional seconds. Evaluate
validation each epoch and unaugmented training every5/endpoint. Keep the first
minimum validation EMA balanced NLL across all1-200, including the previous best.
Plateau LR halves after8 bad epochs, floor.00002, retaining previous state. These
controls reduce selection/optimization risk but cannot promise a smaller gap.
Track train/val accuracy and macro gaps, late NLL trends, chosen and last epochs.
Do not mistake ongoing validation improvement for full convergence or attribute
all low accuracy to overfitting. No early stop because the human requested200.

Only physicalGPU0/1/2/4, one project child per card, Torch3GiB+800MiB margin, CPU
supervisor CUDA invisible. Start16pair and12pair concurrently, then4pair/2pair
on freed slots. No other process killed, no shared environment changes. Each
child preserves source/current/original Git blob identity and parent checkpoint
SHA; verify restored raw/EMA/optimizer/scheduler values before any update.

Measured previous mean epoch: L2 15.57-16.10sec, L4 18.27-18.45sec,
L12 169.58-170.64sec, L16 207.20-215.91sec. Additional100 estimate: L2 about27min,
L4 about31min, L12 about4h45, L16 about6h. Four-card total about6-7h after launch,
plus20-30min final audits/tests/plots; timing estimate, not a completion promise.
Revise from actual logs if contention or validation/evaluation overhead changes.

After all eight have200 and actual GPU exit, CPU verify parent prefix, saved
best/last/config/source/data, restore receipts,200 order/affine/MixUp hashes,
all phase gradients, pair config onlyrho, IDs/labels/argmax/CM/support/macro/NLL.
Lock all eight before any endpoint test. If selected parameters exactly equal
the previously tested parent EMA, reuse the saved old test prediction/result
after CPU identity checks, never re-infer it. Changed selected EMA gets its first
single test behind an exclusive file+directory-fsynced claim before test arrays.
Claims and suite marker never deleted or retried after failure. No raw/last tests
or test-based model selection. Final CPU audit reads saved results only, exports
real1-200 curves, old endpoint reference and current single endpoint point,
PNG/SVG/PDF/CSV plus full confusion/support; inspect actual figures.

New worker: caltech_selected_extend.py; queue: caltech_selected_extend_queue.py.
Run: runs/simulation/caltech_selected20_s17_L2L4L12L16_extend200_4gpu_20261013.
Parent: runs/simulation/caltech_selected20_s17_L2L4L12L16_reg100_4gpu_20261012.
Completion tooling is independent from the sealed historical100 finalizer.
Only T18/main, serialized exact-path Git publication; source sync Git only.
Keep all parent data/PT/predictions/tests, preserve other tasks, exclude laboratory.

## Executable completion (only after all200 and actual fit child/supervisor exits)

Inspect real status/current_processes and existing claims before each phase;
never repeat an existing suite_started/test_started or repair it by deletion.
From the server repository root, with its existing Python environment:

```sh
T=LightGenV2/tasks/t18_optical_residual_ablation
RUN=$T/runs/simulation/caltech_selected20_s17_L2L4L12L16_extend200_4gpu_20261013
DATA=/DATA/DATA1/guest3/t18_caltech_selected20_20261012/caltech20_selected_split.npz
PY=/home/guest3/miniconda3/envs/xml/bin/python
CUDA_VISIBLE_DEVICES='' $PY $T/caltech_selected_extend_finalize.py --phase lock --base $RUN --out $RUN/final --data $DATA
CUDA_VISIBLE_DEVICES='' $PY $T/caltech_selected_extend_finalize.py --phase suite --out $RUN/final --data $DATA
# Wait for all endpoint workers and the test supervisor to exit.
CUDA_VISIBLE_DEVICES='' $PY $T/audit_caltech_selected_extend.py --base $RUN --data $DATA --out $RUN/cpu_report_final --final $RUN/final
```

The lock phase independently verifies every200-step saved artifact and resume
receipt before any pair/local/global lock. The test worker decides reuse solely
by exact selected parameter identity, never by a test score. Reused results are
checked against the previous CPU receipt. The final CPU all-eight barrier
precedes materialization of test labels. Exported hollow points mark previous100
tests, stars mark the currently selected endpoint (explicitly reused if same).
