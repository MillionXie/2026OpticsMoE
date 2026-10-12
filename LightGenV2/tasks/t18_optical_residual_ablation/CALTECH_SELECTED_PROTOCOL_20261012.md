# Caltech20 validation-screened regularized depth study

Human authorization: retain20 classes, screen for learnability, mitigate the large
train/validation gap, replace8 with12 layers, permit four GPUs, synchronize Git.
Depths2/4/12/16 and rho0/.3 start fresh; previous runs/data/weights remain intact.

The author CaltechDATA record DOI10.22002/D1.20086 declares CC BY4.0. The retained
official API response contains rights.id=cc-by-4.0 and the license legalcode URL.
Credit Li, Andreetto, Ranzato and Perona and identify subset/resizing/splitting.
The author archive MD5 is3138e1922a9193bfa496528edbbc45d0; no raw images in Git.

One deterministic selection pass: exclude background and duplicate Faces_easy,
require >=60 source photos, split connected original/cache pixel groups first,
pin BOTH previous50 and20 train/validation roles. Use approximately70% training
and15% validation, capped100/20 groups; retain at least8 test groups per class.
Train-only class centroids of fixed RGB16 thumbnails and finite differences are
scored on validation. Remove lowest validation recall, ties nearest-other margin
then name, until20. No learned feature extractor, optical inference, residual
benefit or test score participates. Freeze the chosen names/IDs/cache before fit;
retain all photos of chosen classes. Proxy accuracy is not optical accuracy.
Data-dependent screening makes this a development task; earlier Caltech20 test
was observed, so future held-out scores must not claim independent task selection.

One common regularized setting: seed17, raw phase initialization +/-.1, effective
batch16, AdamW lr.002/weight_decay0, EMA.95, label smoothing.08, phase smoothing.05,
capture.2, affine rotation20deg/translation5pixels/scale10%, training-only amplitude
MixUp alpha.2. MixUp is deterministic by seed/epoch/macro batch, applied before
splitting microbatches with the same coefficient/permutation on soft targets.
Class weighting applies to each soft target class; loss and auxiliary terms are
sample weighted including tails. Once per macro: clip1, AdamW, EMA. Evaluation
uses full unaugmented images and original hard labels. No optical graph changes
except requested depth12 (six expert/global cycles,61 phases,2,649,088 parameters).
All9 experts, original ASM/OEO and fixed20 detector windows remain; router has no
rho. No CNN/Linear/pretrain or learned electronic readout.

100 epochs per arm, no automatic restarts/extensions or accuracy-driven changes.
Every epoch validation; train at1/every5/endpoint plus selected train. Select first
minimum validation EMA balanced NLL, not the maximum accuracy or last checkpoint.
Maximum36,000sec per arm; time exhaustion/child failure is preserved and stops
new dispatch while healthy children finish. No automatic test in the fit queue.

Only physical GPU0/1/2/4, one task GPU child per card, Torch3GiB plus800MiB margin.
CPU supervisor CUDA invisible. Fresh actual MixUp-objective numerical checks compare
serial micro1 against expert-batched1/2/4; probability<=1e-6, gradientrelative<=.02,
cosine>=.999 and every phase finite/nonzero. Failed modes retained. Largest common
passed mode per rho pair is used and checked again in production. No other process
killed, no fifth GPU, no shared environment changes. Sources/commands/Git/data SHA,
initialization, order and augmentation including MixUp hashes are recorded.

After all eight fits, CPU-only saved-results audit verifies pair identity and
sources against each training Git blob; all eight choices must be locked before
each selected EMA receives at most one test with durable exclusive claim before
materializing test arrays. Failed claims cannot be retried/deleted automatically.
Reports must show train/val macro/NLL/gap and late trends, not claim convergence
from low training error or guarantee70%. Subset changes prevent direct controlled
accuracy comparison with the old20 classes; single seed does not establish causal
or universal residual benefit. Synchronize main and verify actual GPU release.

Preparation: prepare_caltech_selected.py --out /DATA/DATA1/guest3/t18_caltech_selected20_20261012.
The actual data identity is inserted in caltech_selected_profile.json after the
selection manifest is frozen, then committed/pushed before the server fit starts.
Queue: caltech_selected_queue.py --phase suite --data <cache> --profile <profile>
--out <task>/runs/simulation/caltech_selected20_s17_L2L4L12L16_reg100_4gpu_20261012.

## Independent executable completion

`caltech_selected_finalize.py` is separate from the sealed historical20/COIL
entries. It validates the actual frozen profile, dynamic class names/support/IDs,
common100, original source Git blobs, all pair identities and independently
regenerates each epoch's deterministic MixUp hash including the macro tail.
It executes all four saved-PT/CSV CPU pair audits before any pair/local/global
lock. The exclusive suite marker prevents duplicate test supervisors; the worker
claim is file+directory fsynced on Linux before first test-array materialization.
No claims are deleted/retried after failures.

Only AFTER the live fit queue is complete, all eight fits exist, and GPU children
are exited, use these ordered commands from the server repository root:

```sh
T=LightGenV2/tasks/t18_optical_residual_ablation
RUN=$T/runs/simulation/caltech_selected20_s17_L2L4L12L16_reg100_4gpu_20261012
DATA=/DATA/DATA1/guest3/t18_caltech_selected20_20261012/caltech20_selected_split.npz
PY=/home/guest3/miniconda3/envs/xml/bin/python
CUDA_VISIBLE_DEVICES='' $PY $T/caltech_selected_finalize.py --phase lock --base $RUN --out $RUN/final --data $DATA
CUDA_VISIBLE_DEVICES='' $PY $T/caltech_selected_finalize.py --phase suite --out $RUN/final --data $DATA
CUDA_VISIBLE_DEVICES='' $PY $T/audit_caltech_selected_final.py --run $RUN/final --out $RUN/cpu_report_final --data $DATA
```

Inspect existing output/claims before acting. These steps do not authorize
bypassing a failed audit or restarting a claimed test. The final CPU audit waits
for all eight once-only results and supervisor completion before reading labels,
then checks saved hard-label probabilities/CM/support/macro/NLL and checkpoint
identities without optical inference. Exports include full CM CSV, support CSV,
summary and curve CSV plus accuracy/macro/NLL curves, selected summaries and
per-depth row-normalized test CM figures in PNG/SVG/PDF. Actual completed figures
must be visually inspected; no single-seed error bars or invented test history.
