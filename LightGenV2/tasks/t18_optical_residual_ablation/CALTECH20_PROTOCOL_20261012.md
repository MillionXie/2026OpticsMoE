# Caltech20: more training images per category

## 2026-10-12 continuation after completed pilot

The human challenged the prolonged idle wait and requested progress. Continue
the originally requested all-depth100 comparison without another budget question.
Reuse completed L2rho0/rho0.3 fits unchanged; fresh L4/L8/L16 pairs each100epochs.
The bounded registration is caltech20_full_suite_plan.json. The CPU supervisor
caltech20_fit_queue.py starts L16pair on physicalGPU1/2 and L8pair on0/4, then
dispatches L4pair into released allowed slots. Four children maximum, micro4
forL4 and micro1 forL8/L16; all original numerical and fresh production gates
remain enforced. Runtime sources of the completed pair remain unchanged.
The pilot70% target failed; proceeding reports that negative result openly.
No new classes, hyperparameters, candidates or extra epochs. L16two-epoch
timing is excluded from formal fits. Training completes at a CPU audit barrier;
the plan requires all-eight locks and a Caltech20-specific once-test finalizer.
No test is launched by the training queue. Monitor ordinary progress quietly,
notify meaningful stages/failures and act at completion instead of waiting
again for a redundant budget choice.

The latest human request is 20 categories with more images per category. This
replaces the proposed 30-category task and cancels expansion of Caltech50. The
completed Caltech50 two-layer fits and two-epoch timing runs remain archived;
their test split was never evaluated.

## Data selection before new fitting

Use the existing original CaltechDATA archive, DOI 10.22002/D1.20086, verified
against author MD5 3138e1922a9193bfa496528edbbc45d0. Exclude BACKGROUND_Google
and Faces_easy. Among the 25 remaining categories with at least 80 original
images, select 20 without replacement using numpy default_rng(20261012), then
sort their names. Category eligibility uses image counts, never model scores or
residual advantage. This is a count-conditioned subset, not full Caltech101.

Retain all original images of the selected categories. Merge exact original and
cached pixel duplicates; conflicting labels fail preparation. Assign 60 connected
groups to train and 10 to validation per category, with all remaining groups in
test. Preserve every previous Caltech50 train and validation image in its same
role; no previously fitted or validation-observed image may enter new test.
Additional images can come from the previous unread test partition. Preparation
fails if these constraints cannot hold. Use split seed 20261012. Actual image
counts, supports, group counts, selected names, parent manifest and cache SHA
must be recorded before training. Duplicates could increase image counts above
group targets. No independent subject or photographer claim is established.

New data root: /DATA/DATA1/guest3/t18_caltech20_20261012. Preparation uses the
published generic prepare_caltech50.py with caltech20_preparation_plan.json;
the old archive, data root, split, checkpoints and reports are preserved.
Full-image RGB conversion and bicubic150 retain original backgrounds. No crop,
mask, segmentation or learned features are introduced. Task adaptation follows
previous validation observations, so results are exploratory.

## Common optical and fitting contract

Depths 2/4/8/16, each rho0/rho0.3, seed17, 100 full epochs without early stopping.
Nine experts, original ASM/OEO, residual only in expert/global paths, none in
router; no CNN, Linear, pretraining or electronic learning head. All depths use
the same 20 fixed 32-pixel detector windows, rows 8/8/4. Trainable parameters are
449848/889696/1769392/3528784; trainable phase arrays 11/21/41/81. Buffers are
excluded from parameter counts.

Common raw phase initialization range +/-0.1, AdamW lr0.002, EMA0.95,
label smoothing0.02, capture0.2, phase smooth0.02, original augmentation
10 degrees/3 pixels/scale0.05; validation plateau factor0.5, patience8,
minimum lr2e-5. Effective batch16 uses sample-weighted accumulation including
tail batches and auxiliary losses; clip/AdamW/EMA once per macro batch.

Fresh train-only numerical probes select a common execution microbatch for both
rho arms at each depth. Probability error <=1e-6, maximum gradient relativeL2
<=0.02, minimum cosine >=0.999, finite nonzero gradients for every phase array
are unchanged. Production initialization gates must also pass. Failed cases are
retained. These checks are local numerical evidence, not arbitrary-weight
pixelwise equivalence. Physical GPUs0/1/2/4 only, one project child per card,
Torch3GiB plus800MiB reserve; actual exit and memory release are checked.

First run two-layer100 train/validation fits and two-epoch sixteen-layer timing.
Timing is not a formal arm. Review actual learning and runtime before expanding
the remaining six formal100-epoch arms. No extra candidate or extension is
currently authorized by this protocol; any finite paired adaptation requires
prior registration and preserves all base results. Do not search categories or
hyperparameters indefinitely to obtain70% or positive residual results.

## Selection, fit review and final evaluation

Each epoch evaluates EMA on validation. Select minimum validation balanced NLL;
exact ties keep the first epoch. Accuracy never replaces that checkpoint.
Evaluate full unaugmented train using the same EMA every five epochs and at
endpoints; save best/last, predictions, orders/transforms and phase gradients.
Train/validation curves and late NLL trends determine fit interpretation. Low
train accuracy alone does not prove insufficient capacity; a train/validation
gap alone does not prove severe overfitting. A still-improving endpoint must be
reported as a limited budget rather than asserted convergence. No70% guarantee.

CPU-only audits read saved checkpoints, histories, predictions and labels; no
repeat optical inference of completed arms. Check archived Git sources, data,
configuration, first minimum, all phases, IDs, labels, supports, confusion
matrices, macro recall and NLL. Pair initialization/orders/augmentation/budget
match and configuration differs only by rho.

All eight choices and source/configuration/data/checkpoint identities must be
sealed in pair, selection and global locks before any test evaluation. Selected
EMA receives exactly one test evaluation; exclusive test_started marker precedes
test materialization. Failures do not permit automatic retries or marker removal.
Unselected/raw/last candidates never receive test evaluation. Test cannot change
training, categories or checkpoint choice. Final independent CPU audits generate
actual train/validation curves plus one selected test point, summary/support CSV
and PNG/SVG/PDF, single seed without error bars. Complete reports, synchronize
Git and verify actual GPU release before ending the monitor.
