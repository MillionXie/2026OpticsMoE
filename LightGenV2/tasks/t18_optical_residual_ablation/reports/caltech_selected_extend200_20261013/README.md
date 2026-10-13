# Screened Caltech20: authorized continuation to200

Actual start: 2026-10-13 about09:42 Asia/Shanghai. All eight2/4/12/16 x rho0/.3
are authorized to reach200 total epochs. Previous100 fits/tests/weights preserved.
Human chose unchanged regularization and validation-NLL best selection.

CPU supervisor2043961, CUDA invisible. Initial children: L16rho0 GPU1 PID2045571,
L16rho.3 GPU2 PID2045712, L12rho0 GPU0 PID2045853, L12rho.3 GPU4 PID2045994.
L4/L2 pairs automatically follow when allowed slots free. PIDs/epochs here are
snapshots; use live status/current_processes and actual /proc/UUID occupancy.
No fifth GPU, no other processes killed or environment changes.

Four restored-state production gates passed: probability and gradient-relative
errors0, minimum cosine>=.99999982, all phase gradients finite/nonzero.
Raw E100 model, EMA, AdamW moments/steps and plateau scheduler all restored exactly.
L12rho.3 keeps its already-reduced lr.001; other arms retain their saved lr.002.
Global epoch101-200 prevents replaying initial augmentation/MixUp schedules.
Train regularization remains MixUp.2, label smoothing.08, phase smoothing.05,
affine20deg/5pixels/10%; same seed, optical graph, data and classes.
Both old best and new EMA checkpoints compete only by first minimum validation
balanced NLL across1-200; the last epoch is not automatically the selected model.

Eight CPU continuation/finalizer checks passed on the Linux execution environment.
Local source syntax checks pass; native Windows Torch tests could not import the
existing c10.dll (WinError1114), and the shared environment was not altered.
Real CPU partial checks at L16E102/E102 and L12E103/E103 verified original100-row
prefixes, saved states/config/76 source files and their original published Git
blobs, restored raw/EMA/optimizer/scheduler identities,81/61 phase gradients,
independently regenerated102/103 MixUp hashes including the12-sample macro tail,
new global epoch order/affine hashes, validation predictions/IDs/labels/CM/macro/NLL.
These checks are not completed200 fits or selection locks. No new test has run.

Training source commit `f7518c1265e9b37aeb53ebfbd8c98e44bd314526`, original parent
training source `a477d916ce76d5ac6ab109b3c892b0b4361410ce`.76 current dependencies
remain frozen after launch. Independent completion tooling is published separately
at85f8d9fe6; later report commits do not change those training sources.
[Actual launch/restoration/CPU receipt](production_start.json).

## Estimated elapsed time (additional100 only)

| Depth | Previous measured mean epoch | Additional100 per arm | Execution |
|---|---:|---:|---|
| 2 | 15.57-16.10s | about27min | after a slot frees |
| 4 | 18.27-18.45s | about31min | after a slot frees |
| 12 | 169.58-170.64s | about4h45 | initial pair |
| 16 | 207.20-215.91s | about6h | initial pair |

Four-card total training estimate6-7h from launch, final audits/tests/plots another
20-30min. Expected training window15:45-16:45 and full delivery16:15-17:15 today,
Asia/Shanghai; this is an estimate, not completed timing. First continuation
epochs are about160-177s at12 and200-207s at16, consistent with that projection.
Update only if actual contention/evaluation overhead materially changes the ETA.

The deep train/val gap is substantial and continuing training may enlarge it.
The retained MixUp/smoothing/augmentation, restored validation-plateau LR decay,
EMA and best-validation selection are controls, not a guarantee against overfit.
Report accuracy AND macro recall/NLL/support/gap, selected and last performance,
late validation trends and any negative result. Single seed and prior observed
tests remain exploratory development; no full101-class or independent-generalization
claim. Do not use test to choose epochs/settings/depth or to add another candidate.

## Completion

Independent `audit_caltech_selected_extend.py` verifies every200-row saved arm,
parent prefix, state continuity and pair equality. `caltech_selected_extend_finalize.py`
locks all eight only after all CPU checks and actual fit exits, then each changed
selected EMA gets one test behind a file+directory-fsynced exclusive claim. If
parameters exactly match the old tested EMA, reuse its saved test result/predictions
after identity and previous CPU receipt checks; do not repeat inference. No test
or suite retry/delete after failure. Final CPU all-eight completion barrier precedes
test-label materialization and final plots/export. Commands and constraints are in
[the continuation protocol](../../CALTECH_SELECTED_EXTEND_PROTOCOL_20261013.md).

Run `runs/simulation/caltech_selected20_s17_L2L4L12L16_extend200_4gpu_20261013`.
Monitoring stays quiet for ordinary healthy progress and advances completion
automatically; notify meaningful stage results/failure/action, then stop only
after final inspected plots, Git synchronization and actual project GPU release.
