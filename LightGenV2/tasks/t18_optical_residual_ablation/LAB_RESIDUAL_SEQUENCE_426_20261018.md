# Latest residual experimental order: L4 -> new L2 -> new L6

Both TRAIN-teacher comparisons ended without gain (blend selected0): L2 remains
291/411 and L4 remains224/411. New score-gap diagnosis gives L2 wrong-example
median required single-row radius0.10047 and90th percentile0.25317. Under the
old0.08 radius,73/120 wrong examples cannot flip even with their individually
optimal single-row direction; for L4 this is59/187. This is a geometric bound,
not a claim that all are fixable or that their optimal directions agree.

Next is one bounded continuation of the accepted TRAIN-neighbor binary row
Newton repair, radius0.20, all20 wrong classes eligible,192steps/class or180fit
seconds. Other loss/sampling/CPU/freeze/selection rules stay fixed; profile
`lab_caltech20_residual_rowradius_profile_20261019.json`. No radius/seed grid.
Each257-variable row gradient/full Hessian/damped Woodbury solve is tested
against autograd. Evaluate complete411, preserve parent and record both
new-correct and new-wrong. Subsequent short segments require actual accuracy
or tied-NLL progress; no endless repetition after no improvement.

NewL2 has all4833 frames and strictCPU complete: ordinary final5150 fit selected
274/411, then targeted original-head repair improved to291/411(70.8029%). It is
still46 examples below frozen337, not within1pp. L6 admitted after full native
TRAIN1200/TEST411 exact local parity and began fresh router capture; its hardware
chain does not wait for further L2 electronic optimization.

The next bounded L2/L4 readout comparison uses their original frozen simulation
model as a TRAIN-only semantic teacher. Pair current measured TRAIN features
with simulated features of the SAME1200 TRAIN images, estimate one orthogonal
alignment, and analytically fold it into the existing Linear256to20 head. The
alignment cannot amplify feature norms. A single convex parent/new-head blend
gets64 TRAIN CE/LS.02/correct-TRAIN KL.5 updates, not a TEST-driven parameter
gradient or a hyperparameter grid. Teacher is never evaluated on TEST/VALID in
this fit. Both have parent/best/last and full411 CPU/frozen-nonhead audits; these
remain TEST-informed development results, not independent generalization. No
added inference layer, optical change, recapture, SDK or GPU. Prior/shrinkage
failures remain and must not be relabelled as improvements.

L6 numerical admission explicitly uses same-host graph parity, consistent with
the original L4 deployment contract. Exact frozenPTeb3d... retains historical
cached354/411; original source/bridge/portable reference on serverCPU each give
352/411, while laboratory source/bridge give353/411 with identical local outputs.
No checkpoint was changed to obtain these numbers. The first lab gate's extra
cross-host exact-count assertion failed and is preserved as evidence. A fresh
full native TRAIN1200/TEST411 diagnostic must prove maxabs<=3e-6 on every row,
matching IDs/labels/data/PT/reference SHA plus the actual layer-major contract.
Only after all these pass may `--admit-native-cpu` emit the lab admission receipt.
Cross-host scores stay separate; this is not a claim of bitwise equality or a
replacement for cached354. The <=1pp desired physical target remains350/411.

L4's joint5150 trial actually ended after51epochs/608.9fitseconds without an
accuracy gain (best remains the224 parent, last153). The subsequent weighted
TRAIN-prior conditioned head reached TRAIN100% but last TEST100/411, so that
last PT is rejected; best still224 with only a tied-accuracy NLL change. This
supports stronger noise/overfit control, not indefinite extra epochs. One
additional TRAIN-estimated pooled within-class covariance shrinkage readout is
authorized within the existing electronic-head scope. It folds into the same
5140 parameters, uses TRAIN priors and at most25 TRAIN temperature updates, no
TEST-derived covariance or regularization grid. Preserve parent if no full411
gain. Profile `lab_caltech20_L4_residual_shrinkage_profile_20261019.json`.

2026-10-19 overnight authorization keeps both tracks active without a recurring
monitor. All L4 optics/8055 physical CCDs are complete and frozen. Its original
fullphysical128/411 improved to224/411(54.5012%) by the existing5140-head TRAIN
covariance-conditioned fit; strictCPU full-graph replay passed. This is not the
desired342/411 and is TEST-informed development. Next joint original5150 CPU
fit runs independently of newL2 hardware, at most80epochs/600fitseconds. Preserve
the224 parent if no improvement. No new L4 capture or inference layer.

An evidence-driven follow-up addresses physical TRAIN's60/class sampling versus
the original3141 TRAIN distribution. Only the latter's independently verified
TRAIN counts set importance weights, never TEST/VALID frequencies. Profile
`lab_caltech20_L4_residual_terminal_prior_profile_20261019.json` allows one
240step/120fitsecond, original5140-head-only weighted CE/LS.02/correct-TRAIN KL.02
and anchor.0001 fit. TRAIN covariance reparameterization is analytically folded
back into the same Linear256to20 head. Full411 CPU accuracy-first/NLL-tie selection,
all nonhead bytes frozen, best/last/predictions/history/dataSHA retained. This
is a hypothesis, not a promised gain. The already-running joint5150 source must
finish before its newer file version is synchronized to the lab.

2026-10-19 latest clarification: L4 has all8055 CCDs, so its terminal electronic
optimization runs on CPU separately from hardware. `--mode capture-queue` starts
newL2 then newL6 on the singleSDK; it does not wait for L4's electronic accuracy
target. No L4 recapture. Each new2/6 stage still fits downstream masks before the
next fresh stage. Fully measured L4 original baseline128/411 is preserved.

Latest human approval supersedes prior L4->L6 planning and the earlier5pp middle
tolerance. This chat owns residual only; no-residual data/code belong to another
active task. No recurring monitor or new branch/worktree/project copy.

| Order | Frozen simulation | Exact initial checkpoint SHA256 |
|---|---|---|
|L4 current|346/411=84.1849%|`7b0d7f055a16f7f6bbf9858c9ecaea572ecd0b0d6e96c618630f4a4e077c09a5`|
|L2 new|337/411=81.9951%|`b08923b3b43731e2832fc0251a54af9c52535226176070b92b4954f636614c69`|
|L6 new|354/411=86.1314%|`eb3d25718b36f3355a242c83b6e972a11eb2fb0bcd751ef89e86cc2d0298c85d`|

New2/6 versions are from `reports/caltech20_rich_targeted_20261018/targets2_337_6_354`;
their final changes trained only the existing5140 readout, not optical masks.
Nevertheless the user explicitly requests fresh L2 acquisition, and this run
does NOT reuse any old L2 or other-depth CCD. Each model keeps its own fresh
router and layerwise measured-prefix chain. Data remain TRAIN60/class1200 plus
all411TEST, no VALID. L2 needs4833 frames; L6 needs11277. L4 preserves every
already captured valid frame and switches only at a verified complete-stage,
SDK-free, training-child-exited boundary. No hot editing its loaded closure.

Entry `lab_caltech20_residual_sequence_20261018.py`, profile
`lab_caltech20_residual_sequence_profile_20261018.json`. Each complete stage's
TRAIN then TEST is followed by remaining-mask/original-electronic adaptation.
Freeze all acquired optics and input-producing parameters. One TRAIN epoch is
required before early advance; if the best full CPU TEST is within10percentage
points of that model's frozen simulation, proceed. Otherwise at most30epochs
or600seconds plus one bounded local head repair, then disclose the remaining
gap and proceed without endless middle search. This10pp figure is a soft
progression target, not a reason to disable signal/identity guards. Minimum
middle correct counts are L4 305, L2 296, L6 313.

At the final fully measured stage, fit only the existing terminal5150 parameters
(last9-electronic-kernel+alpha+5140head), at most120epochs/900seconds, then
prioritize a bounded original-head local repair. Desired <=1pp final drops need
L4>=342, L2>=333, L6>=350 correct, not guaranteed. Fixed-feature local repair
uses TRAIN-only one-vs-rest binary CE atT2,LS.02, correct-TRAIN teacherKL2,
anchor.2 and full-curvature Woodbury directions with.25step/.08trust radius.
TEST errors/features guide class and same-class TRAIN-neighbor selection;
64TRAIN batch:16 nearby positives,16 random positives,16 hard negatives,
16global replay. Max12classes/128steps or180fitseconds. Unlike margin1 hinge,
binary CE still supplies a signal on the saturated TRAIN100% stage observed in
L4; this is a hypothesis to test, not claimed performance improvement.

All full411 CPU selections use accuracy first; equal correct count requires
NLL decrease>0.0001. Lower correct count is never adopted. Save all parent,
best/last, plans/actual TRAIN indices, strictCPU replay, frozen state and data
hashes. These are explicitly TEST-informed development results, not independent
generalization. Final comparison must be fully measured, not prefix/simulation.

New L2/L6 admission checks all physical TRAIN1200 and TEST411 against their
original source model, portable reference and experimental bridge, preserving
canonical original propagation buffer for simulation-only numerical parity.
The hardware remains2000us/GainX4/wait240,20261015 coordinates,1016nativeactive,
zero-border BMP, original amplitude inverse and linear-voltage phaseLUT; no
claim that the latter is calibrated linear phase. p99<15 orsat>15% stops,
preserves data and requires bounded diagnosis, never blind retry/exposure edits.

2026-10-19 clarification explicitly retires the old L2 residual capture, not
no-residual data. Validate zero process references, preserve every receipt,
phase/report/hash and PT/assets/hardware release, then send only its exact
superseded `caltech20_rich20_L2_layerwise60_20261018/capture` to Windows Recycle
Bin. This removes the old operational CCD paths while remaining recoverable.

When both final TRAIN and TEST are low, first do one bounded full-batch TRAIN
readout optimization rather than assuming a few wrong-class boundaries. Use
TRAIN-only covariance preconditioning (eigenvalue floor2%mean variance), folded
analytically back into the existing5140 head, not a new inference layer. At
most120L-BFGS iterations/120seconds, then bounded local binary repair. Preserve
parent and only accept fullCPU accuracy/NLL improvements. This is additional
experimental adaptation, not a demonstrated gain or an isolated-method ablation.

Historical initial cleanup assessment: Read-only inventory at23:28 found
~5.90TB disk free and~520GB available RAM, so no pressure requires deletion.
Do not delete current L4 CCD, old hardware release/LUT/selection identities,
or no-residual CCD still used by another task. Old L2 evidence remains until
replacement is verified and a recoverable, SHA-verified archive exists; cleanup
must not delay acquisition. No deletion is implied by a newer PT filename.
