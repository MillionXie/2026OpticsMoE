# Latest residual experimental order: L4 -> new L2 -> new L6

## New small-camera authorization, 2026-10-19 (supersedes offline-only scope)

### Same-phase contrast admission PASSED; full stage0 capture actually running

New contrast protocol completed at fixed second candidate40004us/gain1.
20ms failed (mincontrast13), 40ms passed all40 TRAIN: minrawp9950,
mincontrastp9926.5, black-repeat RMSnoise.43394gray, maxclip.00009191.
Model-phase backgroundp9936 leaves headroom. Independent all40 ID/labels/
canonical/raw/crop statistics, both unclipped black repeats, background mean/
noise recomputation and report/hardware/phase SHA checks passed. Original
failed absolute-dark protocol remains FAILED and all evidence is retained.
New per-stage hardwareSHA8ba8189a9e99cc053790177da779d8c7ea810c1cfde9c4c6b7b146fbe63e5f72;
no new LUT/geometry/orientation/input scaling/network dark subtraction.

Published source f5c52d7f197e3cf35169d399f2874acc3970a82c synced main/GitHub/
lab/server,20 tests plus original layerwise selftest passed. Explicit one-stage
controller `L4/stage0_capture_adapt_controller_20261019` actually launched;
first readback173TRAIN receipts (40verified probes+133new), latestnetp9937,
rawp9962/no clip, no failure. It completes1600TRAIN then411TEST, auto runs the
bounded uncaptured-suffix adaptation, then stops for outcome review before
another layer. No recurring/Windows scheduler. Old one-frame stage0 attempt
archived within capture directory, not deleted/reused. Existing router2011
remains unchanged at5ms. L6notstarted.331/411 remains ROUTER-only measured
hybrid, not full physical L4 accuracy. Latest counts are in original run state.

### Absolute-dark exposure comparison FAILED; fresh same-phase contrast check

The five predeclared exposures produced min TRAIN p99 7/13/25/50/99.17,
with negligible clipping and near-linear known-gray above-black response.
However, absolute black p99 rose3/6/10/19/36, so NONE passed the original
max8 rule. All218 diagnostic records/failure retained, SDK released, no
hardware admission or formal stage0 capture inferred from these results.
Exposure-dependent background is observed; sensor dark signal versus optical
leakage has not been separated and neither cause is claimed as established.

A separately named finite protocol `lab_caltech20_DVP_stage_contrast_profile_20261019.json`
tests20/40/80ms, first passing only, using fresh same-model-phase zero-input
repeats and the same fixed40TRAIN (two/class); no TEST. Background mean is
stored for QUALITY DIAGNOSTICS ONLY. Raw camera crops remain unmodified in
the original model and CCD arrays; no network background subtraction.
Criteria fixed before this measurement: background p99<=64 (at least75%
uint8 headroom), no background clip, every TRAIN rawp99>=25, clip<=1%,
contrastp99>=25 and contrastp99/max(1,RMS black-repeat difference/sqrt2)>=5;
same existing known-gray linearity tolerance. Formal capture retains raw15/
clip15% guards and ADDS contrastp99>=15. Old failed protocol is not relabelled.
New hardware identity includes exact background/phase/report SHA. Fresh40
probes only may be reused with full original ID/raw/input/phase/PT checks.
If no candidate passes, stop with all evidence preserved, not another scan.

### Explicit per-layer exposure admission authorized; L4 stage0 next

User replied "继续推进，标定...ok" to the explicit TRAIN/known-pattern-only
per-layer exposure request. This supersedes the old fixed5ms stop for NEW
layers only. Existing measured router2011CCD remains at5000.5us, untouched.
Current selected uncaptured suffix PT is exact1f78217f.../331 hybrid; not old
1ed505 parent. Calibration/next stage must use this new suffix and its phase.

Entry `lab_caltech20_dvp_stage_exposure_20261019.py`, fixed config
`lab_caltech20_DVP_stage_exposure_profile_20261019.json`: first40TRAIN,
two/class20classes, no TEST selection or TEST capture in admission. Ladder
5/10/20/40/80ms, gain1, samewait240/6discard/ROI/flip_v/phase/amplitude inverse.
Choose FIRST candidate with all40p99>=25 and clip<=1%, darkp99<=8; unsaturated
known gray64/96 controls must agree with exposure-linear transfer within
max2gray/15%. At least one usable control, no relaxed guard or automatic retry.
Stop escalating if a TRAIN sample clips>15%; fail if no ladder candidate passes.
Original session recheck first; camera device LUTs unchanged, phaseLUT remains
linear voltage only. No input gain, peak normalization or network dark subtraction.

Each admitted layer gets immutable hardware JSON+SHA, exact PT/phase/data and
calibration-report SHA. Prior measured layers verify their OWN hardware SHA,
not incorrectly relabel all CCD as one exposure. Same selected40 probes may be
reused only after ID/PT/input/phase/raw/crop/exposure audits; report explicit
1971new+40reused=2011unique stage entries. All rejected candidates retained.
Old stage0 failure(one low-signal frame under oldPT) is archived IN PLACE as
`capture/0_failed_low_signal_20261019`, never deleted or used by new PT.

One-stage admission->full TRAIN1600 thenTEST411->suffix adaptation, then manual
outcome review before next layer. Stage0 adaptation profile
`lab_caltech20_DVP_stage0_adapt_profile_20261019.json`: original remaining
masks/post-stage electronics/directalpha/head only, all measured optics frozen.
TRAIN teacher uses frozen parent and SIMULATES only the newly measured stage,
preserving earlier331-prefix performance without treating new physical errors
as teacher targets. T2KL.15, LS.02, limited class emphasis.25; at most20complete
epochs/900fitsec, aim8pp/min2epochs, no target guarantee. No TEST gradients or
VALID; TEST-informed diagnosis/selection remains development, not generalization.
Fresh isolated SDK processes, no recurring task, no other group or L2 changes.

### Latest verified outcome:331/411 hybrid, awaiting exposure decision

Conservative comparison completed5epochs/8000TRAIN presentations, stopped on
the predeclared330 target at331/411=80.53528%. Strict saved-best/last CPU and
independent artifact checks passed: macro69.81796%,NLL.75129592 versus parent
324/411=78.83212%,macro64.31118%,NLL.83808994. Net+7correct comprises16new
correct and9lost; do not claim every old correct prediction was preserved.
Fixed346 simulation gap now3.64964pp. SelectedE5 is real training, notE0.
BestSHA1f78217fb1cf8d1d489a55ce437ee2d47aa15808360db3ee9f30b496849c14ce;
lastSHA63744ba930ac916f61e99417b7eb442e45d03b048279f51fba4fe6f88cf2ea15.
Teacher1600TRAIN-only probabilities/SHA, bestlast, histories, reports and all
411ID/label/probability files retained in `L4/fits/router_conservative_20261019`.
All measured-router parameters remain byte-identical; uncaptured suffix changed.
Any next-stage CCD MUST use this exact selected suffix PT/newphase lineage,
never the old1ed505 parent/bad stage0frame. Both parent and failed frame retained.

This is measured ROUTER plus SIMULATED SUFFIX, NOT full L4 physical accuracy.
TEST-informed development, not independent generalization. Camera5ms control
recheck passed(whitep99152, flip_vPCC.92160, togglePCC.89687); notUSB loss.
Known-white linear-gain estimate for failed stage0 input predictsp997.74 versus
observed8, supporting low dynamic range rather than proving a hardware fault.
No exposure change or signal-guard bypass has occurred. Requested user decision
on TRAIN/known-pattern-only per-layer exposure admission; until answered,
formal capture is stopped at the preserved first stage0frame and L6notstarted.
Refinement worker exited andSDK released; no recurring/background supervision.

### Router complete; stage0 low signal; active refinement authorization

L4 router1600TRAIN/411TEST fully captured/audited. Measured router+simulated
suffix CPU324/411=78.83212%, versus fixed346/411=84.18491%, gap5.35280pp.
Default fit stopped after1epoch due already passing8pp: epoch1 produced321,
selectedE0 remains324 with ALL parameters equal to initial. This is NOT a
trained improvement or full physical task accuracy. The next stage0 first
TRAIN producedp99=8/mean2.609 and stopped at1frame; all2011router+badframe
preserved, SDK/parent/child exited. L6 not started. No blind retry/guard bypass.

Exact generated stage0 amplitude matches saved input(max97/255,p9961/255),
original phase/PT lineage verified. Software diagnosis indicates low optical
drive is a plausible contributor, not proof of camera/optical hardware failure.
An independent same5000.5us known-pattern recheck is admitted; no new exposure
or LUT. User decision requested for TRAIN/known-pattern-only per-layer exposure
admission, separate hardware identities, never a TEST-score-driven search.

User now explicitly requests active outcome review and additional optimization
instead of unattended default tuning. One bounded conservative comparison:
`lab_caltech20_DVP_router_conservative_profile_20261019.json`, source baseSHA
pinned; phase/electronic/headLR2e-4,alpha1e-4,LS.01, class emphasis.25, TRAIN-only
frozen-parentT2KLweight.3. At most6completeepochs/480fitsec, target330 stop after
minimum2, no guarantee and no automatic extra candidate/next-stage capture.
Original architecture and frozen measured upstream unchanged; best/last/full
411TEST CPU/frozen audits required. Parent324 protected, independent named fit
subdirectory; teacher1600TRAIN probabilities and SHA retained. TEST class error
diagnosis/selection is development leakage, no TEST gradient/VALID/TEST teacher.

### User USB replug: recovered reads, finite explicit resume

User physically replugged USB. Windows normal M3S2000M/SDK count1 restored;
independent8frames passed. Camera power-cycle reset exposure to40004us. Before
any new task CCD, original admitted5000.5us/gain1 were explicitly restored and
14monotonic MONO5480x3648 frames/readback/hardware flags passed. These are
diagnostic frames, not new task results; original hardware/LUT identity stays.

Entry adds explicit `resume` (never automatic retry), with a separate recovery
controller in the SAME run. Old failure/gate/pilot remain intact. The empty
failed router attempt is verified0frames/noarrays then renamed in place to
`capture/router_failed_start_20261019`, never deleted. Exact matching24pilot
images may now be explicitly reused as first24TRAIN entries after per-ID/PT/
data/phase/canonical/raw/array audits and fresh generated-input equality.
Reports distinguish2011unique measured entries from1987new+24reused captures.
This explicitly supersedes the earlier separate-pilot-only full-shot rule for
L4 recovery; no old-camera or mismatched-checkpoint CCD is admitted.

Every capture stage uses one fresh child process/SDK lifetime, no re-open loop
in the controller; SDK must release/child exit before suffix fitting. Before
the resumed router capture, SAME connection first verifies dark/white/asymmetric
marker/toggle at ORIGINAL settings, original LUT unchanged. Predeclared drift
tolerance10% of original white-dark span, dark<=max3gray or10%, white clip<.1%,
orientation SAME flip_v/PCC>=.85/margin>=.05 (ideal marker margin.0847), toggle
new>=.85 and new>previous+.1. Any failure stops, retains evidence, no refit/
exposure search/model-score selection. L4thenL6, original model/suffix budget/
TRAIN1600/TEST411/VALID0 remain unchanged. New-camera read recovery is NOT
proof that full acquisition has resumed; actual progress receipts must confirm.

### Actual attempt stopped at USB device loss, 2026-10-19

Source7662f6366 was published and synchronized to lab/server. Fixed data1600
SHAe6fae987... and all92augmentation replays independently verified locally;
no TEST/VALID sources. L4 full1600TRAIN/411TEST original compiled reference
versus deployment bridge CPU maxerror0, TEST346/411 matches frozen simulation.
Actual24TRAIN router pilot completed33.54s, signals and all24ID/PT/phase/crop
array audits passed; arraySHA5d0ef330... . These are valid new-camera data,
not a full physical task accuracy. On the subsequent full-stage camera open,
`dvpStart=-1001` occurred before any additional formal frames. Controller exited,
SDK lock released. Both SDK versions now enumerate0 cameras; Windows reports
USB BootLoader Device VID04B4/PID00F3, not the normal M3S2000M camera. Root
cause (driver/USB/firmware/power) NOT proven. One discovery rescan and one
exact bootloader-device restart returned successfully but did not restore the
normal camera. No hub reset, driver reinstall or firmware flashing performed.

**Blocked on camera USB power-cycle/normal enumeration; do not claim active
full acquisition or tuning.** All24pilot frames/receipts and failed-start
evidence preserved. Run `recovery_handoff_20261019.json` is authoritative;
no automatic retry/recurring task. Before resuming, revalidate camera admission,
retain old failure record, reuse matching24 valid pilot frames rather than
repeat them, and audit a finite per-layer SDK lifecycle (persistent connection
or isolated hardware process). Do not blindly rerun the original launcher or
delete its gate/pilot. L2 frozen329, no-residual groups and old SHS data unchanged.

### User approved fixed TRAIN augmentation; fresh80 workflow

User explicitly approved1508 original unique TRAIN+92fixed augmented inputs
to80/class. Entry `lab_caltech20_dvp_layerwise_20261019.py`, profile
`lab_caltech20_dvp_layerwise_profile_20261019.json`. Original TRAIN3141 only;
per-class source order fixed bySHA256 ofseed-tag+ID, cap80, missing slots use
deterministic integer translations of<=2pixels in150x150 with reflection pad.
Every source ID/index, dx/dy and resulting pixelSHA is retained. Round-robin
class order gives20-class coverage at every20samples. Full411TEST is byte
unchanged; no VALID export or supplementation. No claim of1600 independent
original images. Data/weights remain runtime assets, never Git.

New L4thenL6 begin at the unchanged FINAL SIMULATION PTs7b0d7f.../eb3d25...,
not old-camera-adapted masks. Old original compiled references supply canonical
propagator buffers and independent full TRAIN/TEST same-host bridge gate;
fixed cached scores346/354 are recorded separately from native CPU counts.
Last batch zero-padded to8; dummy samples never captured or scored. Each model
has a separate24TRAIN-only router pilot, then full2011-frame stages, TRAIN1600
immediately followed byTEST411. L4five=10055, L6seven=14077; full total24132,
plus48pilot frames. Pilot frames not silently reused as the full dataset.

After EACH layer, freeze captured upstream, optimize only original remaining
phase masks/electronics/direct-f64-alpha/head. CPU2threads, existing lab4060
only after rejecting foreign compute users/SDK release. Original architecture
and bounded amplitude/phase encoding unchanged. Midmax30epochs/900sec,
finalmax120epochs/1200sec, full epoch boundary; aim<=8pp, carry selected best
after finite budget even if missed and disclose. All1600TRAIN traversed once
per epoch. TEST class-error fraction yields fixed-per-epoch TRAIN CE weights
1+.5*error, plus.03label smoothing; weights recorded. This is expressly
TEST-informed development/selection, NOT independent generalization or TEST
gradient. No VALID, new inference layer, exposure tuning from task score,
periodic PTs, scheduler, automatic restart or new branch/worktree.

Exact best/last strict CPU and frozen-state SHA audits are required before
advancing. Next capture requires previous fit bestSHA and parent-capture PT
lineage equality. New-camera per-frame raw/crop/input/phase/PT/data/hardware
receipts and per-stage full2011ID/crop-array audits retained; darkp99<15 or
saturation>.15 fails closed, all evidence preserved. Only redundant transient
BMPs deleted after canonical/LUT/input SHA records. Raw PNG compression1 changes
storage only, not pixels. Need>=600GiB free before queue. L2/nores untouched.

**Actual known-pattern admission completed,37patterns; SDK released and worker
exited.** Readback5000.5us/gain1, screen-to-canonical `flip_v`; asymmetric-marker
PCC.92055 versus next.79833. Four toggles match new pattern(.8945..9211), not
previous(.0596..0854), atwait240ms plus6discard/one retained frame. Pure device
capture~.51..55s/frame excludes PNG/receipt/model overhead and is NOT a full
acquisition ETA.1ms underexposes;10ms clips4.53%,20ms clips68.91% of white
pattern pixels, so historical20ms is rejected, not reused. Session inverse
held-out intensity RMSE.002605/maxerror.005369; hardware LUTs unchanged.
This is calibration evidence, NOT a trained-mask pilot or accuracy result.
Admission metadata+three crop previews(9files) downloaded andSHA-verified;
all raw calibration frames remain on lab PC. Hardware manifestSHA
3b89d610d939141dec9c5d0955f23765454cd7e7bebfe1c92c5e64e5376713ae.

Original TRAIN3141 SHA38b719... was actually re-read on server: capped80 gives
1508UNIQUE originals.10classes need92additional transformed TRAIN inputs to
reach1600. At admission completion, no model pilot/full collection or fitting
had started. Augmentation is now explicitly approved above. Never
interpret these calibration frames as part of the forthcoming model dataset.

L2 is frozen at329/411. User replaced the unavailable high-speed camera with
the historical small DVP camera and explicitly requests fresh L4 then L6
RESIDUAL layerwise capture, each layer's TRAIN+TEST consecutive, then fit the
remaining uncaptured masks/electronics and select a TEST development PT before
the next layer. Aim <=8 percentage-point gap per intermediate prefix, not a
guarantee; no VALID or TEST gradients. Original alpha/model/encoding protected.
Do not restart old SHS fits or mix any old-camera CCD into the new session.

New coordinates `E:/code/guest/20261019.txt`, SHA5d97a75dc7a39e7761d53c38b8e6123ae052896689d877cd391fc3a1292d05bc,
screen TL(981,152),TR(4327,177),BR(4343,3491),BL(971,3485).
Actual existing DVP camera-only8frames: raw5480x3648 uint8 MONO, gain1,
exposure1009.8333us, all hardware flip/rotate/opposite flags false, SDK released.
Unknown held display was dim; this is not a calibrated optical-failure claim.
Current lab SDK locks/processes absent. Historical DVP20ms/wait240/6discard
is only an admission starting point, NOT reused as verified current settings.

New entry `lab_caltech20_dvp_20261019.py` and calibration-only profile
`lab_caltech20_dvp_calibration_profile_20261019.json`: predeclared exposure
ladder1/5/10/20ms, known asymmetric pattern for orientation and alternating
pattern freshness, session-only amplitude inverse with held-out radiometry.
No model/TEST score used for calibration, hardware LUTs unchanged. Fail closed,
keep raw frames and receipts, no blind retry. Model capture NOT started merely
because the eight-frame device test passed; calibration, source release gates
and new dataset manifest must all be admitted first.

User asks80TRAIN/class. Original TRAIN3141 contains classes with68..79 unique
images. Do not fabricate80 independent original images, duplicate without
disclosure or borrow TEST/VALID. Await explicit choice between original-TRAIN
only fixed augmentation to80 inputs is now approved above. New camera session root:
`runs/hardware/caltech20_residual_DVP_46_20261019`. No new branch/worktree,
lab only receives Git, old CCD/PT/evidence preserved, no recurring automation.

## New offline-only authorization, 2026-10-19

### Latest scope: freeze L2 at329; L4 residual offline gap<=10pp

User fixes L2 329/411 and authorizes L4-only offline optimization from279.
Fixed L4 simulation346/411=84.18491%, <=10pp requires305/411=74.20925%.
No camera/SDK/newCCD, no L2/L6/no-residual change, no recurring task. Preserve
all previous PTs/data and formal direct-alpha bounds(.4001001,.95).

Current native L4 data confirm TRAIN~58-60%, TEST279, last alpha at floor
.4001001,9-conv all strongly negative and head norm107.97. Existing head-only
Huber/BCE/protective/SAM/low-rank/jointNewton already stopped; do not revive
these same plateau searches blindly. New TRAIN diagnostic measures terminal
tanh saturation and compares original stopped-RMS surrogate versus exact RMS
derivatives; no TEST objective. One new original5150-terminal comparison uses
native478 fields, differentiable RMS in BACKWARD only and TRAIN covariance
coordinates folded into the ORIGINAL head each forward. Inference original
architecture/forward unchanged; synthetic forward byte equality at all3alpha
values, derivative finite differences and5140-head folding tested. Never use
the previously rejected approximate compact alpha cache.

Entry `lab_caltech20_native_terminal_20261019.py`, profile
`lab_caltech20_L4_native_terminal_profile_20261019.json`: CPU2/batch8, original
last9conv/directalpha/original5140head only, max20epochs/600fitseconds, target
305 stop. All optics and pre-final electronics frozen, native complete
TRAIN1200/TEST411/noVALID. Correct-TRAIN T2KL2/terminalanchor.1, head coordinate
anchor.01/radius sqrt20*.4, fixed TRAIN seed; save best/last, selected native
features and strict original raw-CCD CPU checks for both states. TEST is
diagnosis/selection development, never gradients; no guarantee of305.

Actual native-terminal TRAIN probe:41.47% conv preactivations |v|>3; RMS
surrogate/native gradient cosine .9999528, so no evidence that backward
detachment alone explains poor transfer. First10 epochs TRAIN~78-80% while
TEST194-207/411, retaining E0 279; this is NOT performance improvement.
One bounded population-prior comparison uses only the previously SHA-verified
original TRAIN3141 counts (source38b719...), not TEST/VALID frequencies. Capture
TRAIN stays1200/60each. Same native5150 method/coefficients; only CE class
weights change. Profile `lab_caltech20_L4_native_TRAINprior_profile_20261019.json`,
max12epochs/360fitseconds, no continuation or seed grid. Start only after first
worker exits and its strict CPU best/last audit completes. L2 remains329.

To restrict the observed joint-head overfit, one final ORIGINAL-head contrast
uses only19 directions from class means of TRAIN1200, in the same TRAIN-only
covariance geometry. Only400 optimizer coordinates, folded into original5140
head; no new inference layer or TEST-derived basis. Same original Newton-CG
coefficients as prior full-head trial; max256 TRAIN batches/90fitseconds, no
scan/automatic continuation. Config `lab_caltech20_L4_class_subspace_profile_20261019.json`,
original joint trainer adds a config-controlled TRAIN-only basis. Orthogonality,
gradient/HVP and exact algebraic folding tested. Frozen native279 prefix only.

TRAIN-class-subspace CE completed256 actual updates and retained E0 279.
A single loss-only contrast in that SAME400-coordinate subspace uses smooth
multiclass Huber margin1/delta1, preserving correct-TRAIN T2KL2 and all prior
coefficients. It stops updating already sufficient TRAIN margins instead of
fitting every confidence value. Analytic gradient/HVP checked to1e-10 against
autograd. Config `lab_caltech20_L4_class_margin_profile_20261019.json`, max256
TRAIN batches/90fitsec, no automatic continuation/parameter scan/new layer.

The native5150 TRAINprior fit also retained E0 279 after12epochs/1800batches;
subspace Huber completed256 and retained279. One finite checkpoint-granularity
audit now EXACTLY replays the first30 recorded TRAINprior minibatches. First
full epoch had already regressed to228, so previously unevaluated microsteps
may have been missed. Same parent279/source coefficients/seed/sample stream;
every TRAIN index batch asserted against the original saved history. Evaluate
each TRAIN-derived checkpoint with TEST selection, no TEST derivative, no new
seed/direction or unbounded continuation. Only best/last saved, profile
`lab_caltech20_L4_early_checkpoint_profile_20261019.json`, max30updates/180fitsec.

### Latest explicit user request: L2 residual gap <=2pp, 2026-10-19

**Completed:329/411=80.04866%, fixed337 simulation gap1.94647pp; new target achieved.**
One actual continuation segment,1370 TRAIN batches,325->329 (newcorrect7,
lostcorrect3); target stop, no further training. Best PT SHA
`6d3b796d188aa2aa7a3fed5bd4df42be61ddb9a46c8eb445c33644eee6480f18`.
Local run `runs/hardware/caltech20_residual_L2_gap2_20261019/whitened_continuation/stage02`.

Laboratory ORIGINAL full measured-graph CPU replay of this exact PT completed:
TRAIN982/1200=81.83333%, TEST329/411, macroTEST68.68977%, NLL1.03209317.
All three recorded CCD stages(router/0/1), all1200TRAIN/411TEST identities,
dataSHA74a434... and unchanged original optical/nonhead state verified.
Local sealed-head versus laboratory full-graph maximum probability error
2.17557e-6 <=3e-6. Remote audit path in original sequence run:
`L2_gap2_CPU_review_20261019/full_CPU_review.json` (SHA
`7a8d80c2fb23d440b2a160ac37ff50b4492df54fb232ce6d1380b78930838ff0`).
Its full TRAIN/TEST predictions and report downloaded and SHA-verified locally.
Independent local audit verified25 files, best/last PTs/predictions, complete
TRAIN sampler ancestry2268 prior+1370 new, and every nonhead tensor unchanged.
Best/last parameters are byte-identical, so reuse the same full CPU result for
last rather than pretend a separate replay. Both325 fallback and all old data
remain protected. TEST-informed development, not independent generalization;
no TEST derivatives/VALID/newCCD/SDK/GPU. Full review used original unchanged
graph sourcead6ff7e; current profile source01f0938f published on main.

User authorizes a new bounded L2-only attempt after325/411 achieved the old
<=3pp target. Fixed simulation stays337/411=81.99513%; <=2pp requires at least
329/411=80.04866%, gap8/411=1.94647pp. Preserve325 as fallback. No camera/new
CCD/SDK, original optical/nonhead tensors frozen, TRAIN1200/TEST411/noVALID,
TRAIN-only gradients and TEST-informed development diagnosis/selection.

First continue the successful TRAIN-whitened Huber row method from325 with
the EXACT prior TRAIN sampler stream reconstructed against all saved batches;
no new seed or coefficient. Profile `lab_caltech20_L2_gap2_cont_profile_20261019.json`,
max3 segments(each<=3passes/180fitseconds), first no-progress/329 stop, internal
09:25 bound. This fresh user authorization supersedes the earlier L2 freeze
only for this L2 attempt; does not revive stopped L4 searches or hardware work.
Local sealed features remain explicitly distinct from a fresh laboratory
raw-CCD full original-model CPU replay. Git/source and artifactSHA checks apply.

### Completed local round, 2026-10-19 09:00

Full measured-graph CPU results verified before the connection interruption:
L2 325/411=79.07543%, gap2.91971pp from337, target achieved;
L4 279/411=67.88321%, gap16.30170pp from346, target305 NOT achieved.
Local sealed-feature recovery reproduced both counts. Protective replay tied279
with NLL1.76428->1.75090, then stopped first no-progress segment. The distinct
coupled Newton-CG comparison completed256 TRAIN batches and retained E0 279;
no accuracy gain, not adopted as a new improvement. No jobs left running.

Local private run `runs/hardware/caltech20_residual_sealed_local_20261019`
contains best/last, complete histories, original1200/411 IDs and predictions.
`sealed_local_audit.json` verified10 completed fit runs, every best/last SHA,
original-head probability composition<=3e-6, and every nonhead tensor identical.
112 local files are SHA-manifested in `sealed_local_manifest.json` (manifest
SHA4342966ef0a305ccb449ff70d28d9e28d5d20c91326ca697dfc8dedb0f4c93a9).
These local recovery PTs are not claimed identical serializations of laboratory
09339.../7e513... files or a fresh full raw-CCD replay.

User reported computer restart. One short successful connection confirmed
no owned workers/SDK lock, created a97,959,529-byte laboratory evidence archive
(158files, SHAd3b4e675014262edd4d34ca217db3bfd6a8ea11f2f1a6ee4e0db76af15a7c584).
Download was interrupted after14,712,832bytes; preserve it as incomplete, NOT
a verified backup. Bounded resume and original raw-CCD review could not connect
again; no review or new training was dispatched. Latest source is on main/
GitHub, laboratory/server Git synchronization still pending restored SSH.
Resume only the manifested archive, then original full-graph CPU review of
local protective PT68f543553b523bf51659afbdd09b979b71c41a0de9f6e539cbf9a6530fad1325;
do not recapture, restart stopped searches, or claim target success for L4.

Laboratory SSH became unavailable after the verified325/279 results. No new
protective-replay laboratory run was dispatched. Those latest PTs are still on
laboratory E:; their local metric/audit summary is saved, but a local PT backup
has NOT yet been completed. Do not claim three-end synchronization or backup.

Local CPU Torch2.4.1/Python3.8 is available without installing anything. The
previous SHA-sealed full measured-graph feature exports and original318/260
PTs are local. An explicit --sealed-backup-root mode admits them only after
manifest/handoff hashes, producer full-CPU report, complete1200/411 IDs,
balanced TRAIN60/class, disjoint splits, all-nonhead byte identity, and
producer-feature/output parity<=3e-6. L4 actual local maxabs1.90735e-6 passed.
All local derived candidates preserve the original head and fixed prefix,
report sealed_feature_composition_verified separately, and set
full_CPU_replay_passed=False/fresh_full_raw_CCD_replay=False. They require no
camera and must not be relabeled as a new full raw-CCD replay. Reproduction of
the fixed already-approved optimizer is recovery, not a new seed scan. L2
target is already achieved on laboratory full replay; no further L2 search.
Source synchronization to laboratory/server is pending restored SSH.

The local fixed-config recovery reproduced269->271->276->279/411 with a
fully reconstructed local TRAIN-batch stream and last-stage E0 stop; L2 recovered325
and stopped. Locally saved PTs are distinct serialized recovery artifacts,
not claimed downloads of the original laboratory PTs. Protective replay
completed10944 TRAIN updates:279 unchanged, NLL1.76428->1.75090; at most3
same-method segments preserve one reconstructed TRAIN RNG and the existing
09:00 internal cutoff, first-no-progress stop. Profile
`lab_caltech20_L4_protected_cont_profile_20261019.json`. No camera/SDK.

Protective continuation stopped at its first no-progress segment: still279.
One distinct comparison addresses coupled class boundaries: all20 original
head rows jointly optimized in the same TRAIN covariance coordinates with
exact softmax CE/correct-TRAIN T2 KL/anchor curvature and12-step Newton-CG.
Original5140 head only; finite256 batches128/90fitseconds, joint trust radius
sqrt20*.4, Newton.25/KL2/anchor.1/damping.1 retained. One fixed TRAIN stream,
no seed/coefficient scan or automatic continuation. Analytic gradient/HVP
checked against autograd to1e-10. Entries `lab_caltech20_joint_readout_20261019.py`
and `lab_caltech20_L4_joint_readout_profile_20261019.json`. No camera/SDK.

BCE continuation finished276->279/411=67.88321%, then next segment had no gain
and stopped. Best stage02 SHA `7e51378473aeaecffaf937b9cd3f5fb9b1f96b6f57bb22ce2bf3a711703921a7`.
Against original260, corrected33 and broke14, net+19; gap remains16.30170pp,
not<=10pp. L2 corrected9/broke2, net+7 and remains325, target achieved.

One evidence-led sampling comparison addresses those destructive changes:
same original257 row/natural coordinates/CE and all coefficients, but negative
batch16 becomes8 TRAIN-risk +8 TRAIN neighbors of at-risk CORRECT TEST queries
(up to2 queries/class,2 same-class TRAIN neighbors/query). These are TRAIN
pixels/features/labels only in the gradient. TEST labels guide selection and
checkpoint development, not derivative; no claim of independent generalization.
No new seed/radius/camera/inference layer. Max3passes/180fitseconds, first
no-progress stop. Profile `lab_caltech20_L4_protected_replay_profile_20261019.json`.

Whitened BCE yielded another REAL fully CPU-verified gain271->276/411=
67.15328%, best SHA `42ad3dcbe85c154c3190c8e29cbf9827c9f9882e11fb497beb86af6200e4a102`.
All three passes improved:272/275/276, NLL1.89639/1.83387/1.82134;
10944 TRAIN updates. Original optical/nonhead state identical, not a new capture.
L4 alone gets max3 finite same-method segments under
`lab_caltech20_L4_whitened_BCE_cont_profile_20261019.json`; replay all saved
TRAIN batches and keep one RNG, stop first no-progress or305. No revival of
stopped Huber or pair methods; L2 remains325, no hardware or new inference layer.

Whitened two-row Huber completed2560 TRAIN updates and retained E0 271;
last249 not adopted, fullCPU/frozen checks passed. One final loss comparison
from271 uses existing analytic binary cross-entropy instead of signed Huber,
the same TRAIN-only whitened coordinates/sampler, original one row only.
`lab_caltech20_L4_whitened_BCE_profile_20261019.json`: max3passes/180fitsec,
first-no-progress pass/target stop; no seed/radius scan. L2 remains frozen325.

L4 same-method continuation completed: stage02 269->271/411=65.93674%,
stage03 retained271 with NO gain, reason first_no_accuracy_or_tied_NLL_progress.
Do not restart this single-row chain. Selected stage02 SHA
`34f75a1d34c0126a76cd8a0848767164aafb3487695a8b437603965965f9bdd2`.
Only a new two-row block comparison proceeds from271: independent offsets in
the SAME TRAIN-only whitened geometry, full20 correct-TRAIN T2 KL, signed Huber,
remaining18 rows byte-fixed. Joint standardized radius sqrt(2)*.4 preserves
the prior per-row scale, no radius scan. Max3passes/20pairs/128steps,
180fitseconds, target/first-no-progress stop; exact original-head delta folding.
`lab_caltech20_L4_whitened_pair_profile_20261019.json`. L2 remains325 unchanged.

Whitened local repair produced a REAL gain, fully original-model CPU verified:
L2 318->325/411=79.07543%, gap12/411=2.91971pp from fixed337, achieving <=3pp;
stop L2. Best SHA `09339bf2813c9d8f1910e391c8a98a3c8640c3a8f10f3a78539044129c9ccf8a`.
L4 260->269/411=65.45012%, still below305, fullCPU/nonhead freeze passed.
Both under original sessions `terminal_whitened_local_gap3_gap10_20261019`.
L2 executed2268 TRAIN updates; L4 executed10944 across three passes, not E0.

Only L4 continues for at most3 same-method segments(each <=3passes/180fitsec),
same continuous sampler reconstructed against EVERY saved TRAIN batch index.
Repeated-class plans use ordered stage records, not a last-class dictionary.
No new seed or optimization coefficient; stop first no-progress segment or305.
Config `lab_caltech20_L4_whitened_continuation_profile_20261019.json`; its09:00
internal safety cutoff is not a user deadline or a recurring task. L2 unchanged.

Low-rank and SAM/EMA completed100epochs each for both depths, full original
CPU best/last/TRAIN/TEST checks passed; selected E0 318/260, no accuracy gain.
The actual compact proposals also failed1.0e-5 to3.1e-5 feature parity, remain
disabled; no alpha was changed. Exact native features were used for training.

New evidence-led step: original TRAIN covariance whitening improves nearest
TRAIN same-class counts on wrong TEST from14/93 to20/93(L2),11/151 to34/151(L4),
and TRAIN leave-one-out counts553->694 and244->642 of1200. This is a distance
diagnostic, NOT a new classification score or independent validation. Use that
TRAIN-only geometry for local row sample selection AND optimization coordinates.
Optimize one257 row at a time, remaining19 rows byte-fixed per stage; signed
Huber, correct-TRAIN KL/anchor, curvature solve. Radius.4 is in standardized
TRAIN-logit geometry, not raw weights. Fold DELTAS into original Linear,
preserve byte-exact E0, no whitening layer at inference. No change to alpha,
conv or optical upstream. Max3passes/180fitseconds, target/first-no-progress
pass stop, one continuous RNG, no seed or radius scan. Exact profile:
`lab_caltech20_residual_whitened_local_profile_20261019.json`.

Active-scope correction before ANY real-data fit: compact fusion proposal failed
synthetic feature gates(max1.19448e-4 and6.31809e-6 >3e-6). Do NOT relax the
bound or claim exact alpha optimization. Alpha is explicitly frozen for this
round, even if selected real-data cache cases happen to pass. Only original
exact native features are used by low-rank/SAM head training; original full-model
CPU replay remains mandatory. Rejected compact errors are preserved in tests.
Low-rank TRAIN variables1124 fold to original5140 parameters; SAM5140 variables.
The following earlier alpha-cache plan is historical, overridden by this rule.

User cannot access the optical path/camera now and explicitly requests further
software optimization. No new capture/SDK, hardware/LUT change or changed
already-captured upstream. Residual L2/L4 targets325/305 remain. Prior finite
method stop records remain intact; this is a different new comparison.

Only the ORIGINAL last direct alpha and5140 Linear head may change; last9conv
and all earlier optical/electronic states freeze. First admit a pooled last
fusion cache including per-sample electronic/optical RMS AND their cross moment.
Full TRAIN1200/TEST411 at parent and both alpha bounds must reproduce original
features within3e-6; failure explicitly freezes alpha and uses original exported
features, never a relaxed/approximate inference graph. No VALID.

Two bounded training procedures: rank4 delta folded into original head, and
SAM on original head, each with correct-TRAIN T2 KL, parent anchoring and EMA.
No inference layer added; cache/folding/alpha-gradient tests precede training;
best/last require original full measured-model CPU replay afterward. Each
max100epochs/180fitseconds, target stops. Full411 TEST used only for development
selection/diagnosis, never gradients. Profiles and exact method parameters in
`lab_caltech20_residual_offline_profile_20261019.json`. Source is Git-only.

## Renewed terminal refinement request, 2026-10-19 morning

Outcome of this bounded round: targets remain UNMET. Six completed CPU fits
(three methods per depth),7792 total TRAIN optimizer updates, all exact
original-model full411 CPU replay/freeze checks passed. No new correct TEST
samples beyond318/260. Do not label tied-count NLL change as accuracy gain.

| Original-terminal method | L2 updates / best / last correct | L4 updates / best / last correct |
|---|---|---|
| Independent pair, signed Huber |3072 /318 /312|1536 /260 /206|
| Independent pair, full20 CE |1536 /318 /300|1536 /260 /169|
| Original20 biases, TRAIN-prior calibration |55 /318 /310|57 /260 /238|

Original accuracy-selected PTs remain authoritative: L2
`radius040_continuation_20261019/stage04/best.pt` SHA
`823d81927be3874f23a0b1f4a97cdda7272c8239ea6bae6045650df05e920dfd`;
L4 `huber_continuation_20261019/stage02/best.pt` SHA
`3d9895aa06d32efe484523ab04307a76d072a30c5cd7216fe4bf64ccadc461dc`.
New L2 pair PT only improves same-count NLL1.070204->1.069678; one already
wrong sample changes to another wrong class, with no correctness-mask gain.
All L4 retained per-sample predictions are unchanged; its selected parameters remain
E0. Preserve both new best/last evidence, do not replace the formal accuracy
record with a worse last PT or imply a new robust result.

Error inspection: L2 chandelier1/12, starfish3/10; L4 butterfly0/10 and
menorah1/10, each only5/60 correct on TRAIN at the selected parent. Among wrong
TEST samples, nearest TRAIN feature has the correct class for14/93(L2) and
11/151(L4). This suggests weak class separation/transfer in the current
representation, not proof of hardware failure or a universal accuracy ceiling.
Original3141TRAIN class counts were independently rechecked on the server by
exact original-cache SHA; no TEST/VALID labels used to estimate the prior.

New runs are `terminal_independentpair_gap3_gap10_20261019`,
`terminal_independentCE_gap3_gap10_20261019`, and
`terminal_biasprior_gap3_gap10_20261019`, under each ORIGINAL L2/L4 session.
Their full411 TEST-selected scores are development metrics, not independent
generalization. All six processes returned0; no SDK/GPU or recurring monitor.
The finite comparisons are complete, not continuing silently in background.
Detailed errors and verified artifacts are in the original private run/local
`caltech20_residual_verified_20261019` evidence paths. No CCD/LUT was changed.

User now asks residual L2 to drop at most3pp from fixed337/411 simulation
(requires325/411), and residual L4 at most10pp from fixed346/411
(requires305/411). Starting complete measured-pipeline results are318/260.
No new L6/no-residual task, capture, upstream mask or LUT modification.

One bounded new comparison removes the previous antisymmetric/zero-sum
constraint: two existing head rows may shift independently,514 variables,
other18 rows fixed per block. TRAIN signed Huber margin plus parent-correct
TRAIN full20-class T2 KL and anchoring; margins use same-class TRAIN
neighbors only. TEST confusion selects pairs and same-class TRAIN neighbors;
TEST chooses PT but never contributes gradient. NoVALID; development metric.
Fresh original-model CPU feature/probability parity precedes training;
strict full411 CPU replay and nonhead SHA follow. Existing CCD identity is
unchanged. Profile `lab_caltech20_residual_independent_pair_profile_20261019.json`:
max3passes/300fitseconds, stop first full no-progress pass, no seed scan or
restart of earlier stopped methods. E0 retained if no improvement.

The independent signed-Huber comparison completed3072/1536 actual updates
for L2/L4 with original fullCPU parity, but retained318/260 correct. L2 only
improved tied-count NLL1.070204 to1.069678; this is NOT accuracy progress.
One additional loss comparison uses standard full20-class CE with.02 label
smoothing, retaining the same independent two-row subspace and TRAIN-only
parent-correct T2 KL/anchor. No change to radius, learning rate, seed or
upstream. Its separate profile is
`lab_caltech20_residual_independent_CE_profile_20261019.json`, max3passes or
180fitseconds, stop first no-progress pass. Not an unbounded candidate scan.

The full20 CE pair comparison also retained318/260 after1536 updates each.
One final bounded low-capacity comparison isolates class-frequency calibration:
freeze every weight/alpha/mask and optimize only the20 ORIGINAL head biases.
Use original3141TRAIN counts already recorded in prior profiles, not TEST
frequency estimates. Weighted TRAIN CE plus correct-TRAIN T2 KL/anchor;
at most64 Newton updates/60fitseconds with monotone TRAIN-objective backtracking.
Trust radius derives once from original TRAIN log-prior norm, not a TEST scan.
This differs from earlier full5140-head prior fitting which overfit TRAIN.
Profile `lab_caltech20_residual_bias_prior_profile_20261019.json`.

## Foreground morning status, 2026-10-19, Asia/Shanghai

The final1pp target has NOT been achieved. Full measured optical-pipeline
development results remain L2=318/411(77.3723%), L4=260/411(63.2603%),
L6=165/411(40.1460%), against fixed original simulation337/346/354 correct.
The respective gaps are4.6229/20.9246/45.9854pp. Do not substitute no-residual
results, weaker simulation PTs, or mixed-prefix scores into this comparison.
All3/5/7 captured stages are complete,1200TRAIN+411TEST each,4833/8055/11277
frames. NoVALID. Original CCDs remain on the laboratory drive; local backups
cover PTs, predictions, protocol/history/metadata/phase and SHA manifests, not
a claim that every raw CCD was downloaded toC.

L6 targeted terminal10 retained E0 165. A finite original remaining-mask
continuation from measured router+0+1+2 completed57epochs at its1200fitsecond
boundary plus fullCPU auditing: mixed best171(E0), last127. No new upstream
was adopted, and no downstream CCD was re-acquired. TRAIN reached100% while
TEST declined. A TRAIN-spectrum-conditioned original5140-head comparison also
completed240steps for each depth: all retained E0 318/260/165, all last TRAIN
accuracies100%, last TEST239/107/48. This rejects more fitting of weak modes
as an observed improvement; it does not prove a universal accuracy ceiling.

One original L2 teacher-constrained suffix comparison completed70epochs at the
600fitsecond epoch boundary plus fullCPU auditing: mixed best283(E0), last277.
No new upstream was adopted and no new global CCD was captured. Entry/profile:
`lab_caltech20_teacher_suffix_20261019.py` with
`lab_caltech20_L2_teacher_suffix_profile_20261019.json`, max80epochs/600fitseconds.
It replaces the prepared-but-UNRUN plainCE prefix2 extension. The earlier
prefix2 admission still proves all411 saved-parent probabilities and60TRAIN
native-phase/canonical input identities. Freeze actual router/expert0; train
only existing global1mask and postexpert0 electronic/alpha/original5140head.
Correct-TRAIN-only original simulation teacher featureMSE.5 andT2KL.2 constrain
geometry, with no teacherTEST input or TEST derivative. It remains MIXED and
requires new global CCDs if a genuinely useful new mask is later adopted;
existing fullphysical318 is independently protected. No new inference layer.

L6's fixed354/411 cached simulation uses the SAME original frozen PT whose
original fullCPU count was352(server)/353(lab); the recorded cross-backend
differences do not authorize a different baseline or selection. The gap table
continues to use original354, and no difference here approaches the final gap.

Read-only balanced TRAIN diagnostics found no large fix from numerical FFT
padding518->1024->1536. Native incident/phase identities passed all7 L6 stages;
expert-layer coarse16 Pearson medians were appreciably below router's~.94.
These observations do NOT establish a device fault, calibrated phase response,
or the physical cause. All-stage metadata show TRAINend->TESTstart0.695-1.364s,
class-matched median TEST/TRAIN p99 ratios~1 and all signal guards passed;
different images within each class still confound an optical-drift inference.
Hardware/LUT/exposure/input gain/coordinates/ROI remain unchanged. The existing
linear-voltage LUT is NOT a measured linear-phase calibration. No blind recapture
or automatic phase-LUT substitution follows from the diagnostic hypotheses.

All outcomes are heavily TEST-informed DEVELOPMENT selections, not independent
generalization estimates. Source travels throughGit only, no new engineering
clone/branch/worktree, no-residual work untouched, no recurring monitor created.
The records below are the chronological protocol/execution history, not a list
of still-running jobs or permission to restart no-progress chains.

## Historical protocol and execution trail

Remaining foreground work may exceed07:00 for a new COMPLETE physical result.
Rather than mechanically repeating final-head losses, one finite L6 optical
suffix continuation reopens the ORIGINAL router+0+1+2 measured prefix, at the
first severe intermediate drop (selected mixed171). Existing original `core.fit`
is reused with a tested depth6 wrapper, no source hot patch or inference change.
Files `lab_caltech20_residual_prefix_extension_20261019.py` and
`lab_caltech20_L6_residual_prefix4_extension_profile_20261019.json`;
120epochs/1200seconds maximum on the OWN idle4060, original masks3/4/5 and
postexpert2 onward electronic/alpha/head only. No SDK/capture in this worker.
All411 saved-parent probability parity and60TRAIN incident/native phase identities
must pass before GPU work; captured prefix byte-frozen, no TEST derivatives/VALID.
This remains a MIXED score, never substitutes for fullphysical165. Any later
deployment requires FRESH affected downstream CCDs3/4/5; no old downstream reuse,
automatic capture, new branch/worktree/project copy, exposure/LUT/input-gain change.

L6 joint5150 recovery ended54epochs/615fitseconds, retained E0 165, last145;
paired2304 TRAIN updates retained165, no accepted correction, no continuation.
Direct alpha audit at the selected165 model: [.95,.66805,.95,.93077,.93896,.89199].
One targeted10 block-coordinate fit now freezes the accepted classifier rather
than moving it simultaneously; selected final alpha.89199 has room within the
ORIGINAL bounds, unlike L2/L4's lower-bound parents. Profile
`lab_caltech20_L6_residual_targeted10_profile_20261019.json`, existing tested
TRAIN-only2/2/2/2 sampler, original final9conv+alpha, max30epochs/360seconds,
all upstream/masks/data untouched, strictCPU/freshfeatures/retain E0 on no gain.

L6 Huber chain stopped first no-progress at165/411(40.146%), no restart;
TRAIN-prior full-head comparison retained E0 165, last41 rejected. Original5150
recovery is an independent finite comparison; fresh selected feature exports are
mandatory. If the selected feature producer changes, a new bounded row correction
may use that NEW basis, never stale features. Otherwise do not restart the stalled
row chain: one coupled two-row comparison uses
`lab_caltech20_L6_residual_finalpair_profile_20261019.json`, original257 independent
antisymmetric variables/other18 raw scores fixed, same tested Huber/KL/LM/radius,
finite continuation with reconstructed TRAIN RNG and first-no-progress stop.

Read-only paired last-stage counterfactual at accepted L2/L4 parameters: nominal
last-stage inference after the same measured prefix273/115 correct versus actual
fullphysical318/260. These are NOT new performance results or evidence that the
original simulation models are273/115; readouts are already adapted to measured
fields. They caution against blindly retraining another mask with an uncalibrated
nominal last-stage model. TRAIN-only L2 canonical input audit also rejects simple
gain2: allTRAIN peak codes168(expert)/215(global), gain2 would clip; no gain applied.
No hardware, LUT, exposure, input encoding, CCD or PT modification follows from
either diagnostic, and neither proves the physical cause.

L6 all11277 frames/7stages audited: each TRAIN1200/TEST411, IDs/canonical SHA,
CCD PNG-to-array pixels/native phase/PT identities passed, SDK released. First
ordinary final fit103/411, conditioned+bounded head132/411, fullCPU proof. At132,
TRAIN177/1200, Shannon feature rank2.428 (participation ratio2.029), ALL intended
TRAIN neighbors active at margin1; this is not L2's inactive-positive-margin case.
First signed-Huber original-row correction reached164/411, strictCPU/nonhead
freeze, finite continuous-RNG continuation follows, stops first no progress.
L2/L4 sameclass-Mixup comparison retained E0 318/260, last286/116 rejected.

After that finite L6 row chain, use one original-TRAIN-prior conditioned head
comparison (`lab_caltech20_L6_residual_TRAINprior_profile_20261019.json`) without
TEST prior estimates, then one original terminal5150 recovery comparison
(`lab_caltech20_L6_residual_recovery5150_profile_20261019.json`) from the better
accepted fullphysical parent. Only existing finalconv/directalpha/head; severe
TRAIN+TEST underfit motivates terminal feature adaptation, not extra dropout.
Original masks/upstream/CCD input identity frozen; no new layer/CCD/GPU/SDK/VALID,
no TEST derivatives. FullCPU/fresh selected features/bestlast/freeze SHA required;
E0 retained on no gain. These are finite comparisons, not a loss/LR/seed grid.

One finite TRAIN feature-augmentation comparison uses
`lab_caltech20_residual_mixuphead_profile_20261019.json` from accepted318/260:
same-class TRAIN-only Mixup, half original/half augmented CE, fixed derangement
and lambda[.25,.75], folded original5140 head; no new layer/CCD or TEST sample in
the derivatives. Keep original neighbor-head KL2/anchor.001/120steps/180seconds
unchanged; record every TRAIN pair/lambda, preserve E0 if no fullCPU gain.
These are virtual electronic training features, NOT extra physical measurements.

Targeted terminal10 completed L2 28epochs/L4 27epochs at the360fitsecond
complete-epoch boundary; neither beat accepted318/260, E0 retained, last307/231
rejected. All fullCPU/5140-head/upstream audits passed, fresh feature banks saved.
No restart. A further READ-ONLY phase-response diagnosis uses the same60 TRAIN
samples, explicit11 fixed hypotheses: linear phase range scales .5,.75,1,1.25,1.5,2
and unit-range power curves gamma .5,.75,1,1.25,1.5,2 (nominal counted once).
The current LUT is linear VOLTAGE, not calibrated phase. Record all11 agreement
statistics, exact nominal-field identity, incident/native-phase/receipt checks.
This neither calibrates/adopts a LUT nor changes a CCD/PT/hardware/inference graph;
no TEST or VALID is used, and higher TRAIN agreement does not prove the cause.

The finite D4/sign TRAIN diagnostic found no compelling common orientation/sign
explanation; no hardware or data transform is adopted. A distinct bounded
terminal10 comparison follows the fine-tuning-summary TRAIN sampling design:
fixed accepted318/260 readouts, only original final9conv/directalpha optimized,
batch8=2 same-class TRAIN neighbors of wrong TEST queries,2 same-class TRAIN
replay,2 hard-negative TRAIN,2 global TRAIN. Fixed E0 feature-distance plans,
error-class round-robin, all actual TRAIN indices logged. No TEST derivative,
prior estimate, new layer or CCD. Profile `lab_caltech20_residual_targeted10_profile_20261019.json`,
max30epochs/360fitseconds, KL2/anchor.1, original alpha bounds, retain E0 if no
improvement; fresh selected feature exports/fullCPU/frozen upstream audit.

L2 radius.40 stopped first no-progress after4 new segments: still318/411,
NLL1.0702038; do not claim extra accuracy. L4 triple block completed1024 TRAIN
updates and stopped first no-progress pass, retained260. No restart of either
plateau. Next read-only diagnostic tests only the finite physical ambiguities
D4 image orientations(8) and nominal/opposite phase sign(2) on the same balanced
60 TRAIN samples. Entry `--optical-transform-diagnostic`; no TEST, no PT, data,
orientation, LUT, exposure or hardware write, no fit/calibration/deployment.
Record all16 hypothesis statistics, not just the best, after exact incident/phase
identity checks. A higher agreement may motivate a future calibrated experiment,
not automatically change current acquisition or prove the physical cause.

Joint dropout fits completed44epochs/600fitseconds each, selected E0 318/260;
last275/177 rejected. Fresh selected feature exports/fullCPU/nonhead audits
passed. One evidence-led L2 radius comparison now changes ONLY.20->.40:
at accepted318,16/93 wrong TEST examples cannot geometrically flip at.20 even
under an ideal single-row direction, zero cannot at.40 (maxrequired.34673).
Profile `lab_caltech20_L2_residual_radius040_profile_20261019.json`. This is not
a guarantee those16 recover or a radius grid; gradients still TRAIN only.

L4 has a separate bounded coupled-three-row comparison rather than blindly
increasing radius: original rows a+=v0,b+=v1,c-=v0+v1,514 variables, other17 raw
scores fixed, TRAIN LSCE/full20-class correct-TRAIN KL2/anchor.1, exact curvature
Woodbury/LM validated against autograd. Profile
`lab_caltech20_L4_residual_triple_profile_20261019.json`, max3passes/8blocks each/
128steps/180fitseconds, first no-progress pass stops. TEST confused triples and
same-class TRAIN neighbors inform development; no TEST derivatives/VALID/new
layer/CCD/SDK/GPU. Both preserve E0 and require complete physical CPU replay.

L2 adaptive margin extension stopped FIRST no-progress after9 attempted new
segments:318/411(77.3723%), NLL1.0734336, no further extension02 permitted.
L2 all60 sampled incident PNG/native phase identities passed; L4 diagnostic
stopped on a receipt path convention, NOT missing CCD or signal failure:
its exact original router is adopted through `arrays.train.original_receipts`
under `L4/capture/train/router`, not the new `capture/router/train` array-only
alias. New diagnostic source follows that manifest path within the same session,
verifies adoption manifest SHA and original receipt keys, preserves old failure,
and uses a fresh output on one bounded rerun.

One final bounded joint-electronics comparison from the MUCH better318/260
parents uses `lab_caltech20_residual_jointdropout_profile_20261019.json`:
original final9conv/directalpha/5140head=5150 only, max60epochs/600fitseconds,
TRAIN feature dropout.10 (off in evaluation), correct-TRAIN parentKL2/anchor.1,
slower original readout5e-5. Earlier weak-KL simultaneous fit overfit, so this
protects accepted boundaries while permitting the original terminal feature
producer to adapt jointly. No new inference layer/CCD/SDK/GPU/VALID; class prior
is ORIGINAL TRAIN only. FullCPU bestlast/frozen upstream/fresh selected feature
exports required, preserve E0 on no gain, no LR/dropout/seed grid or budget reset.

Balanced TRAIN60 optical agreement diagnostic completed (no device access):
L2 coarse16 median Pearson router.93495/expert.31173/global.79238;
L4 router.93962/expert0.45502/global1.82513/expert2.50458/global3.80579.
Pixel Pearson is lower; no calibrated-phase, device failure, causal source or
accuracy ceiling claim follows. Current source strengthens the read-only audit
by comparing every sampled incident canonical PNG pixel and receipt SHA to the
current producer, plus native phase BMP pixels and report SHA, before interpreting
agreement. A new diagnostic output must be used; old reports remain evidence.
This is identity/diagnostic work only, no training/calibration/hardware changes.

L2 adaptive margin completed317/411(77.12895%), selectedstage13, NLL1.0827771.
Its last segment still improved NLL and it stopped ONLY the initial13-segment
budget. Given the user's overnight persistence request, allow at most37 TOTAL
adaptive-margin segments including these13, at most12 per dispatch, no change
of method/radius/seed and FIRST no-progress/target/07:00 still stops. Profile
`lab_caltech20_L2_residual_adaptmargin_extension_profile_20261019.json`. Reconstruct
EVERY original/continued TRAIN batch before carrying RNG; reject restarting a
first-no-progress/deadline/target stop. Fixture test verifies2-level sampler
lineage and refusal of a no-progress chain. This is finite continued training,
not a fresh-seed or candidate grid; never reset the37 total budget.

L4 within-class comparison retained E0 260; last122 rejected. L2 adaptive
positive margin first segment improved312->314 with2 newly correct/0 lost;
its continuous-RNG finite continuation is active (at least317 already fullCPU
verified), not yet a completed final result. Rather than mechanically adding
more L4 terminal-loss variants, a read-only optical agreement diagnostic now
uses3 source TRAIN/class=60 balanced samples, NEVER TEST. For each physical stage,
replay actual measured upstream, compare the original quantized incident-field
propagation to that exact stage's CCD (pixel/coarse16 Pearson, unit-RMS error).
No fitted spatial shift/gain, new inference graph/PT/CCD/SDK/GPU, optical settings
or calibration change. Entry `--optical-diagnostic`; all params frozen, all CCD
array/PT/data SHA recorded. These statistics diagnose mismatch only, not prove
its cause or an accuracy upper bound. No claim linear-voltage LUT is linear-phase.

TRAIN-only ridge-teacher comparison completed, retained E0 312/260 (last
299/206 rejected). The next two bounded comparisons have distinct diagnosed
failure modes, not a seed/coefficient grid. L2's targeted TRAIN neighbors have
40.4% inactive at margin1 (75th percentile1.86695): per-row positive required
margin now uses TRAIN-neighbor75th percentile clipped[1,4], while negative
margin stays1. Profile `lab_caltech20_residual_adaptmargin_profile_20261019.json`.
L4 only10.8% neighbors inactive, so do NOT apply that change blindly; instead
test a normalized within-class TRAIN logit-variance penalty on the failed
neighbor-weighted full-head fit, coefficient1, otherwise same loss/budget.
Profile `lab_caltech20_L4_residual_withinclass_profile_20261019.json`. Within-class
variation is not asserted to be proven optical noise. Both preserve original
5140 head/inference/nonhead, TRAIN-only derivatives, full411 TEST development
selection, strict fullphysical replay, E0 preservation, no SDK/GPU/newCCD/VALID.

Paired Huber retained L2 312 with tied-NLL1.1047536 after first-no-progress
continuation; L4 stayed260 and no continuation was dispatched for its E0 result.
No new accuracy claim. Next is a bounded regularized TRAIN teacher-logit transfer,
because earlier orthogonal feature alignment could not rescale anisotropic
measured/simulated channels. `lab_caltech20_residual_ridgeteacher_profile_20261019.json`
fits an affine map on matched1200 TRAIN only; regularization selected by TRAIN-only
GCV over33 fixed variance-relative values, NOT TEST selection of33 candidates.
The learned readout is folded directly into original5140 head, then optionally
one scalar convex blend is TRAIN-fitted. Original frozen teacher never evaluated
on TEST/VALID. Complete411 TEST remains development selection, no TEST gradient.
All optics/nonhead remain frozen, CPU2/no SDK/GPU/new CCD or inference layer.

The neighbor-weighted coupled head comparison completed120steps, selected E0
312/260; last L2 269 and L4 109 are rejected, not improvements. Diagnosis at
accepted parents: TRAIN L2 943/1200,L4 592/1200; required single-row radius for
remaining TEST-wrong median.10463/.04823, only16/4 cannot flip at.20 even under
an ideal direction. Thus do not blindly increase radius. A bounded paired Huber
comparison instead addresses interacting rows, keeping radius.20 and other18
logits fixed. Profile `lab_caltech20_residual_huberpair_profile_20261019.json`:
TRAIN signed pair margin(delta1), full20-class correct-TRAIN KL2, anchor.1,
exact Hessian/LM, same continuous TRAIN sampler/bounded budget. Derivatives and
curvature are checked against autograd; no new physical/model/TEST-gradient scope.

Actual fixed-head10 fits ended E0 (L2 48epochs,L4 46epochs); no new10-param
weights adopted. Fresh selected feature exports/fullCPU audits passed. Huber
first segments improved L2 304->305,L4 244->255. Same-method continuous-RNG
continuation then selected L2 312/411(75.9124%), L4 260/411(63.2603%), both
stopped at first no-progress. Still not within1pp, all parents preserved.

Next finite comparison addresses coupled boundaries rather than repeating row
fits: original5140 full-head conditioned LBFGS with capped TRAIN sample importance
from the3 same-class TRAIN neighbors of TEST-wrong examples (weight1to3), plus
original TRAIN prior, correct-TRAIN parentKL2 and .001 anchor. Config
`lab_caltech20_residual_neighborhead_profile_20261019.json`, max120steps/180sec.
Exact TRAIN indices/weights are logged; TEST labels/features choose TRAIN examples
but never enter gradients. It is expressly TEST-informed development, not
independent generalization. Full measured graph replay/frozen nonhead audits,
bestlast and E0 preservation required. No new layer, capture, SDK or GPU.

TRAIN-margin diagnosis at accepted304/244 parents: L2 920/1200 correct,
515 below margin1; L4 554/1200 correct,948 below1. Thus unlike the old saturated
router case, a TRAIN-margin objective has active examples here. Next configured
alternative is signed Huber margin(delta1), reducing influence of severely
violated constraints while retaining exact analytic curvature/LM and bounded
original-row updates. Profile `lab_caltech20_residual_huber_profile_20261019.json`.
Keep teacherKL2/anchor.1/radius.20/continuous TRAIN sampling, no parameter grid.
Reference is max other score for positives and true-class score for negatives.
Only TRAIN contributes derivatives; TEST errors select same-class TRAIN neighbors
and checkpoints, explicitly development leakage. The CPU source is not synced
while its earlier terminal10 process runs; first finish/audit that process and
use its recomputed selected feature bank, preserving parent if E0.

Finite pair continuation stopped at first no-progress: L2 selected304/411
(73.9659%), L4 selected244/411(59.3674%), complete physical CPU replay passed.
Both remain below the final ~1pp goal. Next bounded block-coordinate comparison
freezes the accepted5140 head EXACTLY and fits only original final9conv plus
direct effective alpha (10 parameters), max60epochs/600fitseconds, CPU2/batch8.
Profile `lab_caltech20_residual_fixedhead10_profile_20261019.json`: TRAIN-only
original3141 prior weighting, correct-TRAIN parentKL1, anchor.1. This prevents
simultaneous classifier drift observed in the earlier5150 overfit; not a promised
improvement. Optical masks/incident producers/CCDs and inference graph stay fixed.
Full411 TEST informs development selection, never gradients. Export newly
recomputed selected TRAIN/TEST features before any subsequent head fit; alpha/
conv changes invalidate old feature banks. Preserve E0 if no improvement.

First actual pair segment raised L4 232->241/411(58.6375%),13 newly correct/4
newly wrong. L2 stayed303 but NLL fell1.1697800->1.1607912; do not call that
an accuracy increase. FullCPU/nonhead freeze passed. Each may therefore continue
the SAME pair method under the same finite12-new-segment/first-no-progress/
07:00 limit, reconstructing all2304 prior TRAIN batches before advancing its RNG.
Entry `--pair-sequence`, profile
`lab_caltech20_residual_paircontinuation_profile_20261019.json`. No revised
radius/loss/seed search and no reset to a worse parent. This does not change the
live L6 controller, its masks or any collected incident fields.

The continuous-RNG row continuation stopped at its FIRST no-progress segment:
L2 best303/411(73.7226%),four new segments attempted, selectedstage04; L4
still232/411 after one new segment. It did not use the remaining12-segment
allowance for blind repeats. Both strictCPU/frozen audits passed, parents kept.

Next is one bounded antisymmetric pair repair, because L4's first row pass fixed
26 but broke18. Choose up to12 most-confused pairs using TEST diagnosis. Per
pair, update original rows a+=delta,b-=delta with257 degrees of freedom, other
18 scores fixed; TRAIN binary pair CE plus full20-class correct-TRAIN parentKL2,
anchor.1, exact Hessian/LM and radius.20. Hard outside-pair TRAIN negatives and
all-class replay protect collateral behavior.192steps/pair/180fitseconds; no
new layer, SDK, GPU, optical change or TEST gradient. Full411 accuracy/NLL and
new-correct/new-wrong still govern adoption. Profile
`lab_caltech20_residual_pair_profile_20261019.json`. Analytic pair gradient,
full Hessian and damped solve are checked against autograd before dispatch.

Radius0.20 first actual segment improved L2 291->297/411(72.2628%),9 fixed/3
newly wrong, and L4 224->232/411(56.4477%),26 fixed/18 newly wrong. Both fullCPU
and nonhead freeze audits passed. These are additional TEST-informed updates,
not isolated-method causal improvements. Preserve every earlier parent.

Because both made genuine progress, each may continue the SAME accepted method
for at most12 new segments (192steps/class/180fitseconds each), stop on first
no accuracy or tied-NLL improvement, target, or2026-10-19 07:00 Beijing boundary.
No new candidate/seed grid: reconstruct every previous actual TRAIN batch and
continue its RNG stream. Original optical chain stays frozen; noSDK/GPU. Entry
`--row-sequence`, profile `lab_caltech20_residual_rowcontinuation_profile_20261019.json`.
This is a finite sequential fit, not a recurring automation or permission to
resume old capture tasks. L6's independent SDK chain proceeds meanwhile.

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
