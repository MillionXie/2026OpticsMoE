# Caltech deployment regression diagnosis, 2026-10-14

Human authorizes at most three project GPUs, now including physical6 A100; allowed
physical2/5/6 only (latest human instruction supersedes the initial list), memory2GiB per child plus800MiB free margin. CPU supervisor
has no CUDA visibility; no foreign process termination or environment changes.
Old sealed code/runs/checkpoints/predictions remain untouched. Independent T18
diagnostic subclasses import existing graphs; no source clone/worktree/branch.

## Separate causes before changing the formal task

Use frozen old20 train1372/val283, cachea81e78... and manifest91f7aa..., no test
arrays/inference. Nine fresh main4/rho.3 fits, each common40 epochs,7200sec cap,
seed17/macro16/micro1/rawinit.1/lr.002 fixed/EMA.95, no early stop or LR scheduler.
Common LS.08/affine20deg5px10%/MixUp.4/capture.05/phaseTV.005. First minimum
validation balanced NLL selects EMA, per-class support/recall/CM every validation,
unaugmented train every5/end. All initial phase shapes/states must match for the
eight current-graph variants; old9-expert bundle has its own disclosed state.

Variants: base (frozen current graph exact); unquantized (amplitude+phase float,
NOT deployment eligible); fullfield_ln (pooled224 normalization over both axes
instead of rows); rgb_tiles (old fixed R/G/B/mean mosaic, retain new224/RMS.25);
dense_router (all four experts instead of Top2); preserve_global (expert CCD
478 spatial field retained as next global illumination, no pool224/re-fanout
before global; expert transitions still pool/fanout); mixup02; mixup06;
legacy_reference (old9-expert/RGB/OEO/geometry bundle under the same current
training loss/budget, NOT a one-factor comparison). These isolate suspected
information/optimization loss, not promises of recovering a certain score.
Base frozen forward/state identity and all20-class finite nonzero phase gradients
are preflight gates. Failed output retained, no automatic retry; failure drains
already healthy children and stops new dispatch. Final saved CPU audit reconstructs
all40 order/affine/MixUp hashes incltail12, Adam86steps/epoch, source Git blobs,
best/last and saved train/val predictions, with model_forward_calls0.

## Revised formal categories and class-level acceptance

Human explicitly excludes Faces/Motorbikes/airplanes following observed development
failure. Preserve the other17 classes; select exactly3 substitutes from original
>=60-photo pool by fixed train-centroid/validation thumbnail elimination only.
All existing train/val roles pinned; no prior fit/val IDs enter test. No optical
test scores or test pixels used to rank replacements. Exclusion itself was informed
by old observed development results, so this is a changed development task, NOT
independent held-out generalization. All photos of selected classes retained.
Other classes are not silently deleted repeatedly until accuracy is high.

Prepare and freeze revised manifest before any formal fit; distinguish diagnosis
on old20 from future comparisons on revised20. Report each class, zero recall,
recall<.2, minimum recall and confusion patterns. A persistent train/val class
failure blocks formal acceptance and triggers explanation, not test-driven removal.
Main4 residual needs separate validation-based regularization comparison after
diagnostic evidence; improving train alone cannot qualify it. Max previous150
epoch scale for any subsequent fitting trial; no automatic unbounded search.
No formal training is launched by this diagnostic queue. Review causes, validation
class health and physically consistent BMP encoding first, then record the fixed
formal profile. Final overalltest75 goal remains a goal, never fabricated or
achieved by test retries. Fresh physical calibration still required, laboratory
not connected. Matched rho0/.3 formal pairs use identical settings; asymmetric
residual-only tuning must be separately labeled, not a pure residual causal claim.

## Execution handoff to physical2/5/6

The human explicitly restricts all future project GPU children to physical2/5/6.
The old CPU queue and its three own children have exited after checking coherent
completed last/history/gradient/Adam prefixes; old run/checkpoints remain retained.
Continue raw, EMA, Adam and previous best candidate in a separate run, without
replaying completed epochs or changing model/loss/data/budget. Discard only an
unfinished in-memory batch. Total epoch budget stays40, total recorded fit time
includes the saved prefix. Fixed-LR diagnostics have no scheduler to reset.
New execution sources record their own Git identity; original prefix sources are
checked against their original blobs. Only runner/audit/GPU plan may change for
this handoff, every optical/model/other runtime source must still match. Restored
raw/EMA/Adam must be bit-exact before resuming and restored numerical gate must
pass on its actual GPU. The final CPU audit verifies exact parent history and
orders/transforms prefix, source provenance and all40 epochs again.
Daily09:00 Asia/Shanghai reporting is human-requested, independent of the quiet
15-minute continuation monitor. Report actual progress/failures, not a guarantee
of meeting the75% target by the next morning.
