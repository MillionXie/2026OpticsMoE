# SAM versus training-only input amplitude dropout (2026-10-15)

Human explicitly requests other methods including dropout/SAM after189/193
plateaus. New197 source closure is193 plus this protocol, driver/profile/auditor.
Four concrete candidates only: L2 SAM GPU0, L2 dropout GPU2, L4 SAM GPU4,
L4 dropout GPU5; one child/card, global4, non-A100,2GiB+800MiB spare. Do not
change environment/DNS or kill foreign processes. No automatic extra candidates.

Restore both alternatives of each depth from exactly the same audited LAST:
L2 new189 E725 raw/EMA/Adam34800/scheduler725; L4 new193 E650 raw/EMA/Adam31200/
scheduler650. Preserve full history, old best model, all Adam moments and EMA.
No completed epoch replay. Once per candidate restart LR/statistics at726(.0003)
or651(.0001), retain global scheduler epoch and optimizer moments. Later segments
do not restart. Existing MixUp/LS/TV/EMA/capture stay at the parent recipe; the
new method alternatives do not combine SAM and dropout, so effects can be compared.

SAM radius.05 is an L2 norm on the whole trainable phase vector, distinct from
optical residual rho.3. Each macro16/tail9 update computes the first accumulated
gradient, perturbs along normalized gradient, recomputes on exactly the same
inputs/targets/augmentation, restores every parameter exactly from a saved copy,
then clips the second gradient and performs ONE Adam update and ONE EMA update.
Both passes accumulate micro8 with full macro weighting; never one SAM update
per microbatch. Record both passes, actual perturb norm, exact restoration and
per-epoch48Adam updates. SAM doubles gradient passes, not Adam steps or epochs.
Reference: <https://arxiv.org/abs/2010.01412> and author's
<https://github.com/google-research/sam>. Improvement here is not guaranteed.

Dropout is specifically input optical-amplitude spatial dropout: independent
16x16-block Bernoulli masks per sample/macro, p.02(L2) or.05(L4), deterministic
seed17+epoch*1000003+batch*97, no inverted1/(1-p) rescaling. It only reduces
bounded training amplitudes. This is an input augmentation, not dropout of
electronic classifier activations or a changed deployed optical graph. Validation,
test and export have no dropout; SAM applies only to training updates. Record
mask SHA/support/seed; independent audit regenerates each mask without forwarding.

Fixed10 classes, original cache/role IDs/geometry/RGB100/4experts/optical router/
8bit unchanged, no trainable electronic classifier. L2 E-G479364/6 tensors and
L4 E-G-E-G908552/11; L3 accepted model and all no-residual100 remain frozen and
are not evaluated again. No class/filter/split changes or weakened controls.

CPU proof checks actual parent identities+compile and a float64 quadratic SAM
analytic gradient/exact restore/single Adam update, deterministic masks, no real
models/forward/test. Each of four restored GPU numerical proofs checks ordinary
optical micro1-vs8 AND the new method's all10/macro16/tail9 gradients, losses,
state preservation; each real producer independently repeats its own gate.
Proofs/claims are exclusive once; failed run preserves evidence, stops downstream,
no automatic retry/relaxed thresholds. Never edit old197/193/189 sources in place.

Complete25-new-epoch segments,1800fitsec per-segment faultguard, no totalepoch cap.
Each segment waits actual producer exit/rc0, then own exclusive independent saved
science(source/config/PT/rawEMA/shape/init/gradient counts/Adam48E/order/affineMix/
all historical LR restarts/prefix equality/per-macro SAM or mask evidence/saved
ID-label-argmax-CM-macroNLL/wholeprefix firstminimum valEMA balancedNLL).
Only own scienceSHA/completed passed admits selected model testing.

Select only wholeprefix firstminimum validationEMA balancedNLL, not max accuracy
or test. Inherit19 old L2 and12 old L4 distinct model digests. New selected digest
at most one test, unchanged model reuses exact old result; shared atomic digest
claims deduplicate any equal new models across simultaneous alternatives.
Test feedback controls stopping: L2 actual test>=.82, L4 actual overall difference
against frozen100>=.10, no favorable rounding. Otherwise pause at minimum LR2e-5
after75 new epochs without .0002 significant validationNLL improvement. Plateau
is not the selected epoch or a preassigned global epoch limit.

Always show actual/extra epochs, selectedE, train/val/test/macro/gap, all classes,
actual new/inherited distinct tests and negative results. This is fixed screened10
single-seed17 validation/test-informed development with additional residual budget,
not unseen independent generalization or same-budget causal residual evidence.
Test1833 is heavily imbalanced, supports315/80/678/680/20/16/13/9/14/8; show macro.
Prior exact hash/ID intersections0 do not prove no near-duplicates/subject leakage.
Only aggregate MD/CSV/figures Git; rawJSON/PT/IDs/full logs private. Existing
GitHub destination authorized; preserve all foreign T06 changes/commits. No old
helpers/cpu/gates/tests/audits/figures/PDFs/native/formal/lab re-entry. t18 and09
remain ACTIVE, normal progress quiet, new failures and actual final state once.
