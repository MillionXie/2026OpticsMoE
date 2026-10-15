# Classification-preserving electronic fitting, identical8554 head

This is a TRAIN/VAL-evidence-led loss change within the human's continuing
residual-only electronic tuning authorization. No architecture change, no new
optical weight, no new CCD acquisition, no different hardware head. The six-arm
common3-epoch run and all data/claims remain protected. L2 achieved83.96072% vs
79.70540% (+4.25532pp) with residual12/control3 electronic epochs; leave it frozen.
Ordinary CE L4 reached validation plateau at actual66/selected55 but test92.58%;
L3 validationNLL kept improving while required test gap was not restored. That
controller was stopped only at its completed audited idle boundary at stage41,
L3 actual123; separate review_stop receipt retains the true manual stop provenance.
No source/run/status/checkpoint was overwritten or failure receipt synthesized.

Saved TRAIN/VAL diagnostic (no TEST images): L4 original optical TRAIN758/761
correct, ordinary head756:4 correct became wrong and2 wrong became correct;
VAL137/157 became135 with4 lost/2 gained. L3 TRAIN692 became719 but7 originally
correct were lost; VAL128 became131 with0 lost/3 gained. This motivates preserving
already-correct optical TRAIN decisions instead of uniformly sharpening confidence.

Restart each L3/L4 residual head from its fixed common3-epoch trained checkpoint,
NOT a highest-test checkpoint or random head. Same8554 module and frozen simulation
TRAIN mean/std, same optical parents and original verified feature caches. No
control retraining/retesting. On TRAIN only, fixed gate=original optical argmax
equals label. Loss=mean(weight*CE + gate*KL(baseOpticalProbability||newProbability)),
weight=.05 for originally correct and1 for originally wrong. Gate and base are
frozen. No added trainable scalar/branch. New head AdamW reset is explicit because
the objective changes; head weights and buffers are restored, lr.0001/wd.001,
batch64/tail57/12updates per epoch. All new updates are electronic only.

Before NEW tests, fix selection to first highest validation accuracy, ties lowest
balancedNLL, including the already-trained parent E3. This aligns selection with
the human's accuracy target rather than accepting lower accuracy solely for
confidence calibration. No test-max epoch selection or retrospective substitution
into old results. Tests remain development/stop feedback, not unseen generalization.

3-epoch saved segments, target retained original overall accuracy AND original
gap versus the same trained control3. Stop when both met. Review after>=9 new
epochs if no new validation accuracy peak for6 and no.5pp train improvement for6.
No blind sweep, optical tuning, head enlargement or class changes. Old epochs/
extra optimizer budget/reset and new/global epoch counts must all be reported.

New CPU analytical gradient proof checks .05correctCE + unitKL, frozen trained3
parent identity and source. Each actual GPU producer checks new-loss micro1vs8
gradients on all10+6 fixed TRAIN features with0optimizer steps, exact parent state.
No optical model is instantiated. Each saved stage synchronously pauses its head
optimizer while its own exclusive CPU NumPy audit reconstructs every actual step
loss from saved TRAIN logits/order/base/labels, verifies TRAIN buffers, optimizer
state/selection, selected+last TRAIN/VAL probabilities and saved TEST CM/macroNLL.
This stage audit occurs while the producer waits (not a false producer-exit claim);
final worker rc0/actualexit is separately required before final delivery. A failed
stage preserves evidence and stops, no retry/threshold relaxation. Same-head-digest
old test receipts reused, new head digest evaluated once against cached TEST
optical features. Source-only Git, raw features/logits/PT/IDs remain private.
Only non-A100 GPU2/4 for the two remaining residual heads; freeze acceptedL2 and
all control3 heads. Hardware owner receives this same model/weights/buffers after
actual verified target success, with exact manifest+SHA. Scheduled tasks stay paused.
