# Continue the SAME six-layer KD, without resetting optimizer or schedule

Human continues to authorize targeted residual optimization on original20.
The preceding20KD epochs121–140 actually exited rc0 and passed their own
science2785817be614d7ca56f13f1dcee1c414fbe77bef8f4fa18c4f917cf738ebd32b
and TEST8e5d530d60329e35345ab34eabb58b1ef610c9aed9018b124b528385d0f0588f.
ChosenE140 train99.7085/val65.0177/test83.2849/macro62.0726 improves the prior
head-refined83.0911/macro59.9487. Validation balancedNLL fell1.29105->1.24843570
and is still improving; six trails accepted four83.4787 by four correct tests.
This supports a further20 identical KD epochs141–160, rather than changing
strategy or claiming a plateau. No intermediate test/maximum-test epoch choice.

New immutable continuation source/config/claims/output preserve the previous
completed run. Import its exact KD/teacher/micro-gradient/gate helpers; no change
to temperature2/weight.5/correct-dominant-label gating, frozenL2E100 teacher,
MixUp.2/featuredrop.1/LS.05/TV.05/electronicWD.01/flip.5/capture.05/EMA.95.
The generic restored-state GPU gate uses the original fixed epoch121 all20/
macro16/tail12 synthetic-batch plan, but THIS restoredE140 raw state; it is
not represented as the first epoch141 training order. Each actual new training
order/affine/Mix/flip/mask141–160 is independently reconstructed in saved audit.

Restore LAST140 raw/EMA/Adam12040 and ReducePlateau full state including current
LR/best/bad/last_epoch20. NO LR, scheduler-stat, Adam-moment or EMA reset. Final
160epochs/13760Adam. Prior first-minimumE140 retained unless validationEMA
balancedNLL improves by>1e-6. Next20 is a method-review interval, not a fixed
user total training cap or evidence of convergence.1800fitsec safety, no retry.
Frozen no-residual100/accepted2/4 models remain untouched and un-forwarded.

CPU admission independently proves current parent moment restoration and
analytic KD/gates/unchanged student inference contract (0realforward/update/test).
Actual GPU producer again independently checks current all30student gradients
micro1vs4, probabilities/capture, teacher unchanged/no grad,0optimizer/test.
Own exclusive saved science checks prior receipts/parent/source/historySHA,
rawEMA/step13760/scheduler replay from prior current state/141–160 orders/method
coverage and all saved labels/predictions/first minimum,0forward/test images.
Actual producerrc0/exit+own science/completedSHA/auditrc0 before selected TEST.
All already-known three six-layer parameter digests (baseline, head, firstKD)
truly reuse their saved tests; new selected digest gets exactly one TEST.
Own saved TEST audit, no teacher in student inference. Any failure preserved,
stop without retry/threshold relaxation/overwriting frozen sources.

Four reference83.4787 never lowered. Six>four and preferred>10point gap versus
frozen six-control78.0039 are separate unmet targets, not promises. If chosen
new method test is lower than prior83.2849, retain prior and disclose the
negative family result; never choose a favorable intermediate test epoch.
TEST-informed development/family acceptance, extra residual budget/KD and
unequal test supports/macro limitations continue to be disclosed. No unseen
independence/same-budget causal claim. Every summary has actual/selected epochs,
head updates,train/val/test/macro/gap. Only one non-A1004/2/5/0 child with2GiB+
800MiB, no foreign process changes, inference graph/head/classes/data changes,
electronic expansion, device/SDK/acquisition, automation or CCD deletion.
