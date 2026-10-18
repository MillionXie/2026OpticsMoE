# L6 training-only shared readout supervision, no new network/head

Human continuous optimization scope; five new leaves/source391 on frozen386.
Evidence: full-RMS and instance/prototype supervision help L4 but L6 still
does not improve TEST; within-class feature covariance is more concentrated
in L6. All30 gradients are nonzero, so do not claim vanished gradients or a
proven sole cause. Hypothesis is regularizing intermediate representations,
not fixing zero gradients. Require actual completed mild L6 SCI/producer0
before CPU/start. Use its improved model only if strictly better TEST; else
retain earliest old80/E80. No test-driven class/data changes.

During TRAIN only, read amplitudes after global stages1 and3 with the EXACT
same existing5140 linear module and nonaffine16x16 feature extraction. Add
.1 times mean weighted CE of these two predictions on clean MixUp view.
There are no auxiliary parameters, heads, hidden layers, or deployment reads.
Inference runs ONLY original final head and original router+E/Gx3 graph;
actual GPU gate compares clean parent predictions/capture<=3e-6. Same state
key/parameter count30, optical1337740/electronic5200. No architecture swap
at CCD deployment. Keep original prototype/dualview CE/KD/JS/phase anchor,
EMA, alpha(.4001,.95), no alpha reset, NEW Adam/EMA from selectedEMA explicit.
Ten extra joint epochs are a foreground review segment, not authorization end.

Independent CPU proves weighted-CE gradients for one shared W/b with three
uses versus NumPy sum, same5140/30 and no auxiliary state, original JS/MSE/
parents/center checks. Actual gate all20/macro16/tail5 micro1vs2; strict
prob/capture3e-6,gradrel.02/cos.999/all30finite nonzero unchanged. Saved own
science reconstructs allorders/crops, method loss INCLUDING aux,197Adam/E,
last Adam/EMA math, state/PT/source SHA, all savedIDs/CM/NLL and TEST selection.
No extra optical forward in independent saved science. TESTevery5 remains
development feedback, not independent generalization or same-budget rho
causality. Frozen nores50 and all prior outputs/source/failures protected.

Prefer releasedGPU4; global ownmax5 one/card0–4,2GiB+800MiB. No A100, foreign
kill, native/lab, scheduling, or new inference electronics. Source only Git,
exact5paths, private raw arrays/IDs/PT/logs/JSON. Failure retained no retry.
