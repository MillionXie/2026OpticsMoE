# Six-layer student with a TRAIN-only frozen shallow teacher

Human authorizes evidence-driven residual optimization of the fixed original20
and one-router alternating E-G2/4/6 architecture. Baseline both arms100 is fully
completed and independently audited; controls remain100/frozen/no new forwards.
The20-epoch generalization recipe and5140-linear-only refinement also completed
their own audits/exits. All old sources, failures, PTs, histories and claims stay.

Evidence: two-layer selected validation71.3781 exceeds six63.2509. Six clean
TRAIN99.85 versus validation63.25 shows a generalization gap, not insufficient
training accuracy. Stronger generalization recipe did not improve selected six;
linear-only refinement gained one test correct sample (83.0426->83.0911), still
below accepted four83.4787. Hypothesis is shallow-teacher soft supervision during
training, not another learning-rate-only retry or an extra inference network.

One frozen baseline residual L2E100 teacher. Its parameters stay byte-identical,
requires_gradFalse/eval/no-grad. Same encoded TRAIN affine/flip/MixUp input as
student. Only samples whose teacher argmax matches dominant supervised soft label
contribute. Add .5*T^2 KL(teacher_T || student_T), T2, to the unchanged weighted
supervised/capture/phase objective. No test images for choosing teacher/method.
Teacher chosen by stored VALIDATION advantage, not maximum test depth score.
Every epoch records actual eligible coverage and KL, not guessed usefulness.

Student is the exact original six-layer state/contract: one router then E-Gx3,
four224experts,478global,joint8bit, learned ABO RMS fusion alpha(.7001,.95),
5140terminal linear; total electronics5200. Teacher is never registered as a
student submodule and not deployed, exported or used for student validation/test.

Restore full LAST120 raw/EMA/Adam10320 from genreg. No reset/replay of optical
steps, Adam moments or EMA. Keep accepted head-refined E100/headstep2 as prior
first-minimum validation-NLL candidate; its separate20 head updates are explicit,
not optical epochs or Adam steps. One LR.0002/scheduler-stat restart is recorded,
ReducePlateau factor.5/patience3/delta.002/min2e-5. MixUp.2/LS.05/featuredrop.1/
TV.05/electronicWD.01/flip.5/capture.05/EMA.95; no new inference operation.
Effective16/micro4,86Adam per epoch/tail12. Exactly20 new epochs121–140 for this
method review,1800fitsec safety; not a claim of a user-imposed total cap or
convergence. One non-A100 among4/2/5/0,2GiB+800MiB, no foreign process changes.

CPU admission independently proves KD logits gradient/gating in float64,
10320Adam-moment restoration, teacher digest/freeze and unchanged student graph;
zero real forwards/updates/test. Actual GPU producer separately proves all20,
macro16 and tail12 combined CE+KD gradients/probabilities at micro1vs4 and all
student tensors finite/nonzero, teacher unchanged/no gradient/0optimizer/test.
Own saved audit checks sources/config/parents/complete prefix receipt hashes,
rawEMA/Adam12040/scheduler replay/order-affine-Mix-flip-masks/coverage/selected
PT/IDs-labels-CM-macroNLL/first minimum,0forward. Original rc0/actual exit and
own science SHA/completed auditrc0 are mandatory before selected-model TEST.
Any failure persists and stops; no retry, relaxed gate or altered frozen source.

Selection remains first validation balancedNLL minimum including prior best,
1e-6 numerical ties keep earlier. A changed selected digest is tested once;
unchanged prior genuinely reuses its receipt. Own independent saved test audit.
Family acceptance retains the previously accepted six if this chosen method
loses test accuracy, and reports that negative result; never pick a favorable
intermediate test epoch. Four83.4787 reference is not lowered. Six>four and
preferred >10pp gap versus frozen six-control78.0039 are separate checks, not
promises. Previous four>80 is met. All outcomes preserve macro and20-class rows.

Dataset/roles/classes1372/283/2064 unchanged. Validation-aware classes, observed
tests, extra residual budget, method-family test feedback and training distillation
make this development, not unseen independent generalization or equal-budget
causal residual/depth evidence. No A100, new classes/heads/architecture, device
SDK/acquisition, automations, old helper re-entry or further CCD deletion.
