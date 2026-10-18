# Sparse TRAIN input-amplitude dropout3%, retained four/six layers

Latest explicit scope is only four/six; never touch finalizedL2 or its lab release.
Current endpoint evidence: more supervised augmentation lowers TRAIN fitting but
does not improve TEST. Terminal feature dropout is being compared separately.
This companion tests a different location of regularization: the existing TRAIN
input-amplitude16x16 Bernoulli mask, probability.03 instead of0. Keep terminal
feature dropout at source501's.05, not source505's.2; no stacking the two changes.
Sparse input loss constrains optical feature formation, whereas final-feature
dropout constrains readout dependence. This is a testable hypothesis, not a promised
accuracy gain. Input dropout already exists in the trainer; no new model class.

One candidate per depth from same canonical4a224cf97/6 4f9d5e1b. Same data roles,
CE20/80clean/augmented weights, clean teacher/prototype, zoom-out/photometric/MixUp,
other regularizers, rates/EMA/physicalalpha/SLM quantization/head. Micro4, five local
epochs,197Adam updates/tail5; optimizer reset explicit, no fresh weights/alpha reset.
The mask is applied only to the augmented TRAIN input; clean evaluation/export
disables it. Do not alter physical inference or CCD/lab artifacts.

The existing input-mask tensor is tapped before application (same source expression,
no arithmetic change); actual input/feature binary-mask SHA is saved each macro.
The auditor independently replays the NumPy RNG to check both actual mask traces.
Own independent CPU source/teacher/pixel/view-weight/micro4 checks, own actual
all20/tail5 micro1vs4 gradient and clean-export gate, own full saved independent
science including seeded augmentation/masks/Adam/EMA/predictions/parent identity.
Each epoch's actual weights TEST once, direct integer-correct selection, earliest
ties keep canonical; VALID reports only. TEST-informed development and additional
residual budget, not untouched generalization or equal-budget residual effect.

Queue these two companions once on available0/1/2/3/4, one owned child/card,
global<=5 and unchanged2GiB+800MiB. Current jobs keep running. A100/foreign tasks,
frozen controls and L2 untouched.1800fit seconds protection; preserve failures and
claims, no automatic retry or unlimited extension of a negative configuration.
