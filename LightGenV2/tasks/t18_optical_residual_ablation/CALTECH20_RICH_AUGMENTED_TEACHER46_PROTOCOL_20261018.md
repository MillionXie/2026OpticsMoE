# Same-view online frozen teacher, retained four/six layers

Latest human: only four/six, L2 finalized; active evidence-driven optimization,
up to five shared non-A100 GPUs, no timers. Prior eight5-epoch groups did not beat
retained TEST337/333. Increasing augmentation/terminal dropout reduced TRAIN
without heldout gain. Cached TRAIN teacher coverage on difficult classes is100%;
per-class logit scale ratios.80–1.32 do not suggest a gross missing-temperature issue.

Current augmented student KD targets were generated from the original clean image.
Hypothesis: teacher soft uncertainty on the actual augmented input can give better
guidance than the unchanged clean target. Not a claimed unique cause or guarantee.
Two five-local-epoch candidates from the same retained canonical4a224cf97/6 4f9d5e1b,
newAdam explicit/no fresh weights/alpha reset, micro4,197updates/tail5, each epoch TEST.
Use source495's same zoom-out/geometry/photo/MixUp and CE50/50, clean prototype,
same total KD.25: half cached clean KD plus half actual augmented teacher KD.

TRAIN-only frozen official ImageNet ResNet18 plus its already audited512→20
teacher head; no downloads, no teacher optimization, no addition to student/PT or
deployment. Tap the actual augmented RGB before the student's encoder, apply the
identical bicubicAA100/clamp/affine100 RGB operations and then teacher's standard
preprocess. Same IDs, labels, MixUp coefficient/permutation. Augmented teacher
correct-label/confidence>=.5 gating is recomputed on that view. Soft target uses
existing TRAIN-calibrated depth-specific teacher temperature. No TEST gradient.

CPU independently proves separate-view KD gradients, exact teacher RGB/student
encoding reconstruction and official weight/head identity, no real teacher forward.
Own actual micro1vs4/all20/tail5 student gate; teacher.eval/no-grad, frozen digest
before/after. Preserve per-macro augmented teacher512 features, q, mass and IDs/row
offsets. Saved science independently recomputes head logits, gating, temperature,
MixUp, hashes and all inherited student/source/Adam/EMA/selection/prediction checks.
No second optical/teacher forward in the auditor; saved linear verification is not
an independent rerun of ResNet feature extraction. Preserve that evidence limit.

Same physical once-router EG×2/3,224experts/478global/RGB100/8bit/alpha and deployed
5140 linear head; no new inference parameters. Each actual epoch TEST once, cached
canonical TEST reused, max integer correct only/earliest tie; VALID reports only.
TEST-informed development, additional teacher/residual budget, no independent or
equal-budget causality claim. Controls and all L2/lab artifacts untouched.

Queue only these two on available0/1/2/3/4, one owned child/card, total<=5,
2GiB+800MiB. A100/foreign tasks untouched.1800fit-second fault guard. Preserve all
claims/failures, no automatic retry or indefinite extension of an ineffective trial.
