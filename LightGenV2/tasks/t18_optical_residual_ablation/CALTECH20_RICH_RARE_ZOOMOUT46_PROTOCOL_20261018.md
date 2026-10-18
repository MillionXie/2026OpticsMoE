# TRAIN-count-aware scale/position augmentation, retained depths4/6

Human ongoing foreground augmentation optimization scope, no schedules. L2 FINAL,
controls50, original rich20 classes/roles3141/383/411 and deployable graph unchanged.
Only two new warm-started local5 candidates, same a224/4f9d parents as source459.
No fresh weights or alpha reset. Independent CPU, actual own numerical gates,
exclusive saved science and producer rc0/exit required. No failure retry.

Read-only saved TRAIN/VALID review found severe generalization gaps in small
classes (e.g. depth4 ewer65/68 TRAIN vs1/8 VALID), while large classes have small
gaps. These few VALID examples are uncertain, not unique-cause proof. This strategy
uses ONLY TRAIN class supports to choose augmentation probability, not per-class
VALID/TEST score or image. Dataset-specific position/background bias is discussed
by Pinto et al. https://pmc.ncbi.nlm.nih.gov/articles/PMC2211529/ ; AutoAugment
https://arxiv.org/abs/1805.09501 motivates scale/position policy transfer but does
not prove effectiveness for this optical model.

Replace neither task nor model: add resize-with-padding to augmented RGB only,
before the existing mild crop/affine/photometric/MixUp. Probability.75 for TRAIN
support≤102, otherwise.25; isotropic side.72...92×150; translation within central
half of available padding. Bilinear align_corners=False, fill from original image
border RGB means, round/clamp uint8. The resize itself avoids discarding the
original object's frame; inherited later transforms still apply. This differs
from larger random crops and from source479 appearance/erasing. No other new
augmentation combined. Clean KD view and original image cache remain exact.

Original matched KD.25/teacher temperatures, prototype calibration/class weights,
fullRMS/phasecode jitter/rates/Adam/EMA/local5/TEST interval unchanged. New Adam is
explicit because starts are selected EMA/parameter averages. Not same-budget pure
optical residual causality. TEST-informed development, not independent generalization.
One new final local5 endpoint TEST per depth, cached incumbent reused. Preserve
earliest retained candidate on correct-count ties; report all20/macro tradeoffs.

Independent CPU NumPy bilinear interpolation and border-fill proof≤1 uint8 unit,
unchanged clean/source pixels; auditor independently reconstructs every row's
TRAIN supports/RNG/side/offset/hash plus all original source/PT/steps/LR/targets/
predictions/metrics. Up to two new jobs, resource-aware queue, one/card, global≤5
on approved0/1/2/3/4 nonA100,2GiB+800MiB. No lab interference. No10pp guarantee.
