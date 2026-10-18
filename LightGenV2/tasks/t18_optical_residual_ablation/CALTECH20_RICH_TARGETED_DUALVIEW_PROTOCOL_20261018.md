# Five-card targeted continuation, first 10-epoch scientific review segment

Human continuous foreground authorization, no automation, max five non-A100
GPUs0/1/2/3/4, one child/card, 2GiB+800MiB, release every completed child.
Fixed richest20/3141TRAIN/383VAL/411TEST, nores50 permanently frozen. Keep old
best2/4/6, never overwrite its weights/data/captures/claims or old failure.

Evidence source341 TRAIN100/VAL100 probes completed no optimizer/TEST:
abrupt alpha .94→.45 collapses retained representation (not full accuracy).
Changing ONLY RMS derivatives keeps forward bitwise equal, increases stage
electronic CE gradient norm L4x9.04/L6x4.45 and changes L6direction cosine.344;
optical gradient direction/norm nearly unchanged. This supports testing a
gradient-policy correction, not claiming one exclusive cause of overfitting.
Each stage electronic is 9params residual3x3+identity, not a large classifier.

Five cases: fullRMS2/4/6 on GPU2/3/4; historical detachedRMS4/6 on GPU0/1.
Both latter are STILL optical-rho.3 residual models, NOT the frozen nores
controls. All five share every training setting except RMS differentiation.
Warm-start exact retained selected state70/80/80 and preserve effective alpha
(.934–.946). Do NOT repeat the harmful abrupt alpha restart; .4 is allowed
lower bound, not forced final value. Direct float64 physical coefficients,
float32 fusion, projection to next representable value strictly above.4001.
State keys fusion_logits store PHYSICAL alpha, use this source loader only.
The router/E-G cycles/electronic9-perstage/final5140 clean inference function,
parameter counts/8bit/physical forward unchanged. Original-model clean
prob/capture parity is an additional actual GPU gate,<=3e-6.

Selected checkpoints do not contain matching optimizer moments; explicitly
start NEW AdamW and EMA from the exact trained state, not pretend a LAST/raw
optimizer resume. First2 epochs phase+alpha frozen while existing electronic
convs/head adapt, then8 joint epochs. Ten is a SAVED SCIENTIFIC REVIEW SEGMENT,
NOT the end of human optimization authorization. After full scientific review
continue/refine the evidence-supported strategy rather than stop on no gain.

Two-view common training: clean original RGB MixUp and mildly augmented RGB
MixUp, SAME IDs/permutation/lambda/softlabels; teacher cached from clean TRAIN
applied ONLY to the clean view. Old strong-crop cached-teacher mismatch is
avoided. Crop>=.95, original affine/flip/color, feature drop.05 on augmented
view only, MixUp.2/LS.05. No phase/input dropout or CCD noise in this comparison.
Weighted supervision .5clean+.5aug; inverse-frequency exponent.5 rather than1
for the known class imbalance; clean KD .25(Tstudent4/Tteacher16, original
correct/confidence gate), JS consistency .2, phase L2SP .05*sum(deltaRaw^2)/2,
TV.005/capture.01. LRphase.0001/electronic.001/directalpha.0005 cosine10,
EMA.99/electronicWD.02/clip1;197Adam/E, macro16 tail5 micro2. Two views do NOT
mean two Adam updates. Complete last-boundary Adam/project/EMA saved.

CPU proof independent JS value/gradient, inherited fullRMS finite-difference
proof, every parent's exact state/counts/alpha and all helper closures;0real
forward/optimizer/TEST. Actual producer gate all20/macro16/tail5 micro1vs2,
all12/21/30 gradients finite/nonzero, same contexts, ordinary threshold2% and
cos.999/prob3e-6 unchanged. Source346 frozen before run. Each source/config,
parent/prefix/order/affineMix/RNGcrop/view-loss/steps/rawEMA/Adamprojection/
phasealphaFreeze/saved-ID-label-CM-NLL/TEST selection independently verified,
0CPUforward/TESTimages. Only own rawproducerexit0/actualexit then exclusive
audit, own auditrc0 and completed SHA permit calling a result scientifically
complete. Failure preserved/no retry/threshold relaxation.

TEST every5 only new EMA, best by TEST correct then val balancedNLL, old
incumbent retained as tune0. Two new TEST model observations percase/10total;
no control retest. TEST-informed development, unequal budget/asymmetric alpha/
training-only pretrained teacher, NOT independent generalization or pure-rho
causal evidence. Every summary includes actual+selected epochs/TRAIN/VAL/TEST/
macro/whole20recalls and exact retained model identity. No guarantee +10pp.
