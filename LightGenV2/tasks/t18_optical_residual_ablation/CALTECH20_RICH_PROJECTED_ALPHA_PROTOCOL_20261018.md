# Direct bounded alpha and fusion curriculum, same inference function

Actual fresh-sigmoid curves: L2E20 train66.19/val64.23/test65.45, expertalpha
.4124, globalalpha.5641; L4E13 train64.69/val62.92 and expertalphas.409/.414;
L6E5 train55.40/val54.57. These are underfitting early observations, not final
results. The sigmoid-coordinate derivative near .41 is again small, analogous
to the old upper-bound saturation. Simply lowering LR or adding stronger
dropout is not a targeted response to this new condition.

Human authorizes repeated evidence-driven optimization until Beijing09, alpha
lower.4/start.45. New three freshseed17 residual depth2/4/6 full50 begin.45.
Temporarily freeze alpha (phase/electronics still trained) on the deterministic
10-epoch curriculum .45→.65 inclusive, allowing optical features to learn before
the bypass suppresses them. Then learn effective coefficients directly with
AdamW, projecting each scalar into[.4001,.95] after EACH step. No sigmoid in
the actual alpha forward path; no gradient attenuation at a bound. Same count,
same normalized fusion graph and physical8bit optical propagation. The legacy
state key fusion_logits now stores EFFECTIVE alpha: load ONLY ProjectedMoE,
never old sigmoid-coordinate class. State contracts/manifests distinguish it.

To address current underfitting rather than indiscriminately suppress TRAIN,
relax TRAIN crop minimum area.65→.8, feature dropout.2→.1, MixUp.3→.2,
LS.08→.05, electronicWD.05→.02, parameter/inputdrop.01→.005, CCD gain/read
.01/.002→.005/.001, TV.02→.01/capture.05→.02. Increase clean cached TRAIN KD
.05→.1, EMA.99. No augmentation/noise/dropout at inference; teacher never
deployed. PhaseLR.002/electronic.001, physical-alphaLR.0005 groupcosine floors
.00002/.00002/.0001. New alpha coordinate LR is not comparable to sigmoid-logit
LR. Freshphase/electronic updates9850, alpha updates7880 (frozen10E), one macro
Adam/EMA each16/tail5; projection precedes EMA. Scalar alpha EMA remains bounded
by convexity, frozen curriculum copies exact effective values into EMA.

CPU independent proof: exact parent effective alpha conversion, unchanged counts/
seed/nonalpha state, complete TRAIN AND audit namespaces, actual loader/crop/
teacher/synthetic groups plus independent projected-Adam algebra and inward
recovery from both bounds. RealCPU opticalforward/optimizer/TEST0. Each actual
GPU own all20/macro16/tail5 micro1vs4 ordinary/new-method gradients at original
thresholds. New own saved audit adds independent projection of reconstructed
final raw Adam state, exact alpha stepcount/curriculum and trace flags, retaining
source/data/PT/EMA/ID/CM/NLL/TEST-selection checks. Its entire math namespace is
preserved; failure/partial proof remains, no retry or fake completion.

Queue uses only freeGPU2/0/1 and waits for existing fresh-sigmoid jobs; no
duplicate slot or foreign action, ownmax3/Torch2GiB+800MiB/A1000, case7200sec,
queue18000sec protection. Fixed current richest20 TRAIN3141/VAL383/TEST411 and
nores50 PT/results unchanged; old accepted models always retained if superior.
Savebest/last, TESTevery5 (=30 new model observations), no E0 TEST repeat or TEST
gradient. TEST-informed development, not independent generalization, equal
budget or pure-rho causality. Report all attempts, actual/selectedE/train/val/
test/macro/alphas/gaps. Targets not guaranteed. No new architecture/electronic
head, labels/splits, dataset, A100, lab or scheduled task. Source only Git,
data/PT/rawJSON/IDs private. This new strategy is independently bounded, not an
extension or rewrite of completed/failed prior workers.
