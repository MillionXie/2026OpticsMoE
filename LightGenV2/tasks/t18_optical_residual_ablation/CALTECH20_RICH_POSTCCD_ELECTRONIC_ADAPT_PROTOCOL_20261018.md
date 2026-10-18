# Freeze upstream optics; train only EXISTING final electronics under CCD noise

Source407=402+five leaves. Human continuous foreground optimization, no
scheduler. Completed402 one-scalar calibration improves VALID but reduces
TEST4/6 to80.29197/79.80535; not adopted. Keep canonical accepted2/4/6
79.80535/81.99513/80.77859. Do not hard-reset alpha. Stagewise398 reveals
learned9conv filters matter and full-stage alpha lowering is harmful.

Hypothesis: stop further optical/upstream memorization and teach existing
last electronic block to compensate noisy CCD. Preserve all initial trained
physical alphas. Freeze EVERY optical parameter, all earlier electronic
kernels and earlier fusion alphas. Train ONLY last9conv + last alpha1 + SAME
5140 linear head =5150 existing parameters/FOUR tensors per depth. No added
head/hidden layer/deployment parameters, same single router E/G x1/x2/x3.
Last alpha learns directly in float64, projected with original contained
(.4001,.95) effective bounds. All optical CCD captures/upstream inputs stay
unchanged because trainable operations are after final CCD. Old assets stay.

Parents fixed to configuration SHA:2source350selectedE75/prioractual80,
4source373selectedE100/prioractual100,6old9b1selectedE80/prioractual80. Legacy
six-layer sigmoid alpha maps to physically equivalent direct coordinates;
actual gate compares all20 initial clean predictions/capture<=3e-6.
NEW Adam/EMA on four tensors explicitly, no optical updates, no LAST resume.
Five electronic-only epochs/depth,197Adam/E,tail5,effective16/micro8; separately
report frozen optical selected epoch, priorjoint budget and actual/selected
electronic epoch. Do not label these5 as joint optical epochs. AdamW rates
conv.003/head.001/alpha.0003 cosine, WD.01/.05/0,clip1,EMA.95.

Clean MixUp view and mild crop.95/color/affine/MixUp.2/LS.05/featureDrop.05
view share exact IDs/permutation/lambda. ONLY augmented last CCD adds
32x32 structured Gaussian gain.10 and read.01*mean intensity, bilinear toCCD
size, clamp>=0. Clean teacher view/eval/TEST/export noise off. Noise seeded
per macro; micro1vs8 uses same masks/noise. Loss .5CEclean+.5CEaug+.25 self-KL
(T2 cached parent TRAIN only)+.2JS. Inversecount^.5 weights naturalmean1.
No phase/capture loss since those quantities are frozen w.r.t trainables.
No new teacher forward or training with TEST images/labels.

CPU proves CE/selfKL/JS derivatives, exact5150/four tensors, phase/upstream
freeze, physical initial state and deterministic CCD noise. Actual producer
gate all20/macro16/tail5 micro1vs8: p/capture3e-6,gradrel.02/cos.999/allFOUR
finite nonzero; every frozen gradNone. This is correct frozen-model contract,
not pretending all12/21/30 are trainable or relaxing old numeric thresholds.

Save EMA0..5 and genuine fullTRAIN/VAL per epoch, rawLAST/optimizer boundary,
all orders/affine/crops/noise SHA, plus per-macro pc/pa/y/teacher arrays.
After all raw producers0/actualexit, own independent FIT science checks
all frozen tensors, exact985Adam/last update+projection/EMA,6 checkpoints,
independent targets3e-6/loss reconstruction2e-5, all savedID-CM-NLL.
Only then developer TEST selection across these five new head checkpoints,
as human explicitly authorized. Parameter-equivalent epoch0 MUST reuse
parent TEST, including normalized legacy6; never a second parent forward.
Each new digest at mostone TEST, max5/depth, select correct count then lower
VAL balancedNLL then earliestheadEpoch. Own independent saved TEST science
checks every prediction/ID/CM/NLL, reuse and exact selected PT/epoch. Report
same selected model's TRAIN/VAL/TEST; no favorable cross-epoch composition.

This is TEST-informed development, not independent unseen generalization or
same-budget pure-rho causality. Frozen nores actual50 never touched/retested.
Global adoption requires genuine completed science and improved global
score; protect earlier best at tied accuracy and preserve every failed or
negative candidate. Five epochs are a review checkpoint, not automatic end
of ongoing human optimization. No guarantee of reaching10pp.

At most3 children on free0–4, prefer2/3/4, globalmax5 one/card2GiB+800MiB,
1800fitsec/arm fault protection. Actual exits release memory. No A100,
foreignkill/native/lab/extra classes/roles/depth/inference network/scheduler.
Source only Git exactfivepaths; arrays/IDs/PT/logs/rawJSON private.
