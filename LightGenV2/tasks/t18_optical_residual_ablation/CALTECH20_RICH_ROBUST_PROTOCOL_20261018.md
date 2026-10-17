# Caltech richest20: training robustness, explicit TEST-informed selection

Human 2026-10-18 authorizes residual-only CCD noise, input/parameter dropout,
augmentation/EMA/SAM experiments on idle non-A100 GPUs, learned alpha down to0.6,
and TEST-based PT selection. Nores50 and all retained/failed historical artifacts
are immutable. This is TEST-informed model development; repeated TEST selection
is **not** an independent generalization estimate, even though no TEST gradient
or image augmentation is used for optimization. Earlier validation-selection
contracts remain intact for their historical runs.

Fixed quoted richest20 cache SHA38b719b8ad935a0170d059d0e4fffa22f6f22640f979603f7f2543fc2d3f2445,
manifest d4921172a9cc8aa427b89e03772733837be64a3fdaaf826d91050e44df2c459c,
TRAIN3141/VAL383/TEST411 and exact IDs/classes unchanged. Nores actual50 frozen,
alpha lower .8501. Residual initialization is the retained selected joint70
model (L2 same-head anchor1, L4 same-head anchor10, L6 calibratedKD70). Two
head-optimized parents have additional200 LBFGS iterations; disclose these.

Inference remains one router then E/G×1/2/3, four224 experts/global478/RGB100/
joint8bit; stage biasfree9parameter electronic residual+learned alpha and
nonaffine pool/LN/Linear256→20. Phase479364/908552/1337740, electronics
5160/5180/5200; trainable tensors12/21/30. Alpha raw coordinates are remapped
from [.7001,.95] to [.6001,.95], preserving effective initial alpha and clean
probabilities. Fresh AdamW is explicit, not a false full-state restoration.

Four fixed first-review cases: GPU0 L6robust,1 L4robust,2 L2robust,5 L4robustSAM;
one child/card, max4, Torch2GiB+800MiB, shared foreign jobs untouched, A100 barred.
All receive CCD gain/read correlated Gaussian regularization, anchored phase
update DropConnect, input block dropout, RGB color/contrast before tiling, existing
affine/flip/MixUp and feature dropout. These are uncalibrated TRAIN perturbations,
not measured camera noise. Deterministic macro seeds/masks/noise reused across
micro1/4 and SAM passes; no random transformation at evaluation/export/hardware.
SAM separate L4 comparator adds full-macro phase-vector radius.5, exact clone
restore then ONE clip/Adam/EMA. Frozen public teacher cached TRAIN T16/studentT4
KD .1 aids retention; no teacher at inference. No new model parameters/buffers
persist. Nores is neither trained nor re-inferred.

Ten new joint epochs are the first strategy review, not a promise of success or
an automatic authorization to rerun completed stages. Train/VAL clean metrics
each epoch, clean EMA TEST every2 epochs; retain old candidate at tune0 using
saved predictions, never repeat its TEST forward. Select maximum TEST correct
count; ties lower VAL balancedNLL; exact ties earliest. Save best/last PT only,
all five tested prediction arrays with IDs and per-epoch TRAIN/VAL aggregates.
Every new TEST digest is evaluated once in this run; record five evaluations
per case and prior TEST-informed development. No claim of whole-development
once-only testing, equal budgets, pure optical-rho causality or cross-seed proof.

Before fit: independent CPU actual-loader/augmentation/teacher path, alpha/shape/
resource/analytic SAM-Adam proof. Each actual producer runs clean old/new parity,
all20/macro16/tail5 micro1-vs4 finite nonzero gradients and SAM second gradients,
zero optimizer/TEST before starting. Own failures/claims/partial outputs remain
and prohibit automatic retry. After raw producer exit rc0, one CPU independent
saved audit checks frozen sources/config/parents/control hashes, schedules,
IDs/labels/CM/support/NLL, same linear-head probability and pooling arithmetic,
TEST-based selection, Adam steps/final boundary/EMA and method seeds. Zero
optical audit replay. Own audit SHA/completed receipt and actual exit required
before accepting new results. New source only Git; machine JSON/PT/IDs private.

Compare actual and selected epochs separately, always show TRAIN/VAL/TEST and
gap. Preserve incumbents if candidates underperform. Goals L4TEST>.8, L6>L4,
and ≈10pp residual gaps are aspirations, never fabricated guarantees. Review
TRAIN-vs-VAL gap plus TEST/class behavior; lower TRAIN alone is not improvement.
