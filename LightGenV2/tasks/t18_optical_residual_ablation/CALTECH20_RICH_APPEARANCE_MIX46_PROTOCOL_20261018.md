# Four retained-model appearance augmentation comparisons

Human authorization: continue foreground depth4/6 optimization and investigate
published data augmentation methods. L2 is FINAL; controls50, original rich20
TRAIN3141/VALID383/TEST411, optical inference graph and linear5140 head stay fixed.
No lab/device/CCD changes, no scheduling, no fresh weight initialization.

Research basis: AutoAugment (https://arxiv.org/abs/1805.09501) reports transfer to
Caltech101. AugMix (https://arxiv.org/abs/1912.02781) motivates stochastic operation
chains, blending with original content and prediction consistency. This is a
bounded AugMix-inspired adaptation, NOT a reproduction of either paper or a claim
their CNN results apply to optical hardware. No generative augmentation or new data.

Evidence: source474 two fixed TRAIN macro diagnostics found total versus CE phase
gradient cosines0.9288..0.9784; prototype gradients do not dominate CE. Neither
increasing KD, same-class donor patches nor ordinary CE improved the retained
models. Do not invent a unique overfitting cause from these local measurements.

Warm-start depth4 a224cf9742e97351205bda983945d689fa4f6040c56f84f0c51a2c590d03e0e4;
depth6 4f9d5e1b6a9d0a3a7c15484236d53e82ad341d9c347ee09857629e26e2aa92ae.
Four candidates, each5 LOCAL joint epochs, new Adam explicitly recorded because
parents are selected EMA/parameter averages. No phase/alpha reset. Existing matched
teacher temperatures, KD.25, class-weight exponent.5, mild crop/affine/MixUp,
prototype/JS/full RMS/SLM one-code TRAIN jitter and step budget remain unchanged.

Each TRAIN augmented RGB view first receives a mixture of two1–2-operation chains:
gamma.7..1.4, saturation.5..1.5, partial autocontrast.25...75 or blur/sharpness0..1.75;
Dirichlet branch weights, original-image mixture fraction.15...5. No heavy crops,
inversion, solarization or label changes. Matched second family additionally erases
one mean-filled rectangle with probability.5/area.02...1. This is different from
16x16 scattered input dropout. Each image keeps at least50% original contribution
before the optional small rectangle. Clean KD/reference view is byte-identical.
Original RGB cache unmodified; validation/test/inference do not augment.

At most4 simultaneous candidates on approved shared nonA100 GPUs, one/card,
2GiB plus800MiB policy, global≤5. CPU admission independently reconstructs synthetic
pixels via NumPy (≤1/255 boundary rounding), plans and clean/source preservation.
Each producer additionally runs all20/macro16/tail5 micro1vs2 exact gradient gate.
Own exclusive independent source/config/PT/rawEMA/steps/order/LR/targets/planSHA/
saved-prediction audit must pass with actual producer rc0/exit before acceptance.
No retry or failure deletion. All old identities and outputs protected.

Only final local5 endpoint gets a new TEST evaluation, incumbent predictions reused.
TEST-informed development selection is explicitly authorized and must not be called
independent generalization or same-budget pure optical residual causality. Report
all20 classes/macro tradeoffs and retained earliest model on correct-count ties.
No guarantee of10pp. Final candidates that regress never replace retained models.
