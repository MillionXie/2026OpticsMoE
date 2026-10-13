# Proposed bounded main-phase capacity comparison - NOT AUTHORIZED/STARTED

Current three-candidate budget finished with no eligible profile. This is a new
scientific scope and must be explicitly authorized before any GPU fitting. No new
queue/config/training run or test was started by this proposal.

Question: can restricting spatial degrees of freedom in main phase masks reduce the
27-29pp generalization gap while preserving validation/class and deployment gates?
Fixed revised20 data/split/classes, corrected RGB/dense4/preserve478 optical graph,
main4/rho.3/router extra, masks/distances/pitches/OEO/8bitjoint encoding all remain.
No new learned electronic component, CNN/readout or added propagation. Router stays
224x224 and uses its deterministic seed17 initialization. Main masks in all three
new arms start at raw0 (same constant initial physical phase phi=pi, M=-.4), so the
initial physical fields must be exactly identical despite different parameter shapes.
This main initialization changes from the historical random full-resolution study;
its result is therefore not the matched control. A new full-resolution arm is required.

| Proposed arm | Main trainable grids | Main parameters | Router parameters | Total |
|---|---|---:|---:|---:|
| New shared-initialization control | eight224x224 and two478x478 |858376|50176|908552|
| Main spatial capacity128 | ten128x128 |163840|50176|214016|
| Main spatial capacity64 | ten64x64 |40960|50176|91136|

Coarse raw masks are bilinearly interpolated to their original expert224/global478
apertures before phi=2pi*sigmoid(raw), complex residual modulation and actual8bit
quantization. Interpolation is phase-mask parameterization, not an image feature
encoder or electronic classifier. All three share the same physical aperture/propagation
and complete initial physical fields; gradients must be checked with respect to the
actual free coarse parameters. Restricting capacity may harm fitting/accuracy.

All three fresh, MixUp.6, LS.08/capture.05/phase smooth.005/EMA.95/lr.002,
macro16 tail14/micro1, affine20deg5pxscale.9-1.1, same seed17/order/augmentation.
Max150epochs or14400fitsec EACH, min50/early25delta.002, ReduceLR6bad*.5floor2e-5.
Only physical2/5/6, max3children, Torch2GiB+800MiB margin, CPU parent CUDA invisible;
no foreign termination/env modification, actual exit/release at completion. Budget is
three new fits, at most12aggregate GPU-hours, no retries/extension/fourth candidate.
No test arrays or inference. Best/last retained; first minimum valEMA balancedNLL.

Before fitting: publish separate source/profile on main, freeze source closure, CPU
initial physical field equality for all arms, actualGPU repeat/micro1 gradient gates,
all phase gradients finite nonzero, original native amplitude1920x1080/phase1920x1200
random/dim/black roundtrip exact/prob<=1e-6/fivecaptures/blackzero. Save failures and
no automatic retry. Old126/130/ABO and all prior runs protected. New coarse masks
need matching parameterization-aware exporter; old full-resolution plane slicing
must not be silently reused. No laboratory connection.

Same admission: selected valaccuracy>=.65/macro>=.60, eachvalrecall>=.20, no selected
train/valzero, each last20classzero<=half, gap<=15pp, completeall4/input-dependent
routing, selected trained nativeBMP. Only after complete3/actualparentchildrenexit
exclusive savedCPUaudit may select among eligible by minimum valbalancedNLL,
alpha no longer varies. If none eligible, report bounded failure and do not auto-search.
A passing L4 only permits separately bounded2/3verification; formal six arms and
once-only test still need all prior eligibility/CPU/BMP/lock conditions.75% remains a
goal, not a guarantee. This draft specifies the scope, not implemented/passed gates.
