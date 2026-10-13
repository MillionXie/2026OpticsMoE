# Caltech20 deployable main-depth2/3/4 (2026-10-13)

Latest human request replaces the old 2/4/12/16 common200 queue. Main depth excludes
one router: 2=expert/global, 3=expert/global/expert, 4=expert/global/expert/global.
Actual sequential captures are therefore3/4/5. All four expert patches in one stage
share one physical phase plane, not four sequential layers. Two rho arms0/.3 per
main depth, six fresh fits, at most three project GPU children. Physical0/1/2/4
only; A100/GPU6 prohibited. No other project is terminated or displaced.

Human explicitly selected final test overall accuracy>=.75 for main4/rho.3 as the
acceptance goal. This is a goal, not a promised or fabricated outcome. Model choice
uses first exact minimum validation EMA balanced NLL, never test scores. Existing
20 screened categories, train1372/val283/test2064, all IDs/support and cached photos
remain frozen. The test was observed in previous development; new results remain
exploratory development, not independent generalization. Report macro recall,
per-class support/CM/NLL, overall accuracy, train/val gaps and late trends together.

## ABO reference and geometry

Reference is the actual sealed T07 rank72, simulation R@1 .83375 / physical .81125;
these are retrieval metrics, not this classification experiment's expected score.
The old .825 report is a different version. Read T07 README and hardware/README.
Read-only import of original T07 Propagation and bounded_amplitude; sealed model,
weights, software, CCD and laboratory computer are unchanged and not connected.

532nm, z=.1m, nominal logical pitch17um, ASM numerical canvas518, physical active
478, four224 patches in2x2, gap30, active origins(0,0),(0,254),(254,0),(254,254).
This replaces the old T18 nine-expert layout and is an architecture/encoding
change; do not call any improvement a controlled regularization-only effect.
Native mapping follows ABO:478 ->1016 nearest sampling, centered amplitude1920x1080
and phase1920x1200. Non-integer logical/native pitch mapping is the retained ABO
approximation; real sampling, ROI, alignment, phase LUT and intensity gain still
require a fresh bench session. Old camera corner coordinates/CCD receipts cannot
be silently reused for this new task.

Raw RGB -> fixed luminance ->224 bilinear ->sqrt amplitude ->RMS .25.
Before EACH router/expert/global phase, tanh(abs/amplitude_scale.5), preserve zeros.
No extra per-BMP peak normalization. Only active illuminated support contributes
useful brightness statistics; black padding is physically required, not a reason
to stretch the whole BMP. Store active/support mean,p99,max/zero/saturation stats.

The coherent phase leakage is ((1-rho)*exp(i*phi)+rho)*U in expert/global only.
It changes amplitude AND phase. Export amplitude=tanh(abs(U)/.5)*abs(modulation)
and phase=angle(modulation); both have to be loaded jointly. A phase-only BMP cannot
implement this rho. If bench uses a physical coherent bypass instead, its amplitude
ratio and relative phase must be independently calibrated; it is not interchangeable
with an unmeasured accidental DC component. Router always rho0.

Amplitude round255 / phase floor256 modulo2pi are INSIDE training, validation and
export using straight-through gradients. Ideal simulation consumes exactly the
quantized fields that BMP encodes. CPU tests roundtrip actual saved native BMPs and
check six model replay probabilities <=1e-6, stage count, zero preservation, no
CNN/Linear classifier, bounded encoding, support and joint residual decomposition.
No optical device is opened by these tests.

CCD stored as raw linear nonnegative canonical478 field. Computational readout:
relative to own field mean, clamp12, log1p, pool224x224, row LayerNorm eps1e-5 without
learned affine, ReLU, RMS .25 and the same bounded encoding for the next stage.
This is electronic OEO between captures, not an all-optical material nonlinearity.
Final classification uses20 fixed equal64x64 disjoint intensity windows; no learned
electronic classifier, frozen backbone, external teacher or pretraining is added.

## Fitting and overfitting control

Fresh seed17, identical initial phases and order/affine/MixUp schedule across each
rho pair. Macro16 with tail12, production micro1 after micro4 failed the probability gate (1.82167e-5; threshold1e-6). No threshold relaxation. The serial-micro1 numerical gate
passes prob<=1e-6, gradrelative<=.02, cosine>=.999 and every phase gradient finite
nonzero. If a mode fails retain failure, no threshold relaxation or automatic retry.
Train-only MixUp alpha.4 (old.2), LS.08, affine20deg/5pixels/10%, phase smoothness.005,
capture penalty.05. Phase terms changed to avoid assuming old9-expert penalties
carry to the new optics; retain this explicit scientific change. AdamW lr.002,
wd0, EMA.95, clip1, validation each epoch, unaugmented train each5/endpoint.
Max150 epochs /14400sec per fresh arm. Validation NLL scheduler halves after6 bad
epochs, floor2e-5. From epoch50, stop after25 epochs without NLL improvement>.002;
keep the actual first strict lowest-NLL EMA. Record each actual stopped budget;
different early-stop lengths do not establish a pure causal depth/rho result.
No test-directed retries/regularization searches, no new categories, no extension
past this bounded plan. A smaller gap is not guaranteed merely by regularization.

## Execution and acceptance

Worker caltech_deploy234.py; three-card CPU queue caltech_deploy234_queue.py,
CUDA invisible supervisor, each GPU Torch2GiB plus800MiB free margin. Existing
spare-memory permission permits sharing with foreign jobs; never terminate them.
Check card index, name and UUID at each dispatch and real /proc CUDA visibility.
Prioritize4rho.3/3rho.3/2rho.3, then the corresponding rho0 baselines.
Published source files and original Git blobs must match before any dispatch and
within every worker. Failure stops new dispatch and drains healthy active children;
no duplicate start/output overwrite. Fit queue never tests.

After six fits finish and real children/parent exit: independent CPU audits saved
best/last/config/source/data/gradients, all epoch order/affine/MixUp tail hashes,
IDs/labels/argmax/CM/support/macro/NLL and pair initial identity/config-onlyrho.
Only then exclusive durable pair/local/global locks, followed by one selected-EMA
test per arm. Test claim file plus directory fsync must precede test array loading;
failed claim stays, no retry/deletion. All six lock barrier before test labels.
Test result must actually attain main4/rho.3 overall>=.75 to pass, otherwise report
failure honestly; do not tune on test until it passes. CPU final audit reads saved
predictions, no model re-inference, curves/single-test stars/fullCM/support/summary
PNG/SVG/PDF/CSV, actual visual inspection, main local/GitHub/server synchronization.
Final deployment pack: selected PT/config/data/source SHA, all canonical/native
phase BMPs, deterministic per-stage amplitude export/replay code and receipts,
fixed masks/class mapping, explicit physical calibration checklist/session identity.
Offline export parity is not an on-bench hardware performance certificate.
Human approved memory-only3GiB->2GiB transition after actual~.6GiB use. The first main4/rho.3 stopped at committed E7; raw/EMA/Adam/scheduler/history remain in the prior run. New mem2 run resumes E8 without resetting optimizer/EMA/scheduler or replaying completed epochs. Other five arms start fresh with identical seed17/data/training hyperparameters. Resume receipt binds both parent PT/source Git blobs and unchanged optical model source.

## Independent deployment export completion (2026-10-13)

The completion-only export_caltech_deploy234.py saves actual478x478 canonical phase
BMPs separately from1920x1200 native phase BMPs, amplitude multipliers, fixed router/
detector masks, classes, frozen config and source/checkpoint SHA. Dynamic sample
amplitude/native BMPs use the frozen model.physical_fields via save_sample_bmps,
with active and illuminated-support mean/p99/max and zero/saturation statistics.
No extra peak normalization, optical graph or training source change is introduced.
Ten Linux CPU checks (four claim ordering/refusal and six joint encoding contracts)
passed, plus a real completed L4rho.3 selected checkpoint static-export check.
These checks perform no sample propagation/model forward, GPU or device acquisition.
Offline equality still requires a fresh calibrated hardware session before bench use.
