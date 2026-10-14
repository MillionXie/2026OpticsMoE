# Three phase-capacity diagnostic arms actually started

Human clarified that the final network size is fixed, dislikes shrinking, but
explicitly permits this diagnostic promptly, asks why training samples are so few,
and allows a suitable dataset change if needed. No repeated approval request.
The experiment does not automatically adopt a smaller final network.

Run: `caltech_revised20_s17_mainL4rho03_phase_capacity150_gpu256_20261014`.
Actual start2026-10-14 09:29:10 Beijing, CPU parent3509387. GPU6full child3509913,
GPU2main1283509980, GPU5main643510049. Actualproc/cmd/CUDA UUID verified; parent
CUDA invisible, project uses only these three authorized GPUs; no other task changed.
Torch2GiB+800MiB limits, at most three fits each150epochs/14400fitsec, no retry/
extension/fourth fit. These are L4rho.3 diagnostic fits, not the formal six arms.

## Actual source and matched controls

Running135-source closure frozen at `cb7bfb3e2e34cb1fb03860b3db13649176b73c0f`:
original126 + existing matched export/checker2 + seven independent capacity files.
Current files and original Git blobs match. Old126/46011b823 and130depth preparation
unchanged. Report/main HEAD may change without changing this running closure.
All new main raw0 gives physicalphi=pi/M=-.4; identical seed17 router224. Newfull
control908552parameters, main128214016, main6491136. Expanded original expert224/
global478, main4+router5captures, correctedRGB/dense4/preserve478/8bit contract unchanged.
Same physical22-term phase smoothness, MixUp.6/LS.08/capture.05/TV.005/EMA.95/
lr.002/macro16tail14/micro1 and shared deterministic augmentation. The historical
random-main full-resolution fit is not the matched initialization control.

## Actual gates passed before fit

`cpu_initial_native_checks.json` is the original server receipt, SHA-verified
artifact transfer. All3 initial physical fields and initial synthetic outputs
exact; physical smoothness matches for represented constant phases. Three fresh
perturbed states verify parameterization-aware static planes/multipliers and9
random/dim4/255/black dynamic native sequences. Canonical/native reverse478exact,
probabilitymaxerror0, each5captures, black allamplitude0, matchedCCD/routinghelpers,
wrong224global/478expert/oldTop2/nonfiniteCCD rejected.21CPUforwards, dataset/test/
devices/GPU0. It is not trained-weight selection or device acceptance.

`gpu_numeric_checks.json`: actualGPU6, each20classes, scalar/micro1 reversed
gradient accumulation. All3 probabilityerror0, maximumgradientrelativeL2
4.881965e-7, minimumcosine0.9999999999998807, all11actual freephase gradient
tensors finite/nonzero. Same routerinitSHA; different-shaped parameterstateSHA
is expected. Each of the three production children independently passed its
fresh numerical gate, source/config/initialSHA recorded before training.

## Actual early saved state, not a performance conclusion

`live_start_observation.json` records09:35:49 three children healthy atE7,
pending0/completed0/failure0/test0, only authorizedGPU6/2/5 at872/664/660MiB.
One epoch approximately46-48seconds at that stage. Fullval8.43%,128/64val4.60%
are early observations, not converged or selected results and not evidence that
capacity is the cause/solution. Future timing must use actual logs and early-stop
rules, not an unconditional promise. Optical FFT canvas remains518 regardless
of parameter grid, so reducing trainable values need not strongly reduce epoch time.

`cpu_partial_summary.json` records real immutable last-state byte snapshotsE9
for all3. Current/original135sources/config/cache/manifest, independent initial
tensorSHA/shape/count, finite raw/EMA, all11phase per-epoch gradient tensors,
Adam702steps eachparameter/scheduler9 and independently regenerated all9orders/
affine/MixUp tail14 pass. Same common orders AND transforms/MixUp across all3,
config onlygrid/case/contract differs. CPU model_constructed/forward/images/test/GPU0.
Actual full receipt lives in run/cpu_partial_capacity_20261014/checks.json; outputs
and helper must not rerun/overwrite. This is not full end-of-fit CPU audit, chosen
profile, native selected trained weights or formal admission.

Original admission remains val>=65%/macro>=60%/eachvalrecall>=20%/nozero/
last20zero<=half/gap<=15pp/all4nonzero/inputvariation/selectednative. All3actual
completion and parent/childrenexit precede exclusive capacity-aware saved CPU audit.
No pending/partial lock/test, no otherepochaccuracy reselection. Coarse benefit
would be mechanism evidence, not automatic permission to replace the final network.

The previous3MixUp fits each150 completed and failed performance gates, source/audit
and reviewed5PDF preserved; new data/test/old weights not touched. The actual source
archive has9144JPG, selected20only1937; extra trainingdata option and its constraints
are in [data size investigation](../DATA_SIZE_AND_NEXT_DATASET_20261014.md).
Formal2/3/4rho0/.3 and finalonce-test75% remain uncompleted and unguaranteed.
