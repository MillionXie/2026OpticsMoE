# Caltech10 three-main-layer laboratory deployment

Human request 2026-10-15 authorizes this new bench session, superseding earlier
offline-only/no-lab limits only for these two L3 models. Other training windows,
historical runs and equipment configurations remain protected. Use the existing
lab main checkout; source travels by Git, assets by manifest and SHA256.

Selected residual E123 SHA b3d89f3ab2b177bd85997453cb0d0f8289d25b600243ca59c824ac529f59172d;
control E100 SHA 62154e9067aa75494189955421a0d32e74cbe8ff14b4f3caf929262ed11445cc.
Simulation test 87.4523% /80.2510%; residual actually trained125 vs control100.
These are accepted test-informed development candidates, not a new equal-budget
causal comparison. Never deploy historical20-class or nine-expert weights instead.

Frozen ten classes and original detector indices0,1,2,3,4,7,14,15,16,19;
train761/validation157/test1833. Preserve original IDs, pixels and role assignments.
L3 means expert/global/expert plus separate router: four captures/sample/arm,
14664 full test CCDs for both arms. Calibration first uses ten TRAIN images, one
per class, eighty CCDs total; this is not a test accuracy estimate.

Exact graph: fixed RGB100-channel mosaic to224; continuous four-expert optical
routing; preserve478 field for global input; exact existing row-LN/log/ReLU OEO.
No new trained electronic branch, normalization, peak stretching or residual
approximation. Joint amplitude abs(M) and phase angle(M), M=.7exp(i phi)+.3 for
residual main stages, routerrho0. Both actual8bit encodings inside simulation.
478->1016 nearest native mapping with current dual8um devices, phase hv_inverse,
camera flip_v per established machine convention; verify physically on TRAIN.

New coordinates from E:/code/guest/20261015.txt, SHA
cef79c00e0fb1c1d07db0411f58f342283c7d673b4309717773aee5036ff982e:
TL576,212; TR1370,223; BR1359,1011; BL563,999. Old camera corners cannot be reused.
Initial400us/Gain_X4/wait240ms follows this lab's existing SHS setup; no automatic
exposure scan. Record readback, LUT/SDK identities, phase and every input/CCD SHA,
fresh-frame IDs, signal statistics and elapsed time. Stop on p99<15 or saturation
>15%; retain failed evidence, release SDK, bounded diagnosis before new session.
No inference that SDK acknowledgment proves optical phase response.

Single SDK owner, layer-major dataset order, initialized hardware retained across
layers. Canonical input PNGs, raw sensor PNGs, cropped CCD and receipts preserved.
Only just-generated transient native input BMPs are removed after capture, with
canonical pixels and both hashes retained. Never delete valid experimental data.
No blind restart of existing output. CPU strict checkpoint/bridge/native encoding
checks must pass first. Pilot signal/phase/orientation must be inspected before
full TEST launch; derive ETA from actual full-pilot wall time, not sensor FPS.

Report direct physical scores first, overall/macro/support/confusion matrices and
both checkpoint identities. If poor, user permits electronic adaptation, but the
original model has no learned classifier. Latest human instruction explicitly
requires TRAIN-only fitting and TEST-best checkpoint selection, without VAL
acquisition or selection. Define a separate small readout adaptation; report
it as adapted/hybrid, never replace the direct optical ablation. Do not use TEST
gradients or silently add a large classifier. Report TEST-selected development
accuracy, not independent generalization. TRAIN capture for both arms adds6088
CCDs; no VAL capture. Acquisition and fitting time are separate from direct TEST.
