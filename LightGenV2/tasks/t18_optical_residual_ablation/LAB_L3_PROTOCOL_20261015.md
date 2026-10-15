# Caltech10 three-main-layer laboratory deployment

Latest human approval: perform a session-only measured gray-to-field-amplitude
inverse, preserving the original device LUTs. Both paired models use the same
mapping, without changing weights, labels, geometry or per-image normalization.
Bounded acquisition at200us/Gain_X4:17 flat expert-aperture gray levels, three
repeats, followed by independent seven desired amplitude levels on both expert
and center supports (69 frames total including zero/full references). Fixed
previous expert phase; ascending/descending repeats record drift. Reject any
frame with saturation>=0.1%, insufficient dynamic range, or held-out normalized
intensity RMSE>0.04 / maximum error>0.08. Fit an isotonic intensity curve and
invert target I=(gray/255)^2, not target I=gray/255. Preserve raw images and all
receipts, calibration phase/coordinates/source hashes and256-entry mapping.
Calibration dark is used only to fit the response, not silently subtracted from
network input. A passing calibration still requires a fresh TRAIN pilot before
full TEST; this is a radiometric calibration, not a result or electronic fit.

The first69-frame measured inverse failed held-out validation: normalized
intensity RMSE0.0741/max0.2479; original device LUT hash remained unchanged. Do
not deploy it. Full expert-aperture means varied31.37-35.61 at identical gray255;
center gray237 even exceeded its earlier255 reference. One bounded follow-up
uses the same69 patterns/exposure but nine fresh frames per unchanged display,
retaining all621 raw CCDs and per-frame means/IDs. This separates single-frame
temporal variability from a repeatable spatial transfer. No loosened tolerance,
per-image normalization, TEST use or blind repeat is permitted. If this averaged
calibration still fails, retain evidence and stop before full TEST. Averaging for
calibration is not automatically applied to inference or called a performance gain.

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

Runtime admission note: the first cross-platform CPU comparison failed an overly
strict1e-6 criterion (Windows vs Linux differences4.6551e-5/2.3693e-5; ten TRAIN
argmax predictions identical, input difference2.98e-7). Keep this failed evidence.
Do not call that comparison exact. Native/BMP parity still requires<=1e-6, and
the original frozen training forward/encoder bodies must independently match the
deployment body on the same Windows runtime at<=1e-6. Cross-platform numerical
differences are separately recorded, with calibration argmax identity required.

First real pilot stopped after11 CCDs: router ten p9924-33 and PCC.51-.60;
first expert p9911/PCC-.0011, raw maximum23 with background mean6.64 at400us.
All evidence retained, task rc1, SDK lock removed; no full TEST started.
Bounded signal diagnosis uses exactly six frames:400/2000us times black, same
TRAIN router and same TRAIN expert patterns. No masks/ROI/gain/encoding change,
no accuracy selection, no exposure sweep beyond these two values. Record raw
background, phase response, saturation and signal shape, restore camera settings.
If2000us provides usable signal, a new two-arm session may fix2000us equally;
never mix exposure identities, silently replace old data or alter the dark guard.

Six-frame result: expert and black nearly indistinguishable at both exposures;
at2000us means12.4051 vs12.3933, whereas router23.1232, PCC.5611. Exposure alone
is not a valid recovery, despite background p99 exceeding15. Do not start TEST.
Next bounded nine-frame diagnostic holds the same expert phase and400us, using
black plus center224 and four expert apertures at gray64/128/192/255. Record raw
full-sensor and cropped responses. This diagnoses gray response vs illuminated
support, does not create/apply a LUT, stretch model inputs or move optical devices.

Report direct physical scores first, overall/macro/support/confusion matrices and
both checkpoint identities. If poor, user permits electronic adaptation, but the
original model has no learned classifier. Latest human instruction explicitly
requires TRAIN-only fitting and TEST-best checkpoint selection, without VAL
acquisition or selection. Define a separate small readout adaptation; report
it as adapted/hybrid, never replace the direct optical ablation. Do not use TEST
gradients or silently add a large classifier. Report TEST-selected development
accuracy, not independent generalization. TRAIN capture for both arms adds6088
CCDs; no VAL capture. Acquisition and fitting time are separate from direct TEST.
