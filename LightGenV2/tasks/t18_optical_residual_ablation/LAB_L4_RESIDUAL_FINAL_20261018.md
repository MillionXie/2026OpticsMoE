# Frozen L4 residual physical deployment (2026-10-18)

## Latest scope: this controller owns residual only; TEST-informed local repair

The user moved non-residual optimization to another chat/branch. This residual
controller must not modify or run non-residual fits. L2 residual remains paused;
the active physical target is the frozen L4 simulation346/411. A <=1percentage
point drop needs at least342/411 full physical correct predictions (83.21168%).
This is an optimization target, not a guarantee.

The original router suffix fit reached331/411 with TRAIN100%; the parent
controller is held at its SDK-free boundary while its strictCPU audit finishes.
All1611 router frames and the original fit best/last remain immutable evidence.
The new `lab_caltech20_residual_guarded_20261018.py` continuation consumes that
completed fit, not the initial simulation PT and not any old downstream CCD.

After each ordinary downstream-mask/electronic fit, cache the exact retained
model's TRAIN1200/TEST411 terminal features on CPU and verify original-head
output parity. Diagnose TEST confusion and use its wrong examples only to plan
same-class TRAIN feature-neighbors. For each of at most6 difficult classes,
update only its existing257-parameter head row using TRAIN64 batches:16 nearby
positives,16 random positives,16 hard negatives and16 full-class replay. Use
squared classification-margin loss, probability KL on parent-correct TRAIN
samples only, parameter anchoring, bounded full-curvature Newton steps and a
0.05 row-update radius. At most48steps/class or180optimization seconds.
No new inference layer, no TEST gradients, no VALID, and no synthetic CCD.

Selection uses full CPU411 accuracy first; equal correct count may continue
only if NLL decreases by more than0.0001. A lower correct count is never adopted.
Save the original parent/best/last, all actual TRAIN indices, confusion plans,
new-correct/new-wrong counts, full-CPU graph replay and frozen-upstream SHA.
These are TEST-informed development results, not independent generalization.
Fixed-feature curvature is not applied to optical-mask Hessians. Optical suffix
optimization remains the existing bounded fit; row repair is additional and
does not replace per-layer mask optimization.

The controller then captures the next fresh stage with its exact selected PT.
Previous measured-prefix fields stay frozen. Head-only changes cannot alter an
SLM input; any later suffix mask/electronic input change still requires fresh
downstream CCD. No recurring monitor and no no-residual task launches.

## Latest human correction: layerwise mask adaptation, not head-only

The user explicitly corrected the initial fixed-mask acquisition plan. That
controller was stopped at the first complete router TRAIN1200+TEST411 boundary;
the1611 valid router frames and all original files are preserved. The old
head-only plan below is historical, not the current execution policy.

Current entry mode is `layerwise`, with
`lab_caltech20_L4_layerwise_profile_20261018.json`. Adopt the verified router CCD
arrays by immutable hardlinks and retain their original receipt paths/hashes.
After EACH complete stage's TRAIN then TEST, freeze the captured optical masks
and every upstream electronic parameter that determined its incident fields.
Optimize only the still-deployable suffix masks and original electronics/head;
then choose the first highest complete CPU411-TEST development checkpoint before
acquiring the next stage. No VALID and no TEST gradient. The original simulation
checkpoint stays immutable; experimental adaptation has its own best/last.

Router->fit four-stage suffix->expert0->fit three-stage suffix->global1->fit
two-stage suffix->expert2->fit last global suffix->global3->fit original terminal
5150. Each suffix gets60epochs/900seconds (last120/900), complete epoch boundary;
GPU4060 only while SDK released, no other compute owner, CPU TEST selection.
The original propagation transfer constant is retained in simulated suffixes
for reproducibility. Fresh downstream CCDs are mandatory whenever their input
or mask changes. Never use any old fixed-weight downstream CCD after an update.
The first-stage scope is measured router plus simulated suffix, NOT fullphysical.
The complete final scope is allfive physically measured stages,8055 valid frames.

## Initial fixed deployment record (superseded)

Latest human scope: pause further L2 residual fine-tuning after all three capture
stages are complete; deploy the finalized L4 residual model. Existing L4/L6
non-residual CPU refinements are not interrupted. No recurring monitor.

- Frozen simulation checkpoint: `runs/simulation/caltech20_rich20_L4_8394_L6_8248_gain_conditioned_pair_TRAIN_20261018/round01/L4/best.pt`.
- SHA256: `7b0d7f055a16f7f6bbf9858c9ecaea572ecd0b0d6e96c618630f4a4e077c09a5`.
- Simulation target identity: 346/411 (84.18491484%); not an experimental result.
- Explicit direct-effective float64 alpha; optical residual rho0.3; unchanged
  four-stage optical/electronic network and original Linear256to20 head.
- Hardware run: `runs/hardware/caltech20_rich20_L4_residual_final_20261018`.
- Reuse only the verified dataset identity from L2: TRAIN60/class1200 and all
  TEST411, no VALID. Never reuse L2 or no-residual CCD for this new checkpoint.
- Capture router, expert0, global1, expert2, global3 in order. For each stage,
  complete TRAIN then TEST consecutively: 5x1611=8055 frames.
- First deployment keeps all optics fixed and records a full physical baseline;
  then fits only the original5140 readout on TRAIN, CPU120epochs. TEST selection
  is explicitly a development metric, never gradients or independent validation.
- Full411 CPU simulation/source-bridge parity and layer-major replay must pass
  before SDK use. Parity compares both graphs on the same host; cross-host FFT
  probability differences are recorded separately, not called bitwise equality.
  The laboratory fullTEST result must independently remain346/411. Historical
  simulation-cache/CPU comparison changed one already-wrong sample's predicted
  wrong class, with every sample's correctness unchanged; that evidence remains.
  First24 TRAIN frames are retained, signal-gated full-capture
  frames, not an additional acquisition or automatic retry.
  The source reference is a SHA-verified TorchScript export, independently
  compared with the original source model on all411 examples on the server.
  It is used only for audit, never hardware acquisition or optimization. Its
  fixed batch8 pads the last3 with5 zeros and excludes all5 padded outputs;
  this avoids installing simulation-training-only dependencies on the bench.
  The checkpoint omits the non-persistent angular-spectrum transfer buffer.
  Linux/Windows regeneration differed by5.96e-8 in complexfloat32. For the
  numerical audit, copy the original reference's transfer buffer after checking
  that all masks match exactly and transfer difference is<1e-6. Then enforce
  full411 source/bridge output agreement<=3e-6. This does not change wavelength,
  distance, inference parameters or hardware encoding: actual propagation is
  always real CCD during physical capture, not this simulation-only buffer.
- Preserve the current 20261015 coordinates,2000us/Gain_X4/wait240,1016 active
  native pixels, HV-inverse phase, vertical camera flip, zero-border amplitude
  encoding, session amplitude inverse, and linear-voltage phase LUT. The latter
  is not a measured linear-phase calibration. p99<15 or saturation>15% stops;
  preserve all frames and diagnose, do not silently retry or change exposure.
- StrictCPU best/last, baseline, full predictions, all phase/data/checkpoint
  hashes and SDK release must be audited after completion. A launch is not a
  completed measurement; runtime receipts remain the source of actual progress.

Entry: `lab_caltech20_L4_residual_final_20261018.py`. Its shared capture/readout
helper retains the original1000-TRAIN no-residual path while explicitly admitting
the1200-TRAIN residual deployment. Simulation training files are not modified.
