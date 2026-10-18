# Frozen L4 residual physical deployment (2026-10-18)

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
