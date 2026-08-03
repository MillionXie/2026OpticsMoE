# Why the Grocery phase masks barely moved

The old checkpoint was finite and the phase tensors had `requires_grad=True`,
but neither condition proves that the optical masks carried the task.

Measured physical phase standard deviations were:

| checkpoint | Vision | Language |
|---|---:|---:|
| old Grocery best | 0.0318 rad | 0.0135 rad |
| reproduced Grocery best | 0.2106 rad | 0.1309 rad |
| validated CIFAR best | 0.4494 rad | n/a |

The dominant causes are:

1. Grocery has very few optimizer steps.  A 10-SKU epoch is only about eleven
   PK batches, whereas the CIFAR mask receives dense supervision over many more
   batches.
2. The old optimizer grouped phase, adapters and other electronics together.
   Its logs did not report phase gradients or actual phase updates, so a falling
   retrieval loss could be mistaken for a trained mask.
3. The Vision Transformer residual supplied a direct hidden-state bypass around
   a weak optical delta.  Input/output adapters and the final electronic readout
   could reduce the loss faster than the Vision phase plane.
4. Hard Top-4 routing gives an expert mask exactly zero task gradient whenever
   that expert is not selected.  Historical logs often activated only 7--8 of
   16 Language experts in a batch; the original main config also disabled the
   router balance losses entirely.
5. The final Language output adapter was marked trainable even though retrieval
   consumes the detector readout directly.  It never received a gradient and
   inflated the reported trainable electronic parameter count.
6. Historical source configs had drifted from the resolved configs stored with
   the server checkpoints.  The actual canonical run used base LR `2e-3` in
   stages 1/2 and phase LR `1e-3` in stage 3.  The source files and tests now
   match those recorded artifacts.

`configs/grocery10_phase_engaged.yaml` addresses these issues without adding an
artificial phase-variance loss:

- independent adapter, readout, router and phase optimizer groups;
- phase LR `8e-3`, matching the validated CIFAR optical setting;
- Vision residual disabled so the feature cannot bypass the optical branch;
- router balance/importance enabled and per-expert selection coverage logged;
- after five joint epochs, every second epoch is phase-only block-coordinate
  optimization: electronics remain in the forward graph but are not updated;
- the unused Language output adapter is frozen;
- phase gradient RMS/max, per-plane missing gradients, physical phase standard
  deviation, run/epoch phase motion, sigmoid saturation and router coverage are
  written to `train_log.csv` and `metrics/phase_training_latest.json`;
- fixed-scale phase previews and raw tensors are saved below `phase_training/`.
- the small dataset is expanded to 100 deterministic PK optimizer steps per
  epoch; each pool wrap reshuffles samples and the image Dataset applies fresh
  augmentation, so the masks no longer receive only about eleven updates.
- k-space filtering is enabled at 2 degrees (about 84% pass fraction for the
  8 um grid), and per-plane phasor DC power is optimized from a small Gaussian
  raw-phase initialization.

Phase-only epochs do **not** add a loss that merely forces a visually busy mask.
They optimize the original KD/retrieval/gallery objective while preventing the
electronic parameters from absorbing that epoch's update.

Interpretation guide:

- `phase_grad_rms == 0`: broken gradient path or no selected expert;
- finite gradient but `phase_delta_run_rms_rad < 0.01` after five epochs: phase
  LR/coverage is still inadequate;
- increasing raw magnitude with saturation fraction near zero: healthy;
- large `abs(raw_phase)>4` fraction: sigmoid saturation, do not raise LR further;
- unselected experts greater than zero for a complete epoch: router collapse or
  insufficient data coverage.
