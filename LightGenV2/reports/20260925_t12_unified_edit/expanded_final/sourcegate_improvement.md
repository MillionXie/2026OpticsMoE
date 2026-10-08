# 256² small-editor source-gate follow-up (2026-09-26)

The 128² model's apparent cleanliness was partly a scale effect: its predicted
replacement still contains faint source-object remnants, but downsampling
hides them. At native 256², the full-resolution source is much sharper and
the original output equation `tanh(atanh(reference) + delta)` forces the
decoder residual to erase it before drawing a new object. This is difficult
when the old and new silhouettes differ, even when pixel MSE is low.

## Change

The 256² small model now predicts a one-channel **source-retention gate** from
the final decoder feature map. Output is
`tanh(sigmoid(gate_logits) * atanh(reference) + delta)`.
This is a learned internal mixing coefficient, not a ground-truth mask,
post-hoc cutout or pasted target image. At inference the inputs are only
reference image, prompt and seed. The target-minus-reference difference is
used **only in training** to give the gate an auxiliary soft supervision
signal. Existing Qwen-mini two-block head, parallel electronic/optical
bottleneck, alpha constraint, and single-pass decoder are retained. The new
gate adds 49 parameters: 9,502,616 -> **9,502,665** counted parameters.

The model was warm-started from the first native-256 checkpoint and trained
four more epochs, with a lower backbone learning rate and 10× gate learning
rate. The best checkpoint was epoch 4. No from-scratch retraining or GAN was
used.

## Paired evaluation

Full 2,304-pair test RGB MSE, pixels in [-1,1]:

| Model | RGB MSE | RGB L1 | Edge L1 |
| --- | ---: | ---: | ---: |
| Native-256 before gate | 0.003694 | 0.02649 | 0.08230 |
| Native-256 after gate | **0.002484** | **0.02200** | **0.07325** |

The MSE reduction is 32.8%. On the exact same diagnostic subset of 288
held-out edit pairs, each mode improved:

| Mode | Before | After | Relative MSE reduction |
| --- | ---: | ---: | ---: |
| Background | 0.002084 | **0.001787** | 14.3% |
| Object | 0.003891 | **0.002475** | 36.4% |
| Joint | 0.005756 | **0.003754** | 34.8% |

The gate's mean source retention is 0.0716 on pixels where target and source
differ by >0.10, versus 0.9518 on pixels with difference <0.02. These groups
are defined only for auditing against held-out targets, not passed to the
model. Optical fusion alpha remains 0.5000. Complete expert routing and
per-mode metrics are in `sourcegate_audit_final.json`.

The final same-item gallery visibly reduces source-object bleed for pillows
and some tables, while preserving unchanged backgrounds. It does not fully
solve fine lamp stems, thin table elements, or material crispness. Do not use
the lower MSE as a blanket claim of photorealism.

## Timing caveat

`sourcegate_timing.json` is a new same-run benchmark on an RTX 4090, batch 1,
first language block to 256² RGB. The small electronic path was 9.41 ms;
adding the assumed 6.2682 ms optical-device delay gives 15.68 ms. The older
background-only Qwen-28 electronic baseline measured 76.33 ms in that **same
run** (4.87× ratio). GPU load changed between benchmark runs, so do not
compare the new absolute milliseconds directly with the earlier report's
7.06/63.53 ms. The optical delay is an assumption, not measured hardware.

## Files

- `sourcegate_best_model.pt`: final 9.503M-parameter checkpoint.
- `sourcegate_gallery_final.jpg`: input | target | generated comparison.
- `sourcegate_training_summary.json`: full training and test metrics.
- `sourcegate_audit_final.json`: same-pair RGB errors, alpha, expert routing,
  and gate behavior.
- `sourcegate_timing.json`: same-run software timing and optical assumption.

Prior 256² and 128² checkpoints are preserved separately; this change does
not overwrite them. Training, audit, gallery and benchmark processes have
finished and released their GPUs.
