# Multiplane comparison figure notes

## Figure contract

- **Core conclusion:** full-aperture sigmoid OEO preserves D2NN retrieval
  performance, whereas the current per-expert OEO MoE implementations fail to
  fit the task; repeated routing alone does not explain the failure.
- **Archetype:** quantitative grid.
- **Backend:** Python/matplotlib.
- **Final width:** 183 mm double-column figures.
- **Typography:** Arial/Helvetica fallback, 7 pt body text, 8 pt panel labels.
- **Exports:** editable SVG, PDF, 600 dpi PNG and 600 dpi TIFF.
- **Statistics:** one controlled run per architecture; values are descriptive
  point estimates. No standard deviation, confidence interval or error bar is
  available and none is invented.
- **Test use:** all displayed retrieval metrics are the final-epoch test
  metrics. The selection-biased best-observed test result is intentionally not
  mixed into the figures.

## Files

### `fig01_performance_summary`

- Panel a: final test Top-1 for all four primary experiments and the hatched
  fixed-router OEO MoE supplement.
- Panel b: final Top-3 and MRR.
- Panel c: paired final train/test Top-1, showing that both OEO MoE variants
  underfit rather than overfit.

### `fig02_learning_dynamics`

- Panel a: train Top-1 across all 28 partial epochs.
- Panel b: test Top-1 across the same epochs.
- D2NN and D2NN OEO converge together; continuous MoE remains intermediate;
  both OEO MoE variants remain near 20--30%.

### `fig03_parameter_and_phase_audit`

- Panel a: trainable phase count versus final test Top-1.
- Panel b: RMS phase displacement from initialization versus final test Top-1.
- The OEO MoE masks move by a similar phase magnitude to the successful
  baselines, so poor performance cannot be attributed simply to zero phase
  gradients or completely static masks.

### `fig04_router_selection_heatmaps`

- Values are sample selection rates in percent. Top-2 routing means every row
  sums to approximately 200%, not 100%.
- The continuous input router strongly collapses to experts 1 and 3.
- Repeated routing is relatively balanced in early stages, but later Vision
  and Language stages specialize sharply. Balanced early selection alone does
  not recover retrieval performance.

### `fig05_normalized_power_flow`

- Each curve divides mean stage output/reload power by that modality's mean
  stage-1 input power.
- Curves compare relative propagation/reload behavior; they are not absolute
  optical-efficiency measurements across different aperture geometries.
- D2NN OEO creates a stable reload operating level. Continuous MoE loses power
  gradually, while OEO MoE reload can reset or increase power without
  preserving task information.

## Source data

- `comparison.csv`: performance, parameters, phase motion and timing.
- `router_diagnostics.csv`: per-stage expert selection, importance and sparse
  routing weight.
- `power_diagnostics.csv`: per-stage input/output or reload power.
- `source_data/training_logs/*.csv`: complete 28-epoch training history for
  all five runs.

Regenerate every figure from the repository root with:

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval.plot_comparison_figures
```
