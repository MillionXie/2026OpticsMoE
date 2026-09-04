# Figure QA report

Generated: 2026-08-27.

## Automated source preflight

- Backend: Python/matplotlib only.
- Checks passed: 20.
- Warnings: 0.
- Failures: 0.
- Editable SVG/PDF text: passed.
- Raster resolution: 600 dpi.
- Final width: 183 mm.
- Minimum configured font size: 6 pt.

## Exported PDF text audit

| Figure | Minimum text size | Runs below 5 pt | Result |
|---|---:|---:|---|
| fig01_performance_summary | 6.5 pt | 0 | pass |
| fig02_learning_dynamics | 6.5 pt | 0 | pass |
| fig03_parameter_and_phase_audit | 6.2 pt | 0 | pass |
| fig04_router_selection_heatmaps | 6.0 pt | 0 | pass |
| fig05_normalized_power_flow | 6.5 pt | 0 | pass |

## Visual inspection

- All five PNG previews were inspected at exported size.
- Panel labels, axes, legends and annotations are visible without clipping.
- Compact method labels are used in the dense performance summary.
- The fixed-router supplemental ablation is encoded by gray/dashed/hatching
  rather than being visually promoted to a primary method.
- Router heatmaps use one common 0--100% sequential scale.
- No uncertainty bars are displayed because only one run exists per method.
