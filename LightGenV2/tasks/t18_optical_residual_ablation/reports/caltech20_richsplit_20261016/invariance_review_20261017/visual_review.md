# Actual visual review — 2026-10-17

Both final PNG figures were actually opened with view_image and inspected at full readable size. The first accuracy render had overlapping long gap labels; that PNG/SVG is preserved privately. Only annotation wrapping was changed, redrawn from the unchanged aggregate CSV and actually inspected again. No training, evaluation, saved metric, model, source claim or checkpoint changed for this layout repair.

- accuracy_depth246.png: all three depths share one figure; separate TRAIN/VAL/TEST panels, both optical-residual arms, consistent60–100% axes, clear legend, exact two-decimal accuracy labels. The three TEST gaps now have non-overlapping two-line percentage-point labels. No macro panel.
- optimization_curves.png: actual saved latest attempts, clearly marked not all adopted; TRAIN/VAL only, no invented per-epoch TEST or smoothed points. L2/L4 axes are additional optical epochs; L6 is same-head updates. Selection markers match the registered VAL policy.
- Compared plotted numbers to selected_summary.csv and training_curves.csv. The current retained models are TEST80.5353/81.7518/80.2920%, gaps5.3528/3.8929/1.9465pp. No10pp target was declared achieved. The lower-scoring candidates stay rejected and archived.
- SVG counterparts preserve editable vector output; plot_selected_accuracy.py redraws the aggregate comparison into a NEW output directory without models or private data.

Final PNG SHA256:
- accuracy_depth246.png: `8b26ba25a8a8241fcc8487fed0b1d8cddbc658b32c1094ea22de76e00ff29892`
- optimization_curves.png: `40db872c4402d38c1a5afdd17358c16d2a52ffe317b689d17f2131ed211c133d`

The reviewed PNGs remain machine-local under the repository raster-ignore policy. Published SVG counterparts are exported from the same reviewed Matplotlib figures. No PNG force-add or ignore-rule change was used.
