# Visualization Style Guide

The plotting scripts are designed for Word reports and early paper drafts.

## Fonts

- Preferred font: Times New Roman.
- Fallback: DejaVu Serif.
- Default font size: 10 pt.
- Axis labels: 10 pt.
- Tick labels: 9 pt.
- Legend: 9 pt.
- Title: 11 pt.

## Export

Every plot is saved as:

- PNG at 300 dpi
- PDF
- SVG

The scripts also save the source plotting data as `<figure_name>_plot_data.csv`.

## Figure Size

- Single column: 3.35 inch wide.
- Double column: 6.8 inch wide.
- Default: 6.5 x 4.2 inch.

## Lines and Markers

- Line width: 2.0.
- Marker size: 5.
- Use solid lines for training curves and dashed lines for validation curves when both are present.

## Axes

- Remove top and right spines.
- Keep a light gray grid with alpha around 0.25.
- Avoid legends covering the main curve region.

## Colors

Model colors are fixed:

- `general_d2nn`: blue
- `fixed_route_moe`: orange
- `learnable_route_moe`: green
- `lenet5`: purple

Dataset colors are also fixed where applicable.

