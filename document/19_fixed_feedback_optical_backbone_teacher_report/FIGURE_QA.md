# Figure contract and QA record

## Figure 1 contract

- **One-sentence conclusion:** source-fixed inter-stage optical feedback tracks exact BP across classification, segmentation and pose, while ImageNet body training supplies transferable semantics and random feedback weakens when that body pretraining is removed.
- **Archetype:** schematic-led composite with three quantitative dot-summary panels.
- **Inputs:** `source_data/downstream_runs.csv`; 72 run-level values, two body regimes × three tasks × four methods × three downstream seeds.
- **Transformations:** metric values are multiplied by 100; panels show paired seed differences. No row is excluded.
- **Uncertainty:** individual seeds plus mean ± sample SD (`ddof=1`, `n=3`). The No-ImageNet runs share `body_init_seed=2026`; their SD is conditional downstream variance, not backbone-initialization variance.
- **Color semantics:** green = ImageNet-pretrained body; light blue = No-ImageNet body; blue = optical/current operator; magenta = source-fixed feedback; orange = exact-BP electronic scaffold.
- **Intended output:** 182.9-mm double-column figure; editable SVG/PDF, 600-dpi TIFF, 300-dpi PNG preview.

## Figure 2 contract

- **One-sentence conclusion:** function-preserving growth makes 100-stage/15.05-M-phase computation graphs feasible, but the active 16-stage ImageNet run is still before full-depth validation and cannot yet establish semantic scaling.
- **Archetype:** four-panel quantitative status figure.
- **Inputs:** `source_data/p13_growth_history.csv` and `source_data/scale_audit.csv`.
- **Transformations:** parameter counts are converted to millions; fractions to percentages. No stochastic aggregate or inferential test is applied to P13.
- **Uncertainty:** none: the curve is one ongoing run, parameter counts are deterministic, and CUDA timing is an engineering audit with no performance interpretation.
- **Guardrail:** alpha=1 engineering audits are explicitly labelled as non-semantic; full-depth ImageNet begins at growth epoch 10.
- **Intended output:** same formats and width as Figure 1.

## Automated and visual QA

- Plot backend: Python/Matplotlib 3.7.2 via `C:\ProgramData\anaconda3\python.exe`.
- Static Nature preflight: `20 pass, 0 warn, 0 fail` with `--backend python --strict`.
- PDF font audit: Figure 1 minimum `5.9 pt`, Figure 2 minimum `5.9 pt`; both pass the `5 pt` floor.
- Visual inspection: completed on the final PNGs at original resolution; panel labels, titles, legends, axis labels, error bars and caveat annotations are legible with no overlap.
- Editable deliverables: SVG and PDF retain text; raster deliverables are TIFF (600 dpi, LZW) and PNG (300 dpi).

## Exact QA commands

```powershell
& 'C:\ProgramData\anaconda3\python.exe' .github/skills/nature-figure/scripts/validate_figure.py document/19_fixed_feedback_optical_backbone_teacher_report/figures/plot_teacher_report.py --backend python --strict
& 'C:\ProgramData\anaconda3\python.exe' .github/skills/nature-figure/scripts/audit_pdf_text.py document/19_fixed_feedback_optical_backbone_teacher_report/figures/fig1_fixed_feedback_evidence.pdf --min-pt 5
& 'C:\ProgramData\anaconda3\python.exe' .github/skills/nature-figure/scripts/audit_pdf_text.py document/19_fixed_feedback_optical_backbone_teacher_report/figures/fig2_depth_growth_status.pdf --min-pt 5
```
