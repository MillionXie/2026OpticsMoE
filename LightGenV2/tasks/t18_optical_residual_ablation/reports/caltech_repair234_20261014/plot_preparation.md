# Audited study plotting preparation

Independent `plot_caltech_repair_study.py`, implementationf91870ceba3e171301f0b856e1cd3711decfa9f7.
No current partial figures or PDFs were generated. Linux Matplotlib3.8.4 import and
missing-completed-CPU-audit refusal passed: nonzero exit before output creation,
model forward/data/test/device0, frozen126 sources unchanged. Guard smoke
`caltech_repair_study_plot_admission_20261014/checks.json` exists; never rerun it.
[Actual guard receipt](plot_admission_cpu_checks.json).

After all3actual fit/queue exits and real `cpu_report/audit.json` passed, the plot
CLI verifies saved fit/history/PT hashes and each original source Git blob before
exclusive output creation. It reads saved metrics only, with no model/dataset/test
arrays. Accuracy/macro/balancedNLL curves use measured sparse train and every-epoch
val. Stars mark each candidate's first-minimum-NLL valEMA; hollow diamonds mark
train measured using exactly that same selected EMA, including off-grid epochs.
Admission65%/60% lines are explicitly validation thresholds, never test75 goals.
No test points, future extrapolation or single-seed error bars.

Outputs: selected summary, measured curve points, full class support/recall and
full validation CM CSV; PNG/SVG/PDF curves, selected class recall and three
row-normalized validation CMs. Five PDFs total. Label all as revised20/L4rho.3
validation/regularization study, not a depth comparison or independent test.

Before the first actual PDF authoring call, apply the PDF skill and successfully
run its operation-start command exactly once with create/count5/pdf. Then generate
only in a new output directory, preserve failures, transfer generated artifacts
with SHA (never source outside Git), render each PDF with Poppler and actually
view the PNG pages before publishing. `artifact_manifest.json` starts with
pdf_visual_inspection_pending true; compilation/file presence alone is not visual
acceptance. No prepared script is a completed report or successful candidate.
Read the full skill before generating; current skill requires output/source file
citations when delivering PDFs, in addition to real visual checks.

Suggested actual command after the barrier:
`plot_caltech_repair_study.py --base <L4study> --out <L4study>/cpu_report/figures`.
Existing outputs must not be overwritten; a reversible layout repair uses a new
artifact directory and preserves its prior failure/manifest, never refits models.
