# Saved capacity and RESISC figure preparation, 2026-10-14

Independent `plot_saved_validation_studies.py` supports the three phase-grid arms
and the original full-resolution RESISC rho pair without changing either frozen
training closure. Source implementation c447a63299a4a79b8d9143c345e1dee5dd008524.
Actual Linux CPU import with Matplotlib3.8.4 passed; both premature plot calls
were rejected because complete saved CPU audits were absent, before creating
figure output. Exact server JSON bytes are preserved in cpu_admission_checks.json.
This checks rejection paths, not successful authoring or scientific performance.
No model construction/forward, image/test arrays, GPU/device actions, PDF marker
or figures occurred. Existing five project GPU children were unchanged.

After that study alone fully completes and its actual parent/children exit,
its dataset/shape-aware auditor must pass. The plotter requires the real saved
CPU audit plus the one-shot dispatch completed receipt with matching audit SHA,
current/original training source identity, checkpoint/history hashes and saved
selected metrics. It refuses existing figure directories before the failure
writer; partial failures remain and must not be re-entered automatically.

CPU CLI: `plot_saved_validation_studies.py --kind capacity|resisc --base <run>
--out <run>/cpu_report/figures`. Capacity creates five PDFs; RESISC creates four.
For an actual PDF operation, first follow the PDF skill and run its marker once
immediately before authoring with that expected count. Transfer artifacts with
SHA, render every PDF with Poppler and inspect every page before publication.
No marker or authoring has been performed by this preparation.

Figures preserve sparse measured train, every-epoch validation, the first-minimum
validation EMA NLL star and the same selected EMA train diamond. Counts, budgets,
phase parameters and dataset labels are study-specific. CSV includes full
validation confusion matrices, support/recall, last actual20 zero fractions,
routing occupancy/std, eligibility checks and selected summary. Five/two-arm
cross-study curves are not mixed; no test points or single-seed error bars.
Capacity is a mechanism diagnostic, not adoption of coarse grids. RESISC is a
changed dataset, not controlled Caltech accuracy recovery. Validation65/macro60
are admission gates, not the final once-test75 goal.
