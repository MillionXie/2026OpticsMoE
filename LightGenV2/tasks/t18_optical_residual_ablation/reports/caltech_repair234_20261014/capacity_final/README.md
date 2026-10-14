# Completed phase-grid capacity diagnostic, 2026-10-14

All three L4rho.3 arms completed150epochs. CPUparent3509387 and GPUchildren
3509913/3509980/3510049 exited rc0; the three diagnostic allocations released,
while the separate full-resolution RESISC pair still runs on physicalGPU2/6.
No new capacity fit, depth queue, selectednative, selection/suite/test lock or
test evaluation was started. Original raw0/full12864 budget is finished.

The complete shape-aware saved CPU audit passed: original/current135 sources
at cb7bfb3e2e34cb1fb03860b3db13649176b73c0f, PT/config/cache/manifest, init
shape/count and shared initial physical-field proof, all150independent order,
affine and MixUp.6 including tail14, Adam11700/scheduler150, all11finite/nonzero
phase gradients and saved train/val sample IDs/labels/argmax/CM/support/macro/NLL.
No model forward or test arrays/inference in this audit. The exclusive dispatch
claim, success receipt and audit remain server-side; never repeat the auditor.
Numerical/provenance audit success is distinct from performance eligibility.

Each arm uses its first minimum validation EMA balancedNLL, not maximum accuracy:

| Main phase grid | Total free phases | Selected | Train % | Val % | Val macro % | Val bNLL | Gap pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full grid | 908552 | E147 | 86.9181 | 62.8352 | 61.6862 | 1.490373 | 24.0829 |
| Main grid 128 | 214016 | E146 | 67.9775 | 58.2375 | 57.4074 | 1.792691 | 9.7400 |
| Main grid 64 | 91136 | E150 | 56.9823 | 48.6590 | 48.5544 | 1.996249 | 8.3233 |

Eligible=[] / chosen_profile_case=null. Full fails validation65% and gap15pp;
128fails validation65%, macro60% and per-class20% (kangaroo1/12=8.33%);
64fails validation65% and macro60%. Both coarse gaps pass, but their lower
training fit and validation accuracy do not meet admission. All three selected
train/val have no zero-recall class, last20zero fractions pass, allfour routing
slots remain nonzero and weights vary by input; this is not proof of balance
or solved generalization. Per-class weights are retained in the saved CPU audit.

Within this fixed150epoch/seed17/rho.3 comparison, coarser phase control reduced
the train/validation accuracy gap while also reducing training and validation
accuracy. It does not support adopting a smaller phase grid as the solution.
It is consistent with a capacity restriction affecting fit; capacity and
optimization effects are not separated, nor is capacity proven the sole cause.
Late validationNLL trends remain saved in late_validation_trends.json; decreasing
NLL or a smaller gap does not prove convergence or final75% success.

The physical experts224/global478/router224, four main E/G/E/G stages plusrouter,
fivecaptures, correctedRGB/dense4/preserve478/8bitjointM/OEO/20windows are unchanged.
All three mainraw0 initial physicalfields match; parameter tensor hashes differ
with grid shape. The earlier random-main MixUp study is not this full-control
pair. No rho0 arm here, so this study cannot establish residual causal benefit.
Revised20train1246/val261 remain development evidence because previous exclusion
was informed by old observed test; no further class filtering/split change.

The user's intended final network stays at full resolution. The independent
RESISC20 pair preserves all908552phase parameters with11198training images;
it tests a different dataset, not controlled Caltech accuracy recovery. Its
first budget remains30epochs/14400fitsec, no automatic extension or test. New
depth2/3 and formal six need genuine corresponding validation/native admission.

Five PDFs plusPNG/SVG/CSV authored from actual audited saved metrics, transferred
with SHA and each rendered as one page at120dpi by bundledPoppler. All five
final pages were actually inspected with view_image and passed; no replotting,
inference/retraining/test. Curves show sparse measured train, every-epochval,
selected first-minimum-NLL val stars and same selectedEMA train diamonds. Full
validation CM CSV and support/recall/last20zero/routing/eligibility CSV included.
Original completed regularization figures and every old marker/output preserved.
