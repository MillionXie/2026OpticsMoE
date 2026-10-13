# Four-layer residual regularization study: bounded failure

All three fresh candidates completed150 epochs with exit_code0. CPU parent3210879
and GPU children3211094/3211169/3211235 exited; actual GPU2/5/6 project allocations
released, including A1006. Complete saved CPU audit passed:126 current/original
source Git blobs at46011b823, data/PT/config/init, every independent150order/affine/
MixUp tail14, Adam11700 steps/scheduler150, all11 phase gradients, selected first
minimum validation EMA and saved sample IDs/labels/argmax/CM/support/macro/NLL.
No model forward or test in that audit. No test arrays/inference/selection locks.

Run `caltech_revised20_s17_mainL4rho03_RGB_dense_global_regstudy150_gpu256_20261014`;
revised20 train1246/val261/test430, seed17, actual8bit RGB/dense4/preserve478 graph.
These are validation development results; observed old test informed class changes.

| MixUp alpha | Selected epoch | Train accuracy | Val accuracy | Val macro recall | Val balanced NLL | Same selected EMA gap |
|---|---:|---:|---:|---:|---:|---:|
| 0.2 | 137 | 92.7769% | 63.9847% | 62.6511% | 1.413775 | 28.7922pp |
| 0.4 | 146 | 90.4494% | 62.0690% | 60.1525% | 1.437414 | 28.3805pp |
| 0.6 | 149 | 91.6533% | 64.3678% | 63.6179% | 1.426896 | 27.2855pp |

No candidate passed the predeclared admission criteria, so no profile was chosen.
All three passed macro>=60%, no selected train/valzero, persistent-zero limit and
all-four nonzero/input-dependent routing. All failed validationaccuracy>=65% and
gap<=15pp. At the selected EMA, alpha.2 also failed each-class recall>=20% because
crab was1/10; alpha.4 failed because kangaroo was2/12; alpha.6 passed every-class
20% floor but still had only64.3678% validationaccuracy and27.2855pp gap.
Higher isolated validationaccuracy at another epoch was not substituted for the
first minimum-NLL selection. Final overalltest75% remains unassessed, not passed.

The composite repaired graph can now fit the training set to90-93% at selected
weights. The earlier low training fit is therefore substantially alleviated on this
revised task, while generalization remains poor. Increasing MixUp to.6 reduced the
selected gap somewhat but did not control overfitting adequately. The graphs and
class set changed together relative to the earlier task; this is not a clean causal
attribution of recovery to one factor or evidence of residual superiority. All study
arms have rho.3; no rho0 comparison was trained in this study.

Selected class behavior illustrates the mismatch: crab train88.24/90.20/84.31%
versus val10/30/20%; umbrella train98.08/98.08/92.31% versus val54.55/36.36/27.27%;
joshua_tree train100% in all arms versus val60/60/50%. Full support, recall and
confusion matrices are retained. Small validation supports10-20 mean class recalls
are coarse and variable; do not silently delete further classes or claim one cause.
Validation NLL still improved late; this does not prove full convergence or adequate
accuracy/generalization. Full routing weights remain nonzero and vary by input;
nonzero is not balanced, and does not resolve this gap.

Conditional2/3validation and formal six-arm training were not started. No extra
candidate, budget extension, automatic retry, test tuning, selected-native claim or
partial/final selection lock was created. Failed performance admission is distinct
from numerical/audit failure. Preserve all best/last/history/predictions/old runs.
Further GPU experiments require a new bounded plan; continue only saved CPU diagnosis
within the existing scope until that plan is authorized.

[Exact selected summary](selected_summary.csv), [measured curves](curve_points.csv),
[full per-class support/recall](class_support_recall.csv), [saved CPU audit](cpu_saved_audit.json),
[actual PDF visual review](visual_review.json). Five PNG/SVG/PDFs show measured train,
every-epoch validation, selected validation stars and same-EMA train hollow diamonds;
no test points. Every PDF was actually rendered and inspected. First export remains
at run/cpu_report/figures; endpoint margin correction used separate figures_reviewed
with no inference/retraining or metric change. Artifacts transferred with SHA; source
only through Git. Offline BMP parity preparation remains protected; no hardware session.
