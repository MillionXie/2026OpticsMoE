# Evidence-driven feature relationship guidance, fixed deployed graph

Human continuous optimization authorization covers TRAIN teacher strategies,
anti-overfit training, alpha≥.6 and TEST PT selection; no schedules. First L2/L4
276-source noise/dropout trials completed own saved science, no improvement,
retained tune0. Public frozen TRAIN-only teacher had TRAIN100/VAL98.9556, student
VAL77-79 despite TRAIN94-97. New hypothesis: guide semantic feature geometry
rather than only class probabilities. This is a hypothesis, not proof of cause
or promised10pp. Two fixed new cases L2GPU2/L4GPU1; old6GPU0/SAMGPU5 may finish
concurrently, total≤4 one/card, no foreign jobs/A100 changes.

TRAIN cached official ResNet18 pooled512 feature IDs, subtract TRAIN mean and
unit normalize; exact same MixUp pairing/lam as student/labels/KD. Compare their
off-diagonal cosine Gram with student's pre-head256 feature Gram. No learned
projection, no teacher at VAL/TEST/deployment, no new inference parameters.
Inspired by relational/similarity-preserving distillation (Park CVPR2019 and
Tung/Mori ICCV2019), adapted cosine-Gram variant, not exact paper reproduction.

Gradient computation: no-grad full-macro optical feature pass, compute exact
Gram loss gradient wrt detached full-macro features, then regenerate same
features with same stored CCD noise/phase mask per micro and backpropagate its
gradient surrogate plus CE/KD. No retained full-macro optical graph; no incorrect
micro-only pairs. Require replayfeature maxabs≤3e-6. Analytic NumPy-equivalent
float64 Gram derivative and toy fullgraph vs surrogate check precede fit; actual
all20/macro16/tail5 micro1/4 gates include this complete path. Relations calibrated
at the feature layer: scale=min(10,.3*||gradCE+KD||/||gradRelation||), not a claim
about phase-group norms. Same two-pass contexts, no BatchNorm running statistics.

Halve first-batch CCD/input perturbation after negative results, phase update
dropout1%, feature10%, RGB jitter10%, MixUp.2. Other retained weights/structure,
fresh Adam/EMA10new epochs and clean TESTevery2 selection unchanged; alpha uses
precise six/double coordinate repair preserving initial effective alpha exactly.
Current original retained L2/L4 heads/optical70 parents, nores50, data roles and
prior failures remain immutable. TEST selection is development, not independent
generalization or equal-budget opticalrho causality. Exactly two new candidates,
not a blind hyperparameter grid. Stop at10epoch evidence review, do not reopen
completed workers. Own rawexit/science/completedSHA and0forward saved audit
required, all old artifacts retained, source only Git/rawJSON/PT/IDs private.

References: https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html
and https://arxiv.org/abs/1907.09682 .
