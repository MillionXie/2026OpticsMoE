# Same8554 CCD head, error-only supervised residual fine-tuning

Continuing the human's residual-electronic-only authorization; all optical weights,
all control3 heads and accepted L2head12 stay frozen. No architecture addition or
physical capture change. Restore the trained classification-preserving L3/L4
heads selectedE12, including identical simulation TRAIN mean/std. Model/feature
implementation is exactly caltech10_sim_head_model.CCDHead;8554 parameters.

The previous9-new-epoch review had L3 test87.8887/gap6.5466 and L4 test93.2897/
gap9.3290, not sufficient to preserve prior gaps. Its fixed .05 correct-example
supervised term still has weight758*.05=37.9 sample equivalents in L4 vs3 actual
original optical TRAIN mistakes. This is an example-weight observation, not a
measured gradient-norm claim. Remove that correct-example CE term entirely:
TRAIN original optical-correct examples receive only KL(base||new), incorrect
ones receive only normal CE. Frozen correctness gate and cached optical base;
no test images/labels enter the gradient. KLweight1, lr.0001/wd.001, batch64/tail57,
seed/order and all graph operations unchanged. Explicit head Adam reset for loss
change; trained head weights/buffers restored, never random or E0 selection.

3-epoch segments, original accuracy+gap double targets and unchanged trained
control3. Highest validation accuracy then minimum balancedNLL selection was fixed
before these NEW tests; trained parentE12 remains eligible, never highest TEST
epoch. Review after>=9 newepochs without valaccuracy peak6 and .5pp train progress6.
New/global head epochs distinguished (global12+new); no optical optimizer steps.
CPU analytic objective gradient proof, actual cached-feature micro1vs8 proof,
independent saved NumPy objective/selection/prediction audit all reused as code
with new fixed sources/profile and unique claims/output. All actual per-step
logits/labels/orders/terms independently recomputed; no optical forward. Stage
producer waits synchronously during CPU audit; final actualexit/rc0 checked
separately. Any failure preserved, no automatic retry/relaxed tolerance. Known
head tests inherit both prior runs and parameter-digest reuse, not a second test.
All old runs and data remain protected. Source-only Git; cached features/PT/logits/
IDs stay private. Same source and trained head used for physical fine-tuning;
coordinate through the human-authorized lab owner, no device access here.
