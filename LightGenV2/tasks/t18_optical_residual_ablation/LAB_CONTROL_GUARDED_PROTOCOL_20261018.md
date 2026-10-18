# Laboratory L4/L6 no-residual development refinement

The user's latest explicit request authorizes continued **laboratory** L4/L6
no-residual adaptation, informed by the chat “微调方法总结”. This does not reopen,
retrain or replace any frozen **simulation** no-residual control. The two roles,
checkpoints and claims must remain separate. No original data or PT is replaced.

The existing complete physical acquisitions remain immutable: L4 7055 and L6
9877 frames, TRAIN1000 (50/class), TEST411, no VALID. Starting PTs are the audited
terminal-joint laboratory selections, 270/411 and 247/411. Before fitting, replay
the full measured graph using those exact selected PTs and SHA-verified CCD
arrays; export matching features and require original TRAIN/TEST accuracy and
head-probability parity <=3e-6. Old original-head features are not interchangeable
with features after changed post-CCD electronics.

Only the original 5140-parameter Linear256->20 and bias are optimized. Selected
post-final-CCD convolution/alpha, all optical masks and every captured upstream
input remain frozen. No new layers, SDK, GPU or CCD capture. CPU two threads.
TRAIN feature normalization is folded algebraically into the original Linear.

First bounded stage: global TRAIN cross-entropy, two TRAIN-supported confusion
pairs selected using TEST diagnostics, squared TRAIN pair-margin penalty and
mild parent KL on parent-correct TRAIN examples. No TEST examples/labels enter a
gradient, no TEST-derived priors. Maximum 600 epochs or 240 fit seconds (complete
epoch boundary), no LBFGS fallback. Feature mixing/noise/dropout are disabled.
Profiles record all weights and exact warm PT SHA. Preserve baseline/best/last,
history, protocol, predictions and source/data identities, including failures.

Checkpoint selection is accuracy-primary on all 411 TEST examples. Equal
accuracy may be retained only when mean TEST NLL improves by >0.0001; NLL-only
and accuracy gains are recorded separately. Lower-accuracy states never replace
the parent. These are **TEST-informed development metrics**, not independent
generalization or same-budget causal residual evidence. Targets L4>=278 and
L6>=280 correct are not guarantees. Review new-correct/new-wrong and TRAIN/TEST
confusions after this stage before choosing any further bounded local-row fit.

L4 residual layerwise acquisition/training continues independently. Its entry,
model, generic layerwise helper, profile, hardware contract and SDK ownership
are protected by before/after SHA checks during this source-only Git deployment.
No recurring monitor or new branch/worktree/project is created.

## Completed first stage and evidence-led local-row continuation

First stage finished with full selected-feature replay and strict CPU checks:
L4 270->273/411 (66.42336%); L6 retained parent247/411 (60.09732%), not its
235/411 last state. The global head change did not solve six-layer errors.

The next stage uses `lab_caltech20_control_localrow_20261018.py`: update only one
existing category row (257 parameters), freezing the other19 rows per stage.
TEST error counts choose the category; TRAIN margins alone choose difficult
positives and negatives. Each64 batch has16 low-margin positives,16 random
positives,16 hard negatives,16 global TRAIN replay examples. No TEST features
enter neighbor selection or gradients. Squared signed margin plus parent KL on
parent-correct TRAIN and an L2 anchor; exact257-variable curvature solved through
a64x64 Woodbury system, bounded step and row radius. All other inference layers
remain unchanged. Synthetic autograd/Hessian and frozen-row checks must pass.

Budget: at most3 passes,4 classes/pass,80 updates/class or240 seconds, stopping
after a pass without accepted progress. Accuracy-primary/NLL-tie selection as
above; every update's TRAIN indices and all TEST correct/NLL results are logged.
This is finite development, not an unlimited candidate scan. Preserve all
parents, baseline/best/last, new-correct/new-wrong, full confusion and source SHA.
