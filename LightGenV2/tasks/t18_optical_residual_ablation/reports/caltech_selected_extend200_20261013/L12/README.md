# L12: completed common200 continuation pair

CPU receipt: 2026-10-13T16:17:00.565718+08:00 (Asia/Shanghai).

Both arms completed200 and their training child processes exited. Real saved-artifact CPU verification passed, with no model re-inference or test array materialization. This completed-pair review is not the global all-eight selection lock; other arms continue.

Checks:76 frozen current/original Git-source identities; exact E100 raw/EMA/AdamW/scheduler restore;100-row history prefix;200-step configuration/checkpoints;61 finite nonzero phase gradients each epoch;200 independently regenerated MixUp hashes including tail12; new global epoch order/affine hashes; same pair initialization/order/augmentation and config onlyrho; saved train/val IDs, labels, probabilities, argmax, confusion, support, macro and NLL.

| rho | First-minimum val NLL epoch | Train accuracy | Val accuracy | Val macro recall | Val bNLL | Train-val gap |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 182 | 92.4927% | 61.8375% | 58.7921% | 1.4575342392 | 30.6553 pp |
| 0.3 | 150 | 95.9184% | 65.7244% | 63.1170% | 1.3526372406 | 30.1940 pp |

Validation-NLL selection is preserved: a higher last-epoch accuracy does not replace a lower-NLL selected checkpoint. The selected train values come from saved full unaugmented EMA evaluation at the selected epoch; they need not coincide with the every5 historical train grid.

- rho0: previous100 selected val accuracy 54.7703% -> 61.8375%; best val bNLL improved 19.376%. Selected train-val gap 14.2530 -> 30.6553 pp. E200 train/val 93.5860/61.4841%, gap32.1019 pp; E100 matched gap14.2530 pp. Last10 valNLL vs prior10 improves 0.489% (negative means worse); E190->200 trainNLL improves 7.267%. Additional100 time19864.45s, mean elapsed per added epoch including final evaluation198.64s; total old100 plus continuation36899.49s.
- rho0.3: previous100 selected val accuracy 60.7774% -> 65.7244%; best val bNLL improved 9.844%. Selected train-val gap 20.1264 -> 30.1940 pp. E200 train/val 97.5948/66.0777%, gap31.5170 pp; E100 matched gap25.7562 pp. Last10 valNLL vs prior10 improves 0.212% (negative means worse); E190->200 trainNLL improves 2.683%. Additional100 time20621.89s, mean elapsed per added epoch including final evaluation206.22s; total old100 plus continuation37766.16s.

Extending the budget gave additional validation benefit while the training-validation gap enlarged substantially. Both validation accuracies at the NLL-selected checkpoints remain below70%. Late validation behavior and the selected epoch must be assessed separately from training fit; no claim of sufficient convergence, universal depth benefit or independent generalization. Data/regularization remained fixed, but this is single-seed exploratory development following validation screening and observed previous100 tests.

GPU policy remains physical0/1/2/4 only; A100 physical6 is excluded for training and endpoint testing. No other process was killed. New foreign occupation of0/1/4 slowed running children and blocks queue dispatch until a permitted card is idle; the original ETA is superseded by the live contention estimate.

Outputs: `cpu_audit.json`, `fit_review.json`, selected/curve/support/fullCM CSV, complete and late PNG/SVG, and the two-page `learning_curves.pdf`. No partial selection/test lock created.
