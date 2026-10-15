# Direct follow-up: hard targets and error-focused distillation

Both scheduled T18 and daily09 tasks were explicitly PAUSED at the latest human request. These stages were followed directly in the active conversation, not through scheduled wakeups. The current controller and all children have actually exited; zero T18 GPU children and no failure markers were read-only verified.

| Depth | Arm | Actual / selected epoch | Train % | Val % | Test % | Test macro % | Test gap pp |
|---|---|---|---|---|---|---|---|
| 2 | residual | 1170 / 1143 | 80.6833 | 75.7962 | 81.7239 | 76.7726 | 4.037097654118938 |
| 2 | frozen_no_residual | 100 / 98 | 67.9369 | 64.9682 | 77.6869 | 71.8949 | — |
| 3 | accepted_residual | 125 / 123 | 90.9330 | 81.5287 | 87.4523 | 84.6657 | 7.2013093 |
| 4 | accepted_residual | 765 / 488 | 99.6058 | 87.2611 | 93.3988 | 90.1104 | 9.492635 |

Latest state: **strategy_review_required_target_unmet**. Actual82% target flag: **False**. The original L2 test-gap4pp requirement is now met (4.037097654pp), while the later82% absolute-accuracy target is not met. No rounding into success. No-residual models remain at the actual100-epoch baseline, unchanged and not re-evaluated. Three/four-layer accepted results are historical protected models; L4 actual765 includes its last completed color trial, selected parameters remain E488.

The preceding217 hard-target consolidation actually trained50 more epochs to1140, selected E1117: train80.1577 / val75.7962 / test81.5603 / macro77.3899; gap3.873432pp. MixUp and label smoothing were removed after TRAIN evidence of underfitting, with full raw/EMA/Adam restoration. It reached review, not a predefined1140 cap.

The new221 strategy restores LAST1140/fullAdam54720 and strengthens KD only on actual TRAIN samples for which the frozen teacher is correct/confident but the current student is wrong (multiplier4; other gated samples keep multiplier1). The teacher gradient had been measured aligned with supervised loss but small. Hard targets, focal1.5, TV.01, capture.2 and EMA.95 are retained; once1141LR.0003 restarts scheduler statistics without resetting moments or EMA. Student inference remains the same two-layer479364 optical phases/6 tensors and no electronic classifier. No new class selection, test-ID targeting, architecture change or weakened control.

CPU proof c0417fee2bf8aed9ae64c9604972b91b11bc45eb7b13f3333fe0b74d0bb4aff3 and GPU2 numeric proof 88018ab22e04904742559de389d0b718b434dc0bcf365403a1539960eb7c6a93 each passed once. Every completed stage has its own producer actualrc0/exit and exclusive science/completed identity before a new selected-model test. The auditor independently reconstructs TRAIN gates, student-error masks, weighted KL from saved private probabilities, full histories/orders/LR/Adam states and selected saved-prediction metrics; zero extra CPU forward/test.

Run: caltech10_L2_error_focused_distill_s17_20261016. Training source: 213ba0cb5be0883cbc08cb8f5e114b777965661a (221 files, current/original verified). Final science SHA: 9419bc65849fad8f5be6718b3e5f7bc1d9592b27f5e5bd91aad73caff9ebc169. Selected checkpoint SHA: 9f1894338c1c24b3af5add209c394fb018a2a8cb2a01145cd2a9c9722ce2a802. Selected test result SHA: 2ae2c5848de70f48d2d787d19edfea2de4b51f377cad86adb5f4c476702d55b7. Saved test audit SHA: 4bfb328d2edb9544e88e39a2ba8385665508542ea6ae25ccec77c22bb8edbacf. New selected digests tested: 1; inherited: 40; total unique known: 41. Reused parameters reuse results; no duplicate forward.

Fixed train761/val157/test1833, test supports315/80/678/680/20/16/13/9/14/8. Validation-informed class screening and repeated test-informed development/stop feedback mean these are development metrics, not an independent unseen generalization estimate. Residual arms receive extra budget and a training-only optical teacher; this is not an equal-budget pure residual causal comparison. Whole-prefix first minimum validation EMA balancedNLL selects models, never highest test accuracy. Macro and all class recalls are retained, including regressions. Exact-ID/pixel disjointness does not prove near-duplicate or subject independence. All old sources/PT/data/history/claims/failures/figures remain protected; raw predictions/probabilities/PT/logs remain private.

The last220 epochs of un-smoothed saved train/validation metrics are in epoch_metrics.csv; privately rendered PNG/SVG were visually checked. Current endpoint metrics may differ from selected-model metrics; the table reports the same selected model across train/val/test. No additional strategy has been launched after this review, and no scheduled followup was restored.
