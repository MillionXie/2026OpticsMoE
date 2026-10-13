# Latest continuation learning curves

Snapshot: **2026-10-13T10:38:10.880624+08:00** (Beijing time).

Depth2/4 remain at their completed parent100 endpoint, waiting for extension GPU slots. Depth12/16 continue toward200. Curves read saved JSON metrics; no training modification, model inference, test arrays or new test claim.

Train uses the same EMA on the full unaugmented training set every5 epochs. Validation is evaluated every epoch. Gap comparisons below always use the same epoch; current validation observations may be newer than the latest measured train point.

| Arm | Epoch | Latest val accuracy | Latest measured train | Same-epoch gap | First minimum val NLL epoch | Best val bNLL |
|---|---:|---:|---:|---:|---:|---:|
| L2_rho0 | 100 | 45.5830% | 53.5714% @ E100 | 7.9884 pp @ E100 | 100 | 2.0218826252 |
| L2_rho03 | 100 | 51.5901% | 57.7988% @ E100 | 6.2087 pp @ E100 | 100 | 1.8386655268 |
| L4_rho0 | 100 | 50.1767% | 62.4636% @ E100 | 12.2869 pp @ E100 | 99 | 1.8796339849 |
| L4_rho03 | 100 | 61.8375% | 76.8950% @ E100 | 15.0576 pp @ E100 | 100 | 1.5277262317 |
| L12_rho0 | 119 | 56.8905% | 74.8542% @ E115 | 16.5503 pp @ E115 | 115 | 1.6130437980 |
| L12_rho03 | 119 | 63.6042% | 91.3265% @ E115 | 26.3089 pp @ E115 | 115 | 1.4151831956 |
| L16_rho0 | 116 | 53.3569% | 73.2507% @ E115 | 18.8338 pp @ E115 | 115 | 1.6690818268 |
| L16_rho03 | 115 | 62.8975% | 88.9942% @ E115 | 26.0966 pp @ E115 | 115 | 1.4574119370 |

## L16 interpretation

- L16_rho0: first-minimum balanced validation NLL improved 7.112% relative to the previous100 selected minimum (1.7968775154 at E99 -> 1.6690818268 at E115). E107–E116 average validation NLL is 7.065% below the E91–100 average. Same-epoch accuracy gap grew from 11.8985 pp at E100 to 18.8338 pp at E115; balanced NLL gap grew from 0.4226 to 0.5214.
- L16_rho03: first-minimum balanced validation NLL improved 4.316% relative to the previous100 selected minimum (1.5231496312 at E98 -> 1.4574119370 at E115). E106–E115 average validation NLL is 4.119% below the E91–100 average. Same-epoch accuracy gap grew from 25.3917 pp at E100 to 26.0966 pp at E115; balanced NLL gap grew from 0.7327 to 0.8507.

L16 still has validation NLL benefit after100; accuracy is noisy and the residual arm is not monotonically improving. The wider matched-epoch accuracy/macro/NLL gaps indicate increased overfitting pressure while validation benefit remains. The residual accuracy gap was29.4005 pp at E110 and returned to26.0966 pp at E115, only0.7049 pp above E100; do not describe the latest point as continuously worsening. Its balanced NLL gap still increased from0.7327 to0.8507. The non-residual accuracy gap increase is larger:6.9353 pp from E100 to E115. This is not evidence of full convergence, nor that every depth must improve monotonically with additional layers. Regularization is unchanged as requested; selection remains first minimum EMA validation balanced NLL over1–200. Single-seed exploratory development, not an independent generalization or causal regularization claim.

## Files

- `accuracy_all.png` / `.svg`: all-depth train and validation accuracy.
- `balanced_nll_all.png` / `.svg`: all-depth train and validation balanced NLL.
- `L16_late_detail.png` / `.svg`: E70+ accuracy, NLL, matched-epoch gaps.
- `curve_data.csv`: exact exported saved metrics (fractions for accuracy/macro).
- `analysis.json`: numerical comparisons and selection epochs.
- `snapshot.json`: exact scalar plotting metrics projected from saved histories, original full-history path/SHA and queue status.
- `process_source_check.json`: preceding live PID/UUID/memory and76-source current/original Git identity verification (separate capture timestamp).

History epoch continuity and exact parent1–100 prefixes were checked at snapshot extraction. This plotting check does not replace the required all-eight200-epoch CPU artifact audit.
