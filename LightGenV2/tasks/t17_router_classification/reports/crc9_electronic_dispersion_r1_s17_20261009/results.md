# MoE regularization candidates

Selection uses validation only; the precise rule is recorded below for each candidate. Only the winning candidate per architecture receives one exploratory test.

| Candidate | Weights | Fine-tune epoch | Val accuracy | Val macro | Train macro | Gap | Test accuracy | Test macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| electronic_load01 | ema | 21 | 82.17% | 78.56% | 96.08% | 17.52 pp | not tested | not tested |
| electronic_load05 | ema | 9 | 82.03% | 78.01% | 94.96% | 16.95 pp | not tested | not tested |
| electronic_load20 | ema | 6 | 82.59% | 78.43% | 94.82% | 16.39 pp | not tested | not tested |

## Selection and validation routing

- electronic_load01: validation score >= parent score - tolerance; maximize effective experts, then pair diversity, then accuracy/macro mean
  Validation effective experts: 1.992; dominant pair: 99.86%; mean power: [0.0, 0.0006039014551788568, 0.5663071274757385, 0.43309131264686584]; pairs: {'2,3': 0.001392757660167131, '3,4': 0.9986072423398329}
- electronic_load05: validation score >= parent score - tolerance; maximize effective experts, then pair diversity, then accuracy/macro mean
  Validation effective experts: 2.071; dominant pair: 97.77%; mean power: [0.009520284831523895, 0.0, 0.5735920667648315, 0.41688767075538635]; pairs: {'1,3': 0.022284122562674095, '3,4': 0.9777158774373259}
- electronic_load20: validation score >= parent score - tolerance; maximize effective experts, then pair diversity, then accuracy/macro mean
  Validation effective experts: 2.001; dominant pair: 99.72%; mean power: [0.0006049773655831814, 0.0006049249204806983, 0.5654886960983276, 0.4333012104034424]; pairs: {'1,3': 0.001392757660167131, '2,3': 0.001392757660167131, '3,4': 0.9972144846796658}

Run prefix: crc9_electronic_dispersion_r1_s17_20261009
Inference geometry, Top-2, two OEO stages and single Linear are unchanged. D4 augmentation and regularization are training-only. Baseline tests were previously seen; new tests are exploratory.
