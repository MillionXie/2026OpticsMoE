# MoE regularization candidates

Selection uses validation only; the precise rule is recorded below for each candidate. Only the winning candidate per architecture receives one exploratory test.

| Candidate | Weights | Fine-tune epoch | Val accuracy | Val macro | Train macro | Gap | Test accuracy | Test macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| recovery_lr03 | online | 13 | 81.48% | 77.80% | 90.13% | 12.33 pp | not tested | not tested |
| recovery_lr10 | online | 13 | 82.17% | 78.73% | 91.62% | 12.90 pp | 81.89% | 78.94% |

## Selection and validation routing

- recovery_lr03: validation score >= parent score - tolerance; maximize effective experts, then pair diversity, then accuracy/macro mean
  Absolute pre-dispersion validation score floor: 79.942724%; admissible: False
  Validation effective experts: 3.262; dominant pair: 40.95%; mean power: [0.13458602130413055, 0.13841108977794647, 0.5468651652336121, 0.18013769388198853]; pairs: {'1,2': 0.018105849582172703, '1,3': 0.27715877437325903, '2,3': 0.29526462395543174, '3,4': 0.40947075208913647}
- recovery_lr10: validation score >= parent score - tolerance; maximize effective experts, then pair diversity, then accuracy/macro mean
  Absolute pre-dispersion validation score floor: 79.942724%; admissible: True
  Validation effective experts: 3.262; dominant pair: 40.95%; mean power: [0.13458602130413055, 0.13841108977794647, 0.5468651652336121, 0.18013769388198853]; pairs: {'1,2': 0.018105849582172703, '1,3': 0.27715877437325903, '2,3': 0.29526462395543174, '3,4': 0.40947075208913647}

Run prefix: crc9_electronic_dispersion_recovery_s17_20261009
Inference geometry, Top-2, two OEO stages and single Linear are unchanged. D4 augmentation and regularization are training-only. Baseline tests were previously seen; new tests are exploratory.
