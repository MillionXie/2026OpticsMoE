# MoE regularization candidates

Selection uses validation only: mean of accuracy and macro recall. Only the winning candidate per architecture receives one exploratory test.

| Candidate | Weights | Fine-tune epoch | Val accuracy | Val macro | Train macro | Gap | Test accuracy | Test macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| optical_early | online | 34 | 81.48% | 75.73% | 81.49% | 5.76 pp | 79.39% | 73.24% |
| optical_late | ema | 1 | 79.53% | 76.76% | 98.75% | 21.99 pp | not tested | not tested |
| electronic | online | 11 | 80.92% | 76.55% | 96.17% | 19.63 pp | 79.39% | 76.00% |

Run prefix: crc9_moe_regularized_r1_s17_20261009
Inference geometry, Top-2, two OEO stages and single Linear are unchanged. D4 augmentation and regularization are training-only. Baseline tests were previously seen; new tests are exploratory.
