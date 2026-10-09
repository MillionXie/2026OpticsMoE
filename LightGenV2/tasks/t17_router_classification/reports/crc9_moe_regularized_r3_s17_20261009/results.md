# MoE regularization candidates

Selection uses validation only: mean of accuracy and macro recall. Only the winning candidate per architecture receives one exploratory test.

| Candidate | Weights | Fine-tune epoch | Val accuracy | Val macro | Train macro | Gap | Test accuracy | Test macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| optical_mild | ema | 16 | 81.75% | 76.50% | 82.60% | 6.10 pp | not tested | not tested |
| optical_balanced | ema | 14 | 82.03% | 77.12% | 83.10% | 5.98 pp | 79.67% | 74.50% |

Run prefix: crc9_moe_regularized_r3_s17_20261009
Inference geometry, Top-2, two OEO stages and single Linear are unchanged. D4 augmentation and regularization are training-only. Baseline tests were previously seen; new tests are exploratory.
