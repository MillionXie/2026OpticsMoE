# MoE regularization candidates

Selection uses validation only: mean of accuracy and macro recall. Only the winning candidate per architecture receives one exploratory test.

| Candidate | Weights | Fine-tune epoch | Val accuracy | Val macro | Train macro | Gap | Test accuracy | Test macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| optical | online | 11 | 81.48% | 76.31% | 82.41% | 6.10 pp | 79.32% | 73.62% |
| electronic | online | 11 | 81.34% | 76.35% | 94.07% | 17.72 pp | 80.08% | 75.35% |

Run prefix: crc9_moe_regularized_r2_s17_20261009
Inference geometry, Top-2, two OEO stages and single Linear are unchanged. D4 augmentation and regularization are training-only. Baseline tests were previously seen; new tests are exploratory.
