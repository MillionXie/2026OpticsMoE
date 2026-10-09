# MoE regularization candidates

Selection uses validation only: mean of accuracy and macro recall. Only the winning candidate per architecture receives one exploratory test.

| Candidate | Weights | Fine-tune epoch | Val accuracy | Val macro | Train macro | Gap | Test accuracy | Test macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| optical_control | parent | 0 | 82.87% | 79.01% | 86.02% | 7.01 pp | not tested | not tested |
| optical_balance005 | parent | 0 | 82.87% | 79.01% | 86.02% | 7.01 pp | not tested | not tested |
| optical_balance02 | parent | 0 | 82.87% | 79.01% | 86.02% | 7.01 pp | not tested | not tested |

Run prefix: crc9_moe_regularized_r5_s17_20261009
Inference geometry, Top-2, two OEO stages and single Linear are unchanged. D4 augmentation and regularization are training-only. Baseline tests were previously seen; new tests are exploratory.
