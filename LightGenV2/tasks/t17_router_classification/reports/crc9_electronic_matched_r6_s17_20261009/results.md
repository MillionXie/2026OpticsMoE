# MoE regularization candidates

Selection uses validation only: mean of accuracy and macro recall. Only the winning candidate per architecture receives one exploratory test.

| Candidate | Weights | Fine-tune epoch | Val accuracy | Val macro | Train macro | Gap | Test accuracy | Test macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| electronic_b32aug25 | online | 20 | 81.89% | 78.35% | 96.58% | 18.23 pp | not tested | not tested |
| electronic_b32aug50 | online | 9 | 82.31% | 79.57% | 96.28% | 16.71 pp | 80.57% | 77.56% |
| electronic_b64aug50 | online | 6 | 82.45% | 78.75% | 95.06% | 16.30 pp | not tested | not tested |

Run prefix: crc9_electronic_matched_r6_s17_20261009
Inference geometry, Top-2, two OEO stages and single Linear are unchanged. D4 augmentation and regularization are training-only. Baseline tests were previously seen; new tests are exploratory.
