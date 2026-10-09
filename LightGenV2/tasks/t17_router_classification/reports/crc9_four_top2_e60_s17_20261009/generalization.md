# Sixty-epoch continuation: generalization diagnostics

All metrics below use fixed weights on full train/validation splits. Online batch training loss is not used to estimate the gap.

| Model | Selected epoch | Selected train / val macro recall | Epoch60 train / val macro recall | Epoch60 gap | Validation peak to epoch60 drop |
|---|---:|---:|---:|---:|---:|
| optical | 51 | 97.56% / 76.31% | 85.47% / 67.34% | 18.13 pp | 8.97 pp |
| electronic | 38 | 87.00% / 75.37% | 87.33% / 70.50% | 16.83 pp | 4.87 pp |
| d2nn | 55 | 91.65% / 74.72% | 83.06% / 66.69% | 16.37 pp | 8.03 pp |

Training and validation gaps are descriptive evidence. A growing gap plus declining validation after the peak supports late overfitting; a gap alone does not distinguish overfitting from distribution differences.
Validation oscillations also need to be considered; the current experiment keeps the original constant learning rates.

Run prefix: crc9_four_top2_e60_s17_20261009
