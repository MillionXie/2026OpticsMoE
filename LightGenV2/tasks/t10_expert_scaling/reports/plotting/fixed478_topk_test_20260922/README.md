# Fixed-478 top-k ablation: held-out test results

All reported values below use the independent test split and three random seeds (17, 27, 37). The uncertainty is the sample standard deviation across seeds (`ddof=1`). Validation metrics are not mixed into this table.

| Dataset | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|
| Kather2016 | 62.50 ± 0.46 | 69.90 ± 0.73 | 71.01 ± 0.69 | 69.73 ± 0.28 |
| BloodMNIST | 81.01 ± 0.16 | 85.53 ± 0.34 | 86.27 ± 0.34 | 86.67 ± 0.20 |
| OrganCMNIST | 46.78 ± 1.62 | 58.73 ± 1.14 | 61.85 ± 0.65 | 61.38 ± 0.21 |

The entries are test accuracy in percent. `test_per_seed.csv` contains all 36 individual measurements and checkpoint/source hashes. `test_summary.csv` contains the means and sample standard deviations; the `evidence` directory preserves each immutable evaluation record.

Kather2016 uses the same N=4 checkpoints in the expert-count and fixed-478 presentations. Therefore, when both tables use the test split and the same three seeds, their N=4 rows are identical. BloodMNIST and OrganCMNIST were trained from caches containing only official train/validation arrays; test inference reads the untouched official MedMNIST source archive and verifies its SHA-256 against the training data manifest.
