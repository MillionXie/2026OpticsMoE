# Qwen3-VL LGVQ baseline results

Only `Linear(2048,1)` is trainable. Qwen3-VL is frozen. Spatial and temporal are selected by prompt and reported separately.

| Frames | Target | SRCC | KRCC | PLCC | RMSE | MAE |
|---:|---|---:|---:|---:|---:|---:|
| 4 | spatial | 0.6440 | 0.4632 | 0.6806 | 8.3308 | 6.6750 |
| 4 | temporal | 0.7440 | 0.5487 | 0.7549 | 9.1132 | 7.4183 |
| 9 | spatial | 0.6420 | 0.4634 | 0.6774 | 8.3559 | 6.6814 |
| 9 | temporal | 0.7514 | 0.5546 | 0.7586 | 8.9736 | 7.2765 |
| 16 | spatial | 0.6470 | 0.4675 | 0.6853 | 8.2603 | 6.6117 |
| 16 | temporal | 0.7636 | 0.5635 | 0.7696 | 8.7918 | 7.1245 |
| 25 | spatial | 0.6474 | 0.4661 | 0.6859 | 8.2520 | 6.6158 |
| 25 | temporal | 0.7732 | 0.5725 | 0.7761 | 8.6565 | 6.9583 |
| 36 | spatial | 0.6614 | 0.4790 | 0.7018 | 8.0792 | 6.4474 |
| 36 | temporal | 0.7820 | 0.5792 | 0.7842 | 8.5129 | 6.8920 |
| 49 | spatial | 0.6642 | 0.4801 | 0.6975 | 8.1278 | 6.5256 |
| 49 | temporal | 0.7836 | 0.5823 | 0.7829 | 8.5311 | 6.8831 |

| Frames | Qwen ms / prompt-video | Pipeline ms / video for two prompts | Peak GPU GiB | Head train seconds | Head ms / prompt-video |
|---:|---:|---:|---:|---:|---:|
| 4 | 21.276 | 70.943 | 5.220 | 2.091 | 0.000020 |
| 9 | 54.214 | 157.689 | 7.087 | 2.061 | 0.000022 |
| 16 | 84.627 | 233.420 | 6.460 | 1.958 | 0.000020 |
| 25 | 136.088 | 398.999 | 5.997 | 2.178 | 0.000020 |
| 36 | 194.280 | 522.308 | 6.769 | 2.134 | 0.000020 |
| 49 | 269.969 | 804.858 | 5.919 | 2.141 | 0.000022 |

Checkpoint selection uses the periodically observed fixed test split, with no validation split, as explicitly requested for this project.
