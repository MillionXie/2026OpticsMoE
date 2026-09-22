# D2NN inference-only cross-task matrix (2026-09-22)

No parameter is optimized. Each row uses the independently trained D2NN optical backbone for that source task; each column plugs in the already-trained target `Linear(784, C)` and evaluates the complete target test split.

| Source optics / target test | EuroSAT | CLEVR | Speech | Physical |
|---|---:|---:|---:|---:|
| eurosat | 80.72% | 50.14% | 35.78% | 28.75% |
| clevr | 19.75% | 76.76% | 24.28% | 10.04% |
| speech | 47.11% | 49.98% | 74.33% | 27.54% |
| physical | 27.60% | 50.10% | 29.58% | 79.30% |

The off-diagonal cells measure direct plug-and-play compatibility. They are not the sequential forgetting matrix. Per-cell class confusion matrices are stored under `details.<source>.details.<target>.confusion_matrix` in `matrix.json`.
