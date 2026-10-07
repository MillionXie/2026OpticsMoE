# Independent verification

`server_cuda_full_current_20260923.json` was generated from
`LGVQ_Temporal_08044_teacher_final_v2_20260923.zip`, not from the parent
checkout.

- host runtime: Python 3.11.15
- PyTorch: 2.6.0+cu124
- GPU used: NVIDIA GeForce RTX 4090, driver 550.144.03
- command: `python -I simulate.py --device cuda --fields 0 --output runs/server_gpu_full.json`
- fields / valid videos: 35 / 558
- status: passed
- all five metric deltas from packaged reference: 0.0
- result file SHA-256:
  `fe03f2d3a62bad214876b14804332e3b87c85b93a82728bdbcbd6647dd7d4198`

This check verifies fixed-weight numerical reproduction.  It does not claim
that a fresh training run will select the same epoch or be bit-identical.
