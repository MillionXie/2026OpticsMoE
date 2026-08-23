# P11 implementation and verification log

## 2026-08-24 implementation

- Created an independent experiment without modifying or stopping active P09.
- Implemented axis-selective angular-spectrum propagation: token/y-only and
  channel/x-only operators whose orthogonal axes are mathematically relayed.
- Organized eight stages as four token-to-channel optical macro blocks.
- Added exact Qwen block-major to row-major permutation only around token-axis
  optics and an exact inverse before canonical electronic processing.
- Kept phase/random/feedback tensors in physical layout so the inherited BP,
  fixed-feedback and deployment-perturbation paths remain consistent.
- Reused P09's initialized residual gate and width-96 electronic mixer modules;
  no trainable parameters were added.
- Added a unique P11 architecture signature, explicit layout/schedule metadata
  and cross-variant strict-load rejection.
- Prepared a P09-matched formal config and an explicit-GPU guarded launcher. It
  was not launched.

## Verification

- Python bytecode compilation: passed locally.
- Unit tests: pending server synchronization.
- Low-utilization GPU ImageNet forward/backward smoke: pending unit tests.
- Formal training: intentionally not started.
