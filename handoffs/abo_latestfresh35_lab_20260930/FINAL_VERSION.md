# ABO final checkpoint — user finalized 2026-10-02

User finalized rank72 lightweight RGB late-branch epoch13 checkpoint. Do not replace it with earlier rank192 or internal-alpha39 experiments.

- Weight: `rank72_epoch13_best_snapshot.pt`.
- SHA256: `25f23260864a00a0f32b5de209c27f5876166018c8d8ab0e880de81e0d2c2c22`.
- RGB branch parameters: 92,168. Optical blocks retain alpha approximately .445–.448.
- Simulation R@1 .83375; same-weight no-light diagnostic .8000.
- Untuned physical R@1 .81125, R@5 .94125, R@10 .9700, MRR .8694430762577247.
- Physical six-stage capture: 1600 gallery + 800 queries ×6 =14,400 valid CCD, receipts retained; report `rank72_physical_full_report_20261001.json`.
- The original800 queries were used during development; not an independently unseen generalization estimate.
- No further ABO training, adaptation or capture authorized by this finalization. External upload remains paused; package locally if needed, upload only after renewed user instruction.
