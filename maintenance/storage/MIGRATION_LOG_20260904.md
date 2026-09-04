# Storage migration log — 2026-09-04

This log records the first low-risk cleanup pass. Original file-level SHA256
values are in `transfer_inventory_precleanup_20260904.csv` and
`bundle_index_precleanup_20260904.csv`; the unsuffixed indexes describe the
post-cleanup state.

## Rehomed source project

- `server_projects/qwen3vl_lgvq_linear_baseline` moved to
  `experiments/qwen3_vl_2b_lgvq_linear_baseline`.
- The temporal frame-count timing import and active manifest paths were updated
  to the canonical location.
- Historical AutoDL paths in `EXPERIMENT_RECORD.md` were intentionally kept as
  provenance, not as active commands.

## Durable local-only evidence

- Caltech CCD recovery, theoretical feature contact sheets, and overview images
  moved into the early-robust Caltech project's `artifacts/local_only/`.
- LGVQ temporal 16/36-frame QA, spatial router figure, and mask-export log moved
  into the LGVQ single-metric project's `artifacts/local_only/`.
- MNIST paper figures and representative amplitude/phase images moved into the
  MNIST v2 project's `artifacts/local_only/`.
- LUT previews, geometry QA, and one-off 3500-us LUT helper scripts moved into
  `experiments/hardware_sdk/artifacts/calibration/local_only/`.
- The 2026-08-24 legacy 5090 timing scripts/report moved into the temporal
  frame-count timing project's `artifacts/local_only/`.
- The final 2026-08-30 Qwen simulation/hardware handoff ZIPs moved to
  `experiments/qwen_optical_platform_handoff/lab_bundles/20260830/`.
- Two old LGVQ source snapshots moved to
  `archive/legacy_code_snapshots/`; they are inactive and ignored by Git.

## Removed after indexing

- Root `runs/_smoke/qwen_bdd_bench2drive`: synthetic smoke output for the
  already removed Bench2Drive project.
- Extracted package-verification trees, obsolete package candidates, patches,
  transport archives, and Git bundles under `.codex_transfer/`.
- Root and `.git` bundle files whose heads were verified reachable or
  patch-equivalent to `origin/main` before deletion.
- Root `.pytest_cache`, `.tmp_results`, `__pycache__`, and `cache/_smoke`.

All Windows deletions in this pass were sent to the Recycle Bin.
