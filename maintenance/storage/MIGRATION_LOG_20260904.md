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
- Two clean local `%TEMP%` worktrees and the one-turn publication worktree were
  unregistered and removed. The otherwise unreachable spatial warm-start
  commit chain is retained by Git tag
  `archive/local-worktree-spatial-warmstart-20260904`.
- The superseded 2026-09-03 temporal-16 lab ZIP was removed after the current
  balanced 2026-09-04 package and sidecar were confirmed present.

## LGVQ checkpoint pruning

- `all_run_index.csv` now covers 187 run directories plus two formal calibrated
  models.
- Seven run directories form the retained spatial-4 and temporal-9/16/36
  dependency closure. Their 124 checkpoint/snapshot files total
  1,760,646,908 bytes.
- 1,571 exploratory checkpoint files outside that closure were removed,
  releasing 17,672,923,796 bytes. Non-PT configuration, log, metric, and figure
  evidence remains in the original run directories.
- All indexed formal checkpoints passed post-prune SHA256 verification, and all
  20 temporal-9 phase-evolution snapshots remain available.

All Windows deletions in this pass were sent to the Recycle Bin.

## 2026-09-05 reconciliation and low-risk cleanup

- Reconciled local, guest3 server, and GitHub source histories at commit
  `1c3b0c16fa7c45645cbdb035bdac80901ef8e2c5`; both working trees have clean
  tracked source.
- Preserved pre-reconciliation tags and patch/untracked-file inventories before
  resetting either working tree.
- Deleted the user-approved `data/bench2drive` copy after confirming no active
  process: 16,828,527,478 bytes across 497,315 files. CARLA was already absent.
- Moved the 1.5 GiB P08 ImageNet `assets`, `logs`, and `runs` from its obsolete
  top-level experiment directory into the canonical FixedFeedbackSFT project.
  Only disposable bytecode caches and empty legacy directories were removed.
- Added a global read-only run index generator. Large-result pruning remains a
  separate, manifest-driven step.
