# Three-location reconciliation result — 2026-09-05

## Source of truth

- GitHub `main`, local `main`, and the guest3 server `main` are aligned.
- Reconciled source commit: `1c3b0c16fa7c45645cbdb035bdac80901ef8e2c5`.
- Local and server pre-reconciliation states remain recoverable through the
  `archive/local-pre-reconcile-20260905` and
  `archive/server-pre-reconcile-20260905` tags and ignored patch archives.
- The merge passed syntax compilation for 103 changed Python files and 352
  independent tests. OpenMoji tests used the server's existing pinned icon
  assets and local Hugging Face cache.

## What was reconciled

- Preserved the active Caltech/MNIST laboratory workflow, LGVQ spatial and
  temporal hardware handoffs, attention-free LGVQ implementation, OpenMoji and
  synthetic editing sources, Caltech multiplane comparison, and FA/ImageNet
  recovery projects.
- Compared server working-tree candidates against the reconciled branch while
  ignoring CRLF/LF differences. Only four OpenMoji report documents and a few
  small experiment metadata/configuration files were genuinely missing; they
  were added before publication.
- Restored the server's deleted tracked documentation rather than treating an
  old dirty worktree as authoritative.
- Added ignore rules for machine-local runs, caches, calibration outputs,
  hardware sessions, reports, evidence, and delivery bundles. Tracked curated
  files are unaffected by these rules.

## Storage operations

- CARLA was already absent when checked.
- Deleted the explicitly approved `data/bench2drive` dataset after verifying
  there was no active process: 16,828,527,478 bytes and 497,315 files.
- Rehomed the remaining 1.5 GiB P08 ImageNet run directory from the obsolete
  top-level `experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone`
  path into its canonical owner at
  `FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone`.
  Its `assets`, `logs`, and `runs` were preserved; only Python bytecode caches
  and now-empty legacy directories were removed.

## Artifact policy

GitHub is the source of truth for code, tests, configs, documentation, and
small manifests. The guest3 server is authoritative for datasets, reusable
feature caches, formal runs, and checkpoints. Local Windows storage is for
hardware configuration, CCD captures, previews, and currently needed delivery
ZIPs. Large artifacts are synchronized by manifest/SHA256, not by mirroring the
whole server onto GitHub or the local machine.

`build_run_index.py` generates a read-only global `all_run_index.csv` with run
sizes, checkpoint counts, recognizable best checkpoints, metric/config
presence, and a conservative retention recommendation. The index is evidence
for later pruning; it is not itself authorization to delete a run.
