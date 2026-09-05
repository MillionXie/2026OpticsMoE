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
- Deleted 21,928,989,320 bytes of per-sample/per-layer `debug_examples` from
  the Optical MLP run. The formal `best.pt`, `last.pt`, metrics, training
  curves, phase masks, and reusable CLIP cache remain in place.
- Rehomed the remaining 1.5 GiB P08 ImageNet run directory from the obsolete
  top-level `experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone`
  path into its canonical owner at
  `FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone`.
  Its `assets`, `logs`, and `runs` were preserved; only Python bytecode caches
  and now-empty legacy directories were removed.
- Removed 351 redundant periodic `epoch_*`/`step_*` checkpoints recorded in
  `checkpoint_prune_manifest_20260905.csv`, releasing 16,473,945,705 bytes
  (15.343 GiB). A post-delete scan found zero remaining candidates. All 142
  distinct `best*` and `last`/`latest` checkpoints named by the manifest were
  verified present after deletion.
- After the approved cleanup and FA artifact rehome, the guest3 account uses
  955,505,769,798 bytes (889.884 GiB), including 925,221,741,005 bytes
  (861.680 GiB) under `2026OpticsMoE`. The shared volume reports about 800 GiB
  free. The largest retained areas are `data` at 653.235 GiB,
  `experiments` at 191.782 GiB, `FixedFeedbackSFT` at 11.190 GiB, and the user
  cache at 25.342 GiB.

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

`prune_redundant_checkpoints.py` is conservative by construction: it only
lists periodic `epoch_*`/`step_*` checkpoints when the same directory contains
both a recognizable `best` checkpoint and `last`/`latest`. It defaults to a
dry run and writes a manifest before `--apply` removes anything.

The corrected `all_run_index.csv` covers 528 actual run directories. After
cleanup, the indexed runs total 68.384 GiB, of which 48.698 GiB is checkpoints. The
remaining 100 runs classified as `checkpoint_without_clear_best` require manual
project-level review and were intentionally not modified.
