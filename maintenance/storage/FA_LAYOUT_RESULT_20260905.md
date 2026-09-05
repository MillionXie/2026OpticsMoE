# Fixed-feedback project layout correction — 2026-09-05

## Canonical contract

- Source code: `FixedFeedbackSFT/projects/<project>/`
- Runtime artifacts: `FixedFeedbackSFT/runs/<project>/<run>/`
- Stable Python interface: `python -m experiments.<project> ...`
- Selected formal evidence: `FixedFeedbackSFT/evidence/`

`experiments/__init__.py` extends the `experiments` namespace with
`FixedFeedbackSFT/projects`, so keeping the historical Python module name does
not require a second physical source tree.

## Corrected server state

Seven FA projects had 11,245,085,851 bytes (1,131 files) of ignored runtime
artifacts left under either `experiments/<project>/` or a source project's local
`runs/` directory. They were moved on the same filesystem to the central
`FixedFeedbackSFT/runs/<project>/` layout. File counts and byte totals matched
before and after every move; no checkpoint was deleted or rewritten. The exact
mapping is recorded in `fixed_feedback_artifact_rehome_20260905.csv`.

The obsolete `experiments/<FA project>` containers held no source files. After
their runtime artifacts moved, their remaining Python bytecode and pytest cache
directories were removed. The required P08 static-stem asset remains at
`FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/assets/`.

## BDD discrepancy

Only these two BDD projects contain versioned source and therefore appear on
both local and server clones:

- `qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual`
- `qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16`

The six additional BDD names seen locally contained zero source files: only
empty directories and stale `__pycache__/*.pyc`. Their bytecode files were
removed. Windows OneDrive ACLs may leave zero-file placeholder directories in
the local checkout; these are neither projects nor Git content.

Current source identity is recorded in `source_project_index.csv`. That file is
built only from Git blobs. `all_run_index.csv` separately describes runtime
artifacts, so an ignored server run can no longer be mistaken for server-only
source, and an empty local folder can no longer be mistaken for a local-only
project.

## Verification

- Physical layout regression: 3 passed.
- Historical `experiments.<project>` import compatibility: all 9 FA projects.
- All FA shell launchers: `bash -n` passed.
- Complete FA test collection with importlib mode: 254 passed.
- Git-tracked source remains synchronized through Git; server-only runs remain
  synchronized by the committed inventory and explicit artifact paths.
