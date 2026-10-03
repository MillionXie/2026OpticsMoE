# 2026OpticsMoE storage policy

The repository itself is the only code root. Do not create project working
directories beside `2026OpticsMoE`.

## Canonical ownership

2026-10-02 clarification: the list below describes retained legacy project
ownership, not permission to create new LightGen experiments in `experiments/`.
Current LightGen work belongs to `LightGenV2/tasks/tNN_task_name/`, following
`LightGenV2/AI_RULES.md` and root `AGENTS.md`. Its release directory is the task's
`releases/`; do not move valid existing runs or packages merely to fix naming.
Existing validated server/lab runtime trees are protected during reconciliation.
Do not create more branches/worktrees without explicit user authorization.

- `experiments/<project>/`: Qwen, LGVQ, D2NN, hardware, and dataset-specific
  code. Put run products in that project's `runs/` and durable reports in its
  `artifacts/`.
- `FixedFeedbackSFT/`: FA/feedback projects and their `projects/`, `runs/`, and
  `evidence/`.
- `opticalmoe/`: reusable OpticalMoE package and its own development results.
- `opticalmoe_experiments/`: retained legacy OpticalMoE experiments.
- `archive/`: deliberate inactive snapshots only; see `archive/README.md`.

The repository root must not contain `runs/`, `server_projects/`, extracted
ZIP projects, or SSH transfer packages.

## Generated and transferable data

- `.codex_transfer/` is temporary staging, not durable storage. Move useful
  evidence into the owning project's `artifacts/` before clearing it.
- `.bundle`, `.tar`, `.tar.gz`, `.tgz`, and generated ZIP files are delivery or
  transport artifacts. Keep final lab ZIPs under the owning project's
  `lab_bundles/`; remove obsolete copies after recording their SHA256.
- Large checkpoints stay under the run that produced them. Formal dependency
  checkpoints and requested mask-evolution snapshots are retained; redundant
  exploratory checkpoints may be pruned only after a dependency-closure audit.

## Protected measurements and baselines

2026-10-03 user clarification: retain historical timing, throughput, power,
energy and efficiency evidence together with the corresponding baselines.
This includes code/config/environment/commit and checkpoint identity, original
metrics and predictions, per-call timing, raw power telemetry, hardware and
measurement scope, calculation scripts, and historical scope corrections.
Superseded or rejected measurements remain labelled audit evidence, not
disposable trial files. Never reuse an old model's measurements as a new model's.
The previously approved baseline-feature-cache cleanup is not authorization to
delete baseline models, original data, reports or measurement evidence.
See [retention contract](MEASUREMENT_BASELINE_RETENTION.md) and its bounded
read-only inventory; unlisted task-specific/server evidence is protected too.

## Indexes

Run the following from the repository root before and after a cleanup:

```powershell
python maintenance/storage/build_storage_indexes.py --root .
```

It writes `storage_inventory.csv`, `transfer_inventory.csv`, and
`bundle_index.csv` beside this document. The indexes contain sizes, timestamps,
and SHA256 values where appropriate, so later users and AI agents can identify
what was moved or removed.

For three-location source reconciliation, also run:

```powershell
python maintenance/storage/build_source_project_index.py
```

The resulting `source_project_index.csv` is the current Git-tracked source
contract. Runtime artifacts are indexed separately in `all_run_index.csv`.
The historical `experiment_sync_audit_pre_reconciliation_20260905.csv` records
local/server/GitHub presence and direct inter-experiment Python imports before
the 2026-09-05 reconciliation; it must not be used as a current sync report.
Follow `SYNC_RECONCILIATION_PLAN_20260905.md` before resetting either working
tree or removing an apparently old experiment directory.
