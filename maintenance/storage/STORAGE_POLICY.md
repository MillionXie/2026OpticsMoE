# 2026OpticsMoE storage policy

The repository itself is the only code root. Do not create project working
directories beside `2026OpticsMoE`.

## Canonical ownership

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
python maintenance/storage/build_sync_audit.py
```

The resulting `experiment_sync_audit_20260905.csv` records local/server/GitHub
presence and direct inter-experiment Python imports. Follow
`SYNC_RECONCILIATION_PLAN_20260905.md` before resetting either working tree or
removing an apparently old experiment directory.
