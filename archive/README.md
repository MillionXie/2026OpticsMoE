# Archive directory

This directory is for deliberate, non-active snapshots only. The everyday
project entrance is `../START_HERE.md`; new task development belongs under
`LightGenV2/tasks/`. Existing compatibility backends and active runtime worktrees
stay in place until their dependencies and experiment identities are audited.

Archive contents are ignored by Git. Every retained archive must have a short
README stating its source, date, purpose, and whether it can be regenerated.
Git-history snapshots should normally be represented by an `archive/*` tag
instead of a copied working tree.

The source-only staging retirements on 2026-10-06 are local Windows archives,
not new projects and not archives copied to every host. Including the retired
T12 metric export, verify their 20 sources and 20 retained original result files with
`python maintenance/git_safety/check_source_archives.py` on that machine.
The checker is read-only, rejects changed hashes and reoccupied restoration
paths, and does not restore files or approve deletions. Missing private archives
on a different host are not evidence that its source synchronization failed.

The 2026-10-04 sibling-directory consolidation preserves complete legacy
exports, including unique uncommitted files and all timing evidence. It is a
recoverable relocation, not approval to delete those exports or their data.
See `../maintenance/storage/EXTERNAL_PROJECTS_20261004.md` for locations and
verification receipts. Archives, weights and data must not be uploaded to Git.
