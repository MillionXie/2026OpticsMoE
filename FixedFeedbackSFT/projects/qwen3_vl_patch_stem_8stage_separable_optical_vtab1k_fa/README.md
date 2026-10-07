# P14: P11 optical backbone transfer on VTAB-1k

## Source entry after repository consolidation (2026-10-07)

The existing implementation is now retained on the repository's `main` line at
this path, with the original `experiments.*` module name unchanged. The historical
server run used commit `a2e6096fbe6ce43e6bbbc55b88f8ceb95aa4fd8b` in
`.worktrees/fa_vtab_20260914`. That code-only checkout was fully archived and
retired after checking process and runtime dependencies; its central run/data
paths remain untouched. All 4,303 original files, including historical timing,
were SHA-verified in the archive. Recovery location and SHA are recorded in
[`SERVER_WORKTREE_BALANCE`](../../../maintenance/storage/SERVER_WORKTREE_BALANCE_20261007.json).
Fourteen original files match that server checkout byte-for-byte.
The retained local launch log adds the original launch observations, and its shell
status command counts only the formal result depth rather than quarantine results;
neither difference changes the model or training. The four existing synthetic CPU
tests pass; this is source consolidation, not a new scientific run or result audit.
Historical commands below describe the original deployment and are not permission
to launch training during cleanup. Do not delete its source backbone, datasets,
run records, checkpoints or timing evidence.

This project extends P12 without altering its locked three-task implementation.
It asks whether the ImageNet-pretrained P11 optical backbone transfers under a
standardized 1,000-label protocol across natural, specialized and structured
vision domains.

## First-stage task panel

| VTAB category | Tasks |
|---|---|
| Natural | CIFAR-100, Flowers102 |
| Specialized | EuroSAT, PatchCamelyon |
| Structured | dSprites orientation, SmallNORB azimuth |

The data package supplies the standard `train800`, `val200`,
`train800val200`, and `test` file lists. The primary formal matrix uses the
1,000-example union for training and touches the test split only after the last
epoch. No task-specific augmentation or method-specific tuning is allowed.

Each task retains exactly four P12 methods: `noft`, `bp`, `fa_pretrained`, and
`fa_random`. NoFT trains a temporary classification head for 50 epochs. The
other methods inherit that exact checkpoint and receive 50 matched adaptation
epochs. The Qwen patch/position stem stays frozen; the reusable backbone still
contains 1,204,224 optical phases and 965,120 electronic parameters.

See `commands/P14_VTAB1K_SIX_TASK_COMMANDS.md` for the pinned paths, smoke test,
three-GPU launcher, status and summary commands.
