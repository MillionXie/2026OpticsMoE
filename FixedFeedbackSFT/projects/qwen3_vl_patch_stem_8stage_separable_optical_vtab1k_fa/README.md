# P14: P11 optical backbone transfer on VTAB-1k

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
