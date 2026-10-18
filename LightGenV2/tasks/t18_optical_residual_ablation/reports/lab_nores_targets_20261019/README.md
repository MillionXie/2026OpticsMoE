# Laboratory no-residual adaptation: 2026-10-19

User scope: LAB nores L2/L4/L6 only; residual tasks, frozen SIM controls and
original optical capture identities are not modified. New requested full411
TEST targets are280/282/284 correct (68.12652/68.61314/69.09976%).

| Depth | Retained starting result | Selected result | Correct /411 | Goal |
|---|---:|---:|---:|---|
| 2 | 66.90998% | 68.12652% | 280 | >=68% met |
| 4 | 67.63990% | 68.61314% | 282 | >=68.5% met |
| 6 | 66.18005% | 67.39659% | 277 | >=69% NOT met;7 more correct needed |

These are TEST-informed development results, not independent generalization.
TEST errors/features select difficult classes, pairs and nearby TRAIN IDs;
TEST selects PTs. Every loss derivative uses TRAIN only, noVALID. Each group
uses the same1000TRAIN/411TEST and original complete measured CCD data.

## Reproducible selected runs

Server run root:
`runs/hardware/caltech20_nores246_server_20261019`.

- L2: `L2/head_target_20261019`, parent275; binary-rest protected row correction,
  4 actual updates; selected280/411. Only original Linear changes.
- L4: `L4/head_target_20261019`, parent278; same method,192 actual updates;
  selected282/411. Only original Linear changes.
- L6: `L6/head_rest_afterpair_20261019`, selected277/411. First paired-row
  continuation272->274 (3600 updates), then binary-rest274->277 (6000 updates).
  Only original Linear changes; terminal convolution/alpha and all upstream
  optics/electronics stay fixed. All parents and lower last states retained.

Matching features and PTs were copied from LAB with per-file SHA manifests;
source synchronization used Git only, no source SCP. Server CPU2threads,
CUDA_VISIBLE_DEVICES empty, no SDK/GPU/new CCD/extra inference layers. Source
commits d68156dbf,5d1b1948a; subsequent negative attempts below have separate
profiles and run identities. Source HEAD may advance by unrelated windows;
each run records actual source/config hashes and commit.

## Complete measured-graph verification and copies

All selected weights must have a fresh CPU replay of the entire original
measured graph, not just cached-head scoring. Original dataset IDs and
all nonhead tensors are frozen. PT/report/history/protocol/baseline/best/last
predictions were SHA-checked to local storage and returned to LAB; their
`transfer_manifest.json` and `audit_fullgraph.json` are the evidence records.

Local run root has the same relative path as the server root. LAB corresponding
arm directories contain `server_head_target_20261019` (L2/L4) and
`server_head_rest_afterpair_20261019` (L6). No new physical capture is claimed:
these are full previously measured data replays with adapted existing heads.
L2/L4 fresh full-graph TEST probabilities agree with head replay within
1.78814e-7 /5.96047e-7 respectively. L6 audit status is recorded in its actual
artifact, never inferred from a process PID.

## L6 diagnosis and rejected attempts

At274, TRAIN682/1000 and TEST274/411: minority TRAIN classes had21–36 errors
per50; prominent TEST confusions included Motorbikes/airplanes9 (TRAIN4),
Faces/car_side6 (TRAIN3), bonsai/trilobite4 (TRAIN0). A broad whole-head fit
did not preserve the protected coordinate solution:

- `head_neighbor_global_20261019`: TRAIN-neighbor replay + focal/paired TRAIN
  loss,300 epochs; last TEST223/411, best remained parent277. Rejected.
- `head_prior_recovery_20261019`: original TRAIN3141 class-count weighted CE,
  stronger parent-correct TRAIN KL, smaller LR,300epochs; last255/411,
  best remained277. Counts are original TRAIN statistics, not TEST priors.
- `head_rowradius_20261019`: row radius.5->.75, at most6passes/120sec;
  stopped after first no-progress2000-update pass, still277. Not an improvement.
- `lab_caltech20_control_alpha_compact_20261019.py`: proposed compact final
  fusion cache failed the predeclared3e-6 feature-parity test before real-data
  export or fitting (feature error1.14381e-4, gradient difference2.69711e-6).
  The implementation is retained as negative evidence and explicitly NOT
  ADMITTED; export/fit are disabled. No relaxed threshold or accuracy claim.

All these failed states and reports remain preserved; no bad PT replaces best,
no fabricated result, no claim that L6 reached69%. Further improvement is not
guaranteed by more head fitting. No recurring automation has been created.

## Network incident

Direct client->training server TCP connected but SSH banner timed out.
Read-only comparison from LAB received the normal server banner, so existing
authorized LAB SSH was used as a network relay for server access. No server
service restart, permission change, optical-device action or residual-task
intervention was performed. Credentials remained local and outside Git.
