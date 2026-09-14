# P14 construction and experiment log

## 2026-09-14: standardized VTAB-1k transfer scaffold

- Added six tasks spanning the three official VTAB families.
- Consumes the published `train800/val200/train800val200/test` manifests rather
  than resampling examples locally.
- Uses deterministic bicubic 224x224 preprocessing and the CLIP normalization
  required by the frozen Qwen patch/position stem. No augmentation is used, so
  orientation/azimuth labels cannot be silently changed by flips or rotations.
- Keeps the existing P11 source SHA, architecture, 1,204,224 optical phases,
  optical gate floor, lightweight electronic body and four P12 feedback groups.
- Locks batch size 64, 50 head-only epochs and 50 adaptation epochs. Updating
  methods load the exact task/seed NoFT checkpoint before configuring feedback.
- Test data are evaluated only once after the fixed final epoch; the initial
  seed-2026 matrix is completed before expanding to seeds 2027 and 2028.
- Added atomic result/checkpoint writes, exact resume state, dataset/source
  identities, phase-motion reporting, summary output, safe selected-task archive
  extraction and a three-GPU launcher that terminates after the queue completes.
- Verified the selected public archive contains the canonical 1,000-example
  union with label spaces 100/102/10/2/16/18 for the six tasks. The downloader
  now rejects any archive except SHA-256
  `dca579dac0ac3d285ec8d898033795f2eb55fe83cbd8d0c10b074455a9d2aa1c`,
  refuses to overwrite an unverified non-empty data directory, and makes an
  already verified extraction idempotent.
- Added physical-GPU provenance to every formal result; the launcher records
  the lane-to-GPU mapping and invokes itself through an absolute script path.

Pending entries in this file must record the archive SHA-256, local/server test
results, launch commit, physical GPU ids, process ids, completion counts and the
first six-task result table. Intermediate accuracy must not be reported as a
completed VTAB result.

## 2026-09-14: server qualification

- Git implementation commit: `a909bbb6f43a55983e984412a2786fcfcef18a73`.
- Server worktree: `/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/fa_vtab_20260914`.
- Extracted 152,219 files for the six selected tasks. Server and local archive
  SHA-256 both matched the pinned digest above.
- Server Python tests: `4 passed`; shell syntax and public `experiments.*` CLI
  import passed.
- CUDA smoke on physical GPU 5 completed one NoFT epoch and one BP epoch over
  eight samples. BP phase mean absolute displacement was `0.00123 rad`, proving
  that the optical phase path received an update.
- The first smoke exposed an `EXIT` trap referencing a function-local temporary
  path under `set -u`. Training itself completed, but the cleanup produced exit
  code 1. The trap now captures the resolved `mktemp` path at registration time;
  a clean exit is required before formal launch.
- The corrected smoke completed NoFT and BP and exited with code 0 on physical
  GPU 1. A deferred three-lane launcher was added because other users occupied
  the initially selected cards between data preparation and launch. Each
  supervisor waits without CUDA until its assigned card has no compute PID,
  then runs its two-task lane and exits; failed lanes receive an explicit marker.
- The first deferred invocation created no CUDA process because the tracked
  shell file has mode `100644` and its background self-invocation therefore
  returned `Permission denied`. All self-invocations now call `bash` explicitly,
  so correctness no longer depends on the executable bit. Stopped PID files are
  checked before the clean relaunch.
