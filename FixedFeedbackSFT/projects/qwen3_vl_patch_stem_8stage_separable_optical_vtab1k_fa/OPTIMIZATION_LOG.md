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
