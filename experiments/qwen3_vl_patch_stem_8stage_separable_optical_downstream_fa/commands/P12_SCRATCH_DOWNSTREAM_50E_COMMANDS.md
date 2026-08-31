# P12 Scratch-P11-body downstream control

This auxiliary control answers a narrower question than the main four-row P12
table: how much does the 90-epoch ImageNet P11 body pretraining contribute?
It retains the exact same frozen Qwen patch/position stem, P11 architecture,
task heads, split logic, 50-epoch head-only stage and 50-epoch adaptation stage.
It does **not** load the P11 ImageNet backbone checkpoint. The adapter, eight
phase planes, eight Slim Spatial Token Mixers and their gates are initialized
once from `P12_SCRATCH_INIT_SEED` (default `2026`).

The source checkpoint's file SHA-256 cannot be known before it is serialized.
For that reason there is deliberately no committed YAML containing a dummy
digest. `prepare` implements a safe two-stage config flow:

1. export a deterministic, fresh P11 body with provenance and semantic tensor
   digests;
2. inspect that artifact and render a runnable YAML containing its real file
   SHA-256 into the isolated run directory.

## 1. Prepare once

From a Git-locked server worktree:

```bash
cd /DATA/DATA1/guest3/2026OpticsMoE
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_scratch_downstream_50e.sh prepare
```

Default artifacts:

- source: `runs/p12_scratch_sources/p11_body_seed_2026/backbone.pt`;
- resolved config:
  `runs/p12_scratch_p11_body_seed_2026_50e/provenance/resolved_config.yaml`;
- results: `runs/p12_scratch_p11_body_seed_2026_50e/`.

Running `prepare` again is idempotent only if the semantic backbone digest,
stem SHA and init seed are identical. It refuses to overwrite a different
source.

## 2. Initial 12-job control

The default is three tasks × downstream seed `2026` × the four existing P12
keys (`noft`, `bp`, `fa_pretrained`, `fa_random`):

```bash
P12_SCRATCH_GPU_LIST=0,1,3,4,5 \
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_scratch_downstream_50e.sh launch
```

Interpretation is slightly different in this no-ImageNet control:

- `noft`: random P11 body + trained task head, so it is the random-feature
  probe;
- `bp`: exact BP adapts the fresh body for 50 epochs after the shared head-only
  endpoint;
- `fa_pretrained`: the code key is retained for a directly comparable four-row
  matrix, but the scientific label is **FA-source-init** because its fixed
  feedback phases are the fresh source phases, not ImageNet-pretrained phases;
- `fa_random`: independent deterministic random feedback.

Do not mix these output paths with the ImageNet-pretrained P12 runs.

## 3. Extend to three downstream seeds

After the seed-2026 control passes its gate, the same source can be expanded to
36 jobs without regenerating it:

```bash
P12_SCRATCH_GPU_LIST=0,1,3,4,5 \
P12_SCRATCH_SEEDS=2026,2027,2028 \
P12_SCRATCH_ADAPTATION_SEEDS=2026,2027,2028 \
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_scratch_downstream_50e.sh launch
```

The queue is dependency-aware and skips strictly complete jobs.

## 4. Monitor and summarize

```bash
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_scratch_downstream_50e.sh status
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_scratch_downstream_50e.sh tail
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_scratch_downstream_50e.sh summarize
```

The primary pretraining comparison is matched by task/method/downstream seed:
pretrained P12 versus this scratch source. The frozen Qwen stem means this is
accurately named **no ImageNet optical-body pretraining**, not “all weights
from scratch.”
