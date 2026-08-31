# P12 phase-only adaptation panel

This panel is a strict companion to the existing joint-adaptation P12 runs. It does not alter their defaults or output tree.

## Four groups

| Code | Optical connector | Trainable parameters |
|---|---|---|
| `noft` | none | task head only |
| `bp` | current exact optical BP | 8 phase planes + task head |
| `fa_pretrained` | fixed P11 source/pretrained phase connector | 8 phase planes + task head |
| `fa_random` | fixed seed-controlled random phase connector | 8 phase planes + task head |

For all three adapting groups, the Qwen stem, input adapter, every mixer, normalization layer, fusion/residual gate, and other electronic-backbone tensor are frozen and excluded from the optimizer. Frozen electronic modules stay in evaluation mode so mixer dropout cannot silently change the fixed backbone. The task head always receives its ordinary exact gradient; FA is used only for the inter-stage optical connector.

## Static test (no training)

```bash
python -m pytest -q experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/tests/test_phase_only.py
bash -n experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_phase_only_fa_50e.sh
```

## One-batch CUDA smoke (isolated output)

```bash
CUDA_VISIBLE_DEVICES=2 \
P12_PHASE_ONLY_TASK=caltech101 \
P12_PHASE_ONLY_METHOD=bp \
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_phase_only_fa_50e.sh smoke
```

## Formal multi-GPU launch

```bash
P12_PHASE_ONLY_REPO_ROOT=/DATA/DATA1/guest3/2026OpticsMoE_p12_phase_only_e305 \
P12_PHASE_ONLY_GPU_LIST=0,1,3,4,5 \
P12_PHASE_ONLY_SEEDS=2026,2027,2028 \
P12_PHASE_ONLY_ADAPTATION_SEEDS=2026,2027,2028 \
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_phase_only_fa_50e.sh launch
```

The queue first produces the panel's own head-only/common start for every task and seed, then launches BP, FA-source/pretrained and FA-random from the byte-identical common start.

## ISIC numeric-audit retry (2026-09-01)

The first ISIC adaptation attempts stopped before epoch 1 because a bitwise
head-gradient repeat check rejected ordinary CUDA reduction noise. After the
audited tolerance fix, restart only the three missing adaptation methods (and
never the completed NoFT/Caltech/LSP results) with:

```bash
P12_PHASE_ONLY_ISIC_RETRY_GPUS=0,3,5 \
  bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_phase_only_isic_numeric_retry.sh launch
```

Status and logs:

```bash
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_phase_only_isic_numeric_retry.sh status
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_phase_only_isic_numeric_retry.sh tail
```

This launcher refuses to overwrite an existing `result.json`. Each method has
its own PID and `logs/retry_after_numeric_fix.log`.

The final head-gradient audit requires, per tensor, maximum absolute difference
`<=5e-4`, relative L2 difference `<=1e-3`, and cosine `>=0.99999`. The absolute
limit was calibrated after an epoch-9 CUDA repeat produced `1.715e-4` absolute
difference while relative L2 was `2.164e-4` and cosine was `0.99999999`. A
single allow-listed prior panel digest permits the epoch-9 checkpoint to resume;
all other checkpoint identity fields remain exact.

## Identity and e305 compatibility

`phase_only.py`, its queue/smoke wrappers, this config and these tests are add-on files. They are intentionally not added to `settings.py::IMPLEMENTATION_FILES`, so the locked e305 P12 implementation digest remains unchanged and its strict common-start loader stays usable. This does **not** hide the new behavior:

- the resolved settings include `phase_only_panel`, the panel-config hash and `panel_implementation_sha256`;
- therefore every phase-only checkpoint receives a distinct `config_digest`;
- `phase_only_panel.json` records parameter names/counts, trainable counts and optimizer groups;
- `result.json` includes the full phase-only identity;
- output is isolated under `runs/p12_phase_only_fa_50e`.

The 2026-09-01 retry permits exactly one historical base digest and one panel
digest. The only base diff adds optional source-provenance return fields; it
does not change tensor loading, forward, loss, or optimization. Both the old
NoFT result and its common-start checkpoint must still match task, seed,
dataset manifest, source checkpoint, completed epoch count, and the exact
allow-listed digests. No general implementation-hash bypass is used.
