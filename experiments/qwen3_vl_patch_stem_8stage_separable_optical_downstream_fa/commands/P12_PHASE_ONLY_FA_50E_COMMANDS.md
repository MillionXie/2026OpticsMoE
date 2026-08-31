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

## Formal multi-GPU launch (do not run until approved)

```bash
P12_PHASE_ONLY_REPO_ROOT=/DATA/DATA1/guest3/2026OpticsMoE_p12_phase_only_e305 \
P12_PHASE_ONLY_GPU_LIST=0,1,3,4,5 \
P12_PHASE_ONLY_SEEDS=2026,2027,2028 \
P12_PHASE_ONLY_ADAPTATION_SEEDS=2026,2027,2028 \
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_phase_only_fa_50e.sh launch
```

The queue first produces the panel's own head-only/common start for every task and seed, then launches BP, FA-source/pretrained and FA-random from the byte-identical common start.

## Identity and e305 compatibility

`phase_only.py`, its queue/smoke wrappers, this config and these tests are add-on files. They are intentionally not added to `settings.py::IMPLEMENTATION_FILES`, so the locked e305 P12 implementation digest remains unchanged and its strict common-start loader stays usable. This does **not** hide the new behavior:

- the resolved settings include `phase_only_panel`, the panel-config hash and `panel_implementation_sha256`;
- therefore every phase-only checkpoint receives a distinct `config_digest`;
- `phase_only_panel.json` records parameter names/counts, trainable counts and optimizer groups;
- `result.json` includes the full phase-only identity;
- output is isolated under `runs/p12_phase_only_fa_50e`.

For a formal run, create a worktree directly from locked commit `e305e0b`, apply only these new add-on files, and verify `implementation_sha256()` equals the completed P12 artifact before launching. Do not run this panel from a worktree where any file in `IMPLEMENTATION_FILES` differs from e305.
