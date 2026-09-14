# P14 VTAB-1k six-task transfer commands

P14 evaluates the frozen P11 ImageNet backbone on six standard VTAB-1k tasks.
Every task uses the provided `train800val200` list for the final 1,000-example
fit and evaluates the provided `test` list once. The four methods are NoFT,
exact BP, FA-source and FA-random. The first matrix uses seed 2026; seeds 2027
and 2028 are launched only after the first matrix passes its result audit.

## Required server paths

```bash
export P14_REPO_ROOT=/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/fa_vtab_20260914
export P14_DATA_ROOT=/DATA/DATA1/guest3/2026OpticsMoE/data/vtab-1k
export P14_SOURCE_BACKBONE=/DATA/DATA1/guest3/2026OpticsMoE/FixedFeedbackSFT/runs/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/p11_imagenet1k_pretrain_bs96_90e/checkpoints/backbone.pt
export P14_STEM_CHECKPOINT=/DATA/DATA1/guest3/2026OpticsMoE/FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/assets/qwen3_vl_static_stem_224.pt
export P14_OUTPUT_ROOT=/DATA/DATA1/guest3/2026OpticsMoE/FixedFeedbackSFT/runs/qwen3_vl_patch_stem_8stage_separable_optical_vtab1k_fa/p14_vtab1k_six_task_50e
export P14_GPU_LIST=1,3,4
export P14_SEED=2026
```

## Data extraction

The selected public archive is pinned to Hugging Face dataset revision
`102b19d8a1e23c47fb3941e736ca8b1d49d2552c`. Extract only the six registered
tasks and write a provenance manifest. Its required SHA-256 is
`dca579dac0ac3d285ec8d898033795f2eb55fe83cbd8d0c10b074455a9d2aa1c`;
the preparation command fails closed on a different file:

```bash
cd "$P14_REPO_ROOT"
/home/guest3/miniconda3/envs/xml/bin/python -m \
  experiments.qwen3_vl_patch_stem_8stage_separable_optical_vtab1k_fa.prepare_data \
  --archive /DATA/DATA1/guest3/2026OpticsMoE/data/_archives/vtab-1k-combined.tar.gz \
  --output-root "$P14_DATA_ROOT"
```

## Test, launch and audit

```bash
cd "$P14_REPO_ROOT"
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_vtab1k_fa/commands/p14_vtab1k_six_task_50e.sh smoke
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_vtab1k_fa/commands/p14_vtab1k_six_task_50e.sh launch
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_vtab1k_fa/commands/p14_vtab1k_six_task_50e.sh status
```

The launcher requires exactly three idle GPU ids and refuses to overlap an
existing compute process. It assigns two datasets to each GPU lane, runs NoFT
before the three updating methods, and exits each lane after its work is done.
The monitor writes `summary.json`, `summary.csv`, and a post-completion GPU
snapshot; no persistent CUDA service remains.
