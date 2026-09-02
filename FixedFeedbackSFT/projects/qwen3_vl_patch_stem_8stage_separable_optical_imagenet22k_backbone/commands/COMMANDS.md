# Commands

All paths are below `2026OpticsMoE`. Runtime checkpoints remain under the
central `FixedFeedbackSFT/runs` tree. The formal launcher always executes a
CPU-only manifest/asset/disk preflight before it checks GPUs, creates a log
directory, or starts `torchrun`.

```bash
# Build a reviewed index once.
IN22K_INDEX_ACTION=build \
IN22K_SOURCE_ROOT=/authorized/path/train \
IN22K_SOURCE_DECLARATION=/authorized/path/fall11_train_declaration.json \
IN22K_INDEX_OUTPUT=/data/indexes/fall11_full_train \
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet22k_backbone/commands/01_audit_or_build_index.sh

# Full checksum audit (use fast_audit only for routine startup checks).
IN22K_INDEX_ACTION=audit \
IN22K_INDEX_OUTPUT=/data/indexes/fall11_full_train \
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet22k_backbone/commands/01_audit_or_build_index.sh

# Explicitly non-performance 100-batch plumbing smoke on physical GPU 5.
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet22k_backbone/commands/02_gpu5_plumbing_smoke.sh

# Full Fall11 launch. This currently hard fails because no data/index exists.
export IMAGENET22K_TRAIN_INDEX=/data/indexes/fall11_full_train
LARGE_DATA_RECIPE=fall11_full IN22K_GPUS=0,1,2,3,5 \
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet22k_backbone/commands/03_launch_imagenet_large.sh

# MIIL-P Fall11 launch requires independently reviewed train and validation indexes.
export IMAGENET21K_P_TRAIN_INDEX=/data/indexes/miil_p_fall11_train
export IMAGENET21K_P_VALIDATION_INDEX=/data/indexes/miil_p_fall11_val
LARGE_DATA_RECIPE=miil_p_fall11 IN22K_GPUS=0,1,2,3,5 \
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet22k_backbone/commands/03_launch_imagenet_large.sh
```
