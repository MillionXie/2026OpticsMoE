# MOS aggressive task-oriented fine-tuning

This configuration is a second-stage fine-tune of the best checkpoint produced
by `spaq_mos_epoch77_regularized_finetune_sam.json`. The source checkpoint is
stored at global epoch 90 (77 source epochs plus 13 fine-tuning epochs).

Use:

```text
configs/spaq_mos_epoch90_aggressive_rank_nin_sam_3way_batch8.json
```

The checkpoint only initializes model parameters. The new run resets AdamW,
the cosine scheduler, and SAM, and retains the latest throughput path:

- student and inference batch size 8;
- exact three-way rotating train partitions;
- all 10,013 training samples are covered once every three epochs;
- cache-local batch fetching;
- batched expert phase modulation;
- intermediate optical fields are retained only on visualization epochs.

## Objective

The previous fine-tune was dominated by intermediate teacher-feature losses.
This run makes the real MOS objective dominant:

```text
L =
    1.0   * SmoothL1(student_mos, ground_truth_mos)
  + 1.0   * NormInNorm_p1_q2(student_mos, ground_truth_mos)
  + 0.1   * PairwiseRanking(student_mos, ground_truth_mos)
  + 0.1   * SmoothL1(student_mos, teacher_mos)
  + 0.005 * RouterBalance
```

Vision-hidden and answer-hidden teacher losses are both zero. The frozen
teacher is therefore only a weak output-level regularizer; the optical student
is optimized end-to-end mainly from SPAQ MOS labels. Because hidden losses are
disabled, student training also skips loading teacher hidden/tap tensors from
the sharded teacher cache.

The pairwise loss uses target pairs separated by at least 0.02 in normalized
MOS (two points on the 0--100 scale). Norm-in-Norm follows the official
recommended `p=1, q=2` normalization. Single-sample remainder batches return a
differentiable zero for these two batch-relative losses; SmoothL1 remains
active.

## Run

From the repository root:

```text
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224/configs/spaq_mos_epoch90_aggressive_rank_nin_sam_3way_batch8.json --phase student_train
```
