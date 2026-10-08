# Reproduction commands

Run from the LightGenV2 repository root on the laboratory server. The formal
device contract is physical GPU 6, `NVIDIA A100-PCIE-40GB`.

```bash
export PATH=/home/guest3/miniconda3/envs/xml/bin:$PATH
export PYTHONPATH=$PWD
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6

MODEL=/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
MANIFEST=/DATA/DATA1/guest3/2026OpticsMoE_a100_formal/LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/a100_dataset_contract_20260907/lgvq_prompt_group_split.csv
CHECKPOINT=/DATA/DATA1/guest3/2026OpticsMoE_a100_formal/LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/qwen3vl_quality_tokens_r448_dataset_once_a100_20260907/checkpoints/frames4/best_checkpoint.pt
OUTPUT=LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/a100_temporal_batch_scaling_20260907

python -m LightGenV2.tasks.t06_video_quality_assessment.quality_token_batch_benchmark sweep \
  --model "$MODEL" --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" --batch-sizes 1 2 4 8 16 \
  --sweep-warmup 3 --sweep-trials 30

python -m LightGenV2.tasks.t06_video_quality_assessment.quality_token_batch_benchmark formal \
  --model "$MODEL" --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" --formal-batch 16

python LightGenV2/tasks/t06_video_quality_assessment/reports/a100_audited_table_20260907/build_report.py
```

The sweep warm-up is used only to identify the steady power/throughput plateau.
The formal 558-video evaluation performs zero explicit warm-up and includes its
first GPU batch in the timing distribution.
