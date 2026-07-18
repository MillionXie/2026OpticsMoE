# Run commands

Run from the repository root. Commands are single-line for direct copy and paste.

## Prepare data

```bash
python -m experiments.qwen3_vl_2b_spaq_zeroshot_iqa --config experiments/qwen3_vl_2b_spaq_zeroshot_iqa/configs/spaq_zeroshot.json --phase prepare_data
```

## Smoke evaluation

```bash
CUDA_VISIBLE_DEVICES=2 python -m experiments.qwen3_vl_2b_spaq_zeroshot_iqa --config experiments/qwen3_vl_2b_spaq_zeroshot_iqa/configs/spaq_zeroshot_smoke.json --phase all
```

## Full zero-shot evaluation

```bash
CUDA_VISIBLE_DEVICES=2 python -m experiments.qwen3_vl_2b_spaq_zeroshot_iqa --config experiments/qwen3_vl_2b_spaq_zeroshot_iqa/configs/spaq_zeroshot.json --phase all
```

