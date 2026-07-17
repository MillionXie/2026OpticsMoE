# Run commands

Run these commands from the repository root. They are intentionally single-line commands so they can be pasted directly into a shell.

## Inspect and persist the SPAQ split

On the first run this command automatically downloads and extracts SPAQ when it is absent.

```bash
python -m experiments.qwen3_vl_2b_spaq_multitask_iqa --config experiments/qwen3_vl_2b_spaq_multitask_iqa/configs/spaq.json --phase prepare_data
```

## Download only

```bash
python -m experiments.qwen3_vl_2b_spaq_multitask_iqa --config experiments/qwen3_vl_2b_spaq_multitask_iqa/configs/spaq.json --phase download
```

## Full run

```bash
CUDA_VISIBLE_DEVICES=2 python -m experiments.qwen3_vl_2b_spaq_multitask_iqa --config experiments/qwen3_vl_2b_spaq_multitask_iqa/configs/spaq.json --phase all
```

## Smoke run

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_multitask_iqa --config experiments/qwen3_vl_2b_spaq_multitask_iqa/configs/spaq_smoke.json --phase all
```

## Run phases separately

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_multitask_iqa --config experiments/qwen3_vl_2b_spaq_multitask_iqa/configs/spaq.json --phase extract
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_multitask_iqa --config experiments/qwen3_vl_2b_spaq_multitask_iqa/configs/spaq.json --phase train
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_multitask_iqa --config experiments/qwen3_vl_2b_spaq_multitask_iqa/configs/spaq.json --phase test
```

## Offline/local checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_multitask_iqa --config experiments/qwen3_vl_2b_spaq_multitask_iqa/configs/spaq.json --model-id /path/to/Qwen3-VL-2B-Instruct --cache-dir "$HF_HOME" --local-files-only --phase all
```

## Tests

```bash
pytest experiments/qwen3_vl_2b_spaq_multitask_iqa/tests -q
```
