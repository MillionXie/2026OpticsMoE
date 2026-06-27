# Commands

Run these from `opticalmoe_experiments/`.

USPS smoke:

```bash
python transfer_adaptation/scripts/train_transfer_prompt.py \
  --config transfer_adaptation/configs/transfer_usps_prompt_only.yaml \
  --run_name transfer_usps_prompt_only_smoke \
  --epochs 1 \
  --smoke_test \
  --device cuda
```

USPS full:

```bash
python transfer_adaptation/scripts/train_transfer_prompt.py \
  --config transfer_adaptation/configs/transfer_usps_prompt_only.yaml \
  --run_name transfer_usps_prompt_only_size5000_seed7 \
  --device cuda
```

KMNIST smoke:

```bash
python transfer_adaptation/scripts/train_transfer_prompt.py \
  --config transfer_adaptation/configs/transfer_kmnist_prompt_only.yaml \
  --run_name transfer_kmnist_prompt_only_smoke \
  --epochs 1 \
  --smoke_test \
  --device cuda
```

KMNIST full:

```bash
python transfer_adaptation/scripts/train_transfer_prompt.py \
  --config transfer_adaptation/configs/transfer_kmnist_prompt_only.yaml \
  --run_name transfer_kmnist_prompt_only_size5000_seed7 \
  --device cuda
```

Rebuild tables:

```bash
python transfer_adaptation/scripts/build_transfer_tables.py \
  --runs_dir transfer_adaptation/runs \
  --out_dir transfer_adaptation/results
```

Run prompt swap only:

```bash
python transfer_adaptation/scripts/run_target_prompt_swap.py \
  --run_dir transfer_adaptation/runs/transfer_usps_prompt_only_size5000_seed7 \
  --checkpoint best.pt \
  --device cuda
```

