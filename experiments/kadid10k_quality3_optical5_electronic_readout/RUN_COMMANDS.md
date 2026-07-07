# Run commands

All commands are single-line commands. The first command automatically downloads KADID-10k when the shared data directory is empty.

```bash
python -m experiments.kadid10k_quality3_optical5_electronic_readout.run --config experiments/kadid10k_quality3_optical5_electronic_readout/configs/kadid10k_quality3_smoke.json --phase prepare_data
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.kadid10k_quality3_optical5_electronic_readout.run --config experiments/kadid10k_quality3_optical5_electronic_readout/configs/kadid10k_quality3_smoke.json --phase all
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.kadid10k_quality3_optical5_electronic_readout.run --config experiments/kadid10k_quality3_optical5_electronic_readout/configs/kadid10k_quality3.json --phase all --epochs 100
```

```bash
pytest experiments/kadid10k_quality3_optical5_electronic_readout/tests -q
```
