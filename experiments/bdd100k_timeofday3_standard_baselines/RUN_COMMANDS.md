# Run Commands

Run from the repository root.

## Linux / macOS

Use forward slashes. Backslashes are shell escape characters on Linux and will
break the config path.

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.bdd100k_timeofday3_standard_baselines --config experiments/bdd100k_timeofday3_standard_baselines/configs/bdd100k_timeofday3_standard_d2nn64.json --phase all
CUDA_VISIBLE_DEVICES=4 python -m experiments.bdd100k_timeofday3_standard_baselines --config experiments/bdd100k_timeofday3_standard_baselines/configs/bdd100k_timeofday3_lenet5.json --phase all
CUDA_VISIBLE_DEVICES=5 python -m experiments.bdd100k_timeofday3_standard_baselines --config experiments/bdd100k_timeofday3_standard_baselines/configs/bdd100k_timeofday3_resnet18.json --phase all
CUDA_VISIBLE_DEVICES=6 python -m experiments.bdd100k_timeofday3_standard_baselines --config experiments/bdd100k_timeofday3_standard_baselines/configs/bdd100k_timeofday3_vgg11_bn.json --phase all
CUDA_VISIBLE_DEVICES=2 python -m experiments.bdd100k_timeofday3_standard_baselines --config experiments/bdd100k_timeofday3_standard_baselines/configs/bdd100k_timeofday3_mobilenet_v2.json --phase all
```

If the server stores BDD100K somewhere else, override the config without editing
JSON:

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.bdd100k_timeofday3_standard_baselines --config experiments/bdd100k_timeofday3_standard_baselines/configs/bdd100k_timeofday3_standard_d2nn64.json --data-root /DATA/DATA1/guest3/data/bdd100k_timeofday3 --phase all
```

Quick import/data-path smoke config:

```bash
python -m experiments.bdd100k_timeofday3_standard_baselines --config experiments/bdd100k_timeofday3_standard_baselines/configs/bdd100k_timeofday3_smoke_d2nn64.json --phase prepare_data
```

Aggregate finished test metrics:

```bash
python -m experiments.bdd100k_timeofday3_standard_baselines --phase compare \
  --baseline-output-dir experiments/bdd100k_timeofday3_standard_baselines/runs/bdd100k_timeofday3_standard_d2nn64 \
  --baseline-output-dir experiments/bdd100k_timeofday3_standard_baselines/runs/bdd100k_timeofday3_lenet5 \
  --baseline-output-dir experiments/bdd100k_timeofday3_standard_baselines/runs/bdd100k_timeofday3_resnet18 \
  --baseline-output-dir experiments/bdd100k_timeofday3_standard_baselines/runs/bdd100k_timeofday3_vgg11_bn \
  --baseline-output-dir experiments/bdd100k_timeofday3_standard_baselines/runs/bdd100k_timeofday3_mobilenet_v2
```

## PowerShell

```powershell
python -m experiments.bdd100k_timeofday3_standard_baselines --config experiments\bdd100k_timeofday3_standard_baselines\configs\bdd100k_timeofday3_standard_d2nn64.json --phase all
python -m experiments.bdd100k_timeofday3_standard_baselines --config experiments\bdd100k_timeofday3_standard_baselines\configs\bdd100k_timeofday3_lenet5.json --phase all
python -m experiments.bdd100k_timeofday3_standard_baselines --config experiments\bdd100k_timeofday3_standard_baselines\configs\bdd100k_timeofday3_resnet18.json --phase all
python -m experiments.bdd100k_timeofday3_standard_baselines --config experiments\bdd100k_timeofday3_standard_baselines\configs\bdd100k_timeofday3_vgg11_bn.json --phase all
python -m experiments.bdd100k_timeofday3_standard_baselines --config experiments\bdd100k_timeofday3_standard_baselines\configs\bdd100k_timeofday3_mobilenet_v2.json --phase all
```

Quick import/data-path smoke config:

```powershell
python -m experiments.bdd100k_timeofday3_standard_baselines --config experiments\bdd100k_timeofday3_standard_baselines\configs\bdd100k_timeofday3_smoke_d2nn64.json --phase prepare_data
```

Aggregate finished test metrics:

```powershell
python -m experiments.bdd100k_timeofday3_standard_baselines --phase compare `
  --baseline-output-dir experiments\bdd100k_timeofday3_standard_baselines\runs\bdd100k_timeofday3_standard_d2nn64 `
  --baseline-output-dir experiments\bdd100k_timeofday3_standard_baselines\runs\bdd100k_timeofday3_lenet5 `
  --baseline-output-dir experiments\bdd100k_timeofday3_standard_baselines\runs\bdd100k_timeofday3_resnet18 `
  --baseline-output-dir experiments\bdd100k_timeofday3_standard_baselines\runs\bdd100k_timeofday3_vgg11_bn `
  --baseline-output-dir experiments\bdd100k_timeofday3_standard_baselines\runs\bdd100k_timeofday3_mobilenet_v2
```
