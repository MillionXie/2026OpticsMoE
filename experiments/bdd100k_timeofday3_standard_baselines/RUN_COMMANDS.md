# Run Commands

Run from the repository root.

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

