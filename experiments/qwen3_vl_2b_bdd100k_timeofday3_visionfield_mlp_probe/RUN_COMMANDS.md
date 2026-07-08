# Run commands

## Extract fixed features

```bash
python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_visionfield_mlp_probe.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_visionfield_mlp_probe/configs/bdd100k_timeofday3_visionfield_probe.json --phase extract_features
```

## Train probe

```bash
python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_visionfield_mlp_probe.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_visionfield_mlp_probe/configs/bdd100k_timeofday3_visionfield_probe.json --phase train_probe
```

## Probe inference

```bash
python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_visionfield_mlp_probe.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_visionfield_mlp_probe/configs/bdd100k_timeofday3_visionfield_probe.json --phase probe_inference
```

## All phases

```bash
python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_visionfield_mlp_probe.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_visionfield_mlp_probe/configs/bdd100k_timeofday3_visionfield_probe.json --phase all
```

## Smoke run

```bash
python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_visionfield_mlp_probe.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_visionfield_mlp_probe/configs/bdd100k_timeofday3_visionfield_probe_smoke.json --phase all
```

## Tests

```bash
pytest experiments/qwen3_vl_2b_bdd100k_timeofday3_visionfield_mlp_probe/tests -q
```

