# Run commands

```bash
# Validate the prepared ImageFolder dataset
python -m experiments.bdd100k_weather4_optical5_mlp 
  --config experiments/bdd100k_weather4_optical5_mlp/configs/bdd100k_weather4.json 
  --phase prepare_data

# Smoke test
python -m experiments.bdd100k_weather4_optical5_mlp 
  --config experiments/bdd100k_weather4_optical5_mlp/configs/bdd100k_weather4_smoke.json 
  --phase all 
  --device cuda

# Full training, 100 epochs
python -m experiments.bdd100k_weather4_optical5_mlp 
  --config experiments/bdd100k_weather4_optical5_mlp/configs/bdd100k_weather4.json 
  --phase all 
  --device cuda 
  --epochs 100

# Balanced subset
python -m experiments.bdd100k_weather4_optical5_mlp 
  --config experiments/bdd100k_weather4_optical5_mlp/configs/bdd100k_weather4_balanced.json 
  --phase all 
  --device cuda 
  --epochs 100

# Evaluate an existing best checkpoint
python -m experiments.bdd100k_weather4_optical5_mlp 
  --config experiments/bdd100k_weather4_optical5_mlp/configs/bdd100k_weather4.json 
  --phase test 
  --device cuda
```
