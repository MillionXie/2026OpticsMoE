# Run commands

```bash
# Prepare data
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run \
  --config experiments/bdd100k_timeofday3_optical5_vs_cnn/configs/timeofday3_optical5.json \
  --phase prepare_data

# Optical smoke
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run \
  --config experiments/bdd100k_timeofday3_optical5_vs_cnn/configs/timeofday3_smoke_optical5.json \
  --phase all

# CNN smoke
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run \
  --config experiments/bdd100k_timeofday3_optical5_vs_cnn/configs/timeofday3_smoke_cnn.json \
  --phase all

# Optical full training
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run \
  --config experiments/bdd100k_timeofday3_optical5_vs_cnn/configs/timeofday3_optical5.json \
  --phase all

# CNN full training
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run \
  --config experiments/bdd100k_timeofday3_optical5_vs_cnn/configs/timeofday3_cnn.json \
  --phase all

# Compare
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run \
  --phase compare \
  --optical-output-dir runs/bdd100k_timeofday3_optical5_enhanced \
  --cnn-output-dir runs/bdd100k_timeofday3_electronic_cnn
```

