# Run commands

复制下面任意一行直接执行；所有命令均为单行，不包含续行反斜杠。

## Prepare data

```bash
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run --config experiments/bdd100k_timeofday3_optical5_vs_cnn/configs/timeofday3_optical5.json --phase prepare_data
```

## Optical smoke

```bash
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run --config experiments/bdd100k_timeofday3_optical5_vs_cnn/configs/timeofday3_smoke_optical5.json --phase all
```

## CNN smoke

```bash
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run --config experiments/bdd100k_timeofday3_optical5_vs_cnn/configs/timeofday3_smoke_cnn.json --phase all
```

## Optical full training

```bash
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run --config experiments/bdd100k_timeofday3_optical5_vs_cnn/configs/timeofday3_optical5.json --phase all
```

## Continuous optical smoke

```bash
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run --config experiments/bdd100k_timeofday3_optical5_vs_cnn/configs/timeofday3_smoke_optical5_continuous.json --phase all
```

## Continuous optical full training

```bash
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run --config experiments/bdd100k_timeofday3_optical5_vs_cnn/configs/timeofday3_optical5_continuous.json --phase all
```

## Continuous optical balanced subset

```bash
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run --config experiments/bdd100k_timeofday3_optical5_vs_cnn/configs/timeofday3_balanced_optical5_continuous.json --phase all
```

## Compare O-E-O optical5 and continuous optical5

```bash
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run --phase compare_optical --optical-output-dir runs/bdd100k_timeofday3_optical5_enhanced --continuous-output-dir runs/bdd100k_timeofday3_optical5_continuous
```

## CNN full training

```bash
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run --config experiments/bdd100k_timeofday3_optical5_vs_cnn/configs/timeofday3_cnn.json --phase all
```

## Compare

```bash
python -m experiments.bdd100k_timeofday3_optical5_vs_cnn.run --phase compare --optical-output-dir runs/bdd100k_timeofday3_optical5_enhanced --cnn-output-dir runs/bdd100k_timeofday3_electronic_cnn
```
