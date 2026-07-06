# Run commands

所有命令均为单行，不包含续行反斜杠。

## Smoke test

```bash
python -m experiments.fashion_mnist_optical5_continuous --config experiments/fashion_mnist_optical5_continuous/configs/fashion_mnist_smoke.json --phase all --device cuda
```

## Uniform random phase initialization

```bash
python -m experiments.fashion_mnist_optical5_continuous --config experiments/fashion_mnist_optical5_continuous/configs/fashion_mnist_uniform.json --phase all --device cuda
```

## Zero phase initialization

```bash
python -m experiments.fashion_mnist_optical5_continuous --config experiments/fashion_mnist_optical5_continuous/configs/fashion_mnist_zeros.json --phase all --device cuda
```

