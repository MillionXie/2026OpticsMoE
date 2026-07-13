# Commands

Run from:

```bash
cd /DATA/DATA1/guest3/2026OpticsMoE/opticalmoe/d2nn_cifar10_4class_moe9_5layer480
```

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/config.yaml --smoke_test
```

Full training:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/config.yaml
```

Change `optics.phase_init`, `prompt.top_k`, or the commented k-space fields directly in `configs/config.yaml` for later ablations.
