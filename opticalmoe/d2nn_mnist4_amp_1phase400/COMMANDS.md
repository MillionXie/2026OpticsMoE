# Commands

Run from this folder:

```bash
cd /DATA/DATA1/guest3/2026OpticsMoE/opticalmoe/d2nn_mnist4_amp_1phase400
```

Smoke test:

```bash
python train_d2nn_mnist256.py --config config_phase_zero.yaml --smoke_test --epochs 1
```

Zero phase initialization:

```bash
CUDA_VISIBLE_DEVICES=0 python train_d2nn_mnist256.py --config config_phase_zero.yaml
```

Uniform phase initialization:

```bash
CUDA_VISIBLE_DEVICES=0 python train_d2nn_mnist256.py --config config_phase_uniform.yaml
```

Gaussian phase initialization:

```bash
CUDA_VISIBLE_DEVICES=0 python train_d2nn_mnist256.py --config config_phase_gaussian.yaml
```

Evaluate a run:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --run_dir runs/d2nn_mnist4_amp_1phase400_sigmoid_zero_seed7 --checkpoint best.pt
```
