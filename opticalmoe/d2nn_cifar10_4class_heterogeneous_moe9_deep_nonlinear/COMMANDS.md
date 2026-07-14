# Commands

Run from the repository root. No line-continuation backslashes are used.

## Unit tests

```bash
python -m pytest opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_deep_nonlinear/tests -q
```

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_deep_nonlinear/train.py --config opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_deep_nonlinear/configs/config.yaml --device cuda --smoke-test --disable-visualization --run-name cifar10_4class_heterogeneous_moe9_deep_nonlinear_smoke
```

## Full training

```bash
CUDA_VISIBLE_DEVICES=3 python opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_deep_nonlinear/train.py --config opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_deep_nonlinear/configs/config.yaml --device cuda
```

The default config uses parameter-free per-sample/per-stage intensity
LayerNorm followed by ReLU. Detector-region CE and router importance loss both
have zero weight. Set `loss.normalize_detector_plane_mse: false` to compare
against the original unnormalized detector-plane MSE.
