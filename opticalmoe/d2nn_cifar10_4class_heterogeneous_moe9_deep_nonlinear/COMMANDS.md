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

## Ten-class training

```bash
CUDA_VISIBLE_DEVICES=3 python opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_deep_nonlinear/train.py --config opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_deep_nonlinear/configs/config_cifar10_10class.yaml --device cuda
```

## Ten-class smoke test

```bash
CUDA_VISIBLE_DEVICES=3 python opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_deep_nonlinear/train.py --config opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_deep_nonlinear/configs/config_cifar10_10class.yaml --device cuda --smoke-test --disable-visualization --run-name cifar10_10class_heterogeneous_moe9_deep_nonlinear_smoke
```

The 4-class and 10-class configs default to per-sample/per-expert intensity
LayerNorm with independent trainable affine maps for every expert and stage,
followed by ReLU. Set `nonlinearity.normalization.affine_sharing: per_stage`
for the lower-parameter shared-affine ablation. Set
`nonlinearity.normalization.per_expert_enabled: false`
for the old stage-global-statistics ablation, or set
`nonlinearity.normalization.elementwise_affine: false` for non-affine
LayerNorm. The normalized output is not multiplied by routing amplitude again.
Detector-region CE and router importance loss both have zero weight; router
balance uses `0.2`. Set `loss.normalize_detector_plane_mse: false` to compare
against the original unnormalized detector-plane MSE.
