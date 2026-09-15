# Qwen3-VL CIFAR-10 four-layer optical-router classification

This is an independent CIFAR-10 classification experiment. It does not modify the
Caltech101 retrieval package or reuse its 64-dimensional retrieval head.

## Architecture

The frozen Qwen3-VL shell feeds the existing four physical feature stages:

1. Vision expert CCD stage.
2. Vision global CCD stage.
3. Language expert CCD stage.
4. Language global CCD stage.

The language detector representation uses mean-and-max pooling to produce 384
features. The task head is `LayerNorm(384) -> Linear(384, 10)` and returns raw
logits. Training uses cross entropy plus the existing router-balance,
router-importance, optional phase-DC, and CCD operating-point constraints.

## Important initialization rule

`warmstart_body` strictly verifies and loads only `vision_optical` and
`language_optical`. It never loads the old `retrieval_readout` or optimizer.
`random` starts the compact student, routers, optical phases, and ten-class head
from deterministic random seeds.

The expected Warmstart5 file is not bundled with this project. Its required
SHA-256 is:

`6a27f54d8c869cce46150583383a127b0ba47b3d34503f5753aa23974ac1e55d`

## Server commands

From `/root/autodl-tmp`:

```bash
PY=/root/miniconda3/envs/opticsmoe/bin/python
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=120

$PY -m experiments.qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification \
  --config experiments/qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification/configs/server/electronic_topk2_smoke.yaml \
  --phase check --check-batch-size 2

$PY -m experiments.qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification \
  --config experiments/qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification/configs/server/electronic_topk2_quick.yaml \
  --phase train
```

The live log prints per-step loss, cross entropy, accuracy, throughput, learning
rate scale, GPU memory, and active experts. Every epoch prints training and
validation accuracy plus their gap. The official test split is evaluated exactly
once using the best-validation checkpoint.

Outputs are written under:

`/root/autodl-tmp/runs/qwen3_vl_cifar10_router_classification/`
