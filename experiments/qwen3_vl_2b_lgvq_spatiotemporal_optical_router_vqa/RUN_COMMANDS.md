# Run commands

Run every command from the `2026OpticsMoE` repository root. Formal server paths
below match the uploaded LGVQ server; edit them only if that server layout moves.
On the current single-GPU Seetacloud host, start with:

```bash
cd /root/autodl-tmp/workspace/opticsmoe
```

## 0. One-time environment check

The verified server uses `/root/miniconda3/bin/python` directly (no named conda
environment is required):

```bash
/root/miniconda3/bin/python -c "import torch, transformers, cv2; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

The cache path requires a local Qwen model directory. It never downloads from
Hugging Face.

## 1. Build the fixed prompt-group manifest

```bash
python -m experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa.prepare_manifest \
  --dataset-root /root/autodl-tmp/workspace/LGVQ \
  --output experiments/qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa/artifacts/lgvq_prompt_group_split.csv \
  --seed 42
```

Do not continue unless the report is exactly `train=2250`, `validation=0`,
`test=558`. The command cross-checks `prompt_cls.json` against `MOS.txt` by
normalized path and never exports alignment.

## 2. Cache real frozen-Qwen inputs directly from MP4

Choose a free GPU; this stage loads Qwen3-VL-Embedding-2B once. The output
contains about 4.5 GB of float16 Vision tensors, so leave sufficient RAM and
disk space.

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa.cache_qwen_inputs \
  --dataset-root /root/autodl-tmp/workspace/LGVQ \
  --model-path /root/autodl-tmp/models/Qwen3-VL-Embedding-2B \
  --manifest experiments/qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa/artifacts/lgvq_prompt_group_split.csv \
  --output experiments/qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa/artifacts/qwen_inputs_196x1024.pt \
  --batch-size 8 \
  --device cuda
```

If Qwen cache generation runs out of VRAM, lower only `--batch-size` to 4 or 2;
this does not change the cached values or experiment definition.

The generic `--phase cache` entry is retained only for inspecting/migrating old
sample-ID-bearing assets. Formal preflight intentionally accepts only the direct
Qwen contract above, so new runs do not depend on a nonexistent
`source_feature_cache`.

## 3. Preflight and fast synthetic wiring test

```bash
python -m experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa \
  --config experiments/qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa/configs/release/e2.yaml \
  --phase preflight

python -m experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa \
  --config experiments/qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa/configs/release/e2.yaml \
  --phase smoke
```

Preflight must report the formal geometry, a 2,250/0/558 split, Qwen cache shape
`[2808,4,196,1024]`, fixed prompt, and `alignment=forbidden`. Smoke deliberately
uses a tiny CPU plane and is not an accuracy benchmark.

## 4. Fair router experiments

These four commands keep data, feature blocks, fusion range, losses and output
head fixed while changing the router mechanism/top-k. O2 necessarily adds a
trainable router phase, detector propagation/capture regularizer and its own
phase learning rate; those are part of the optical-router mechanism. The
current server has only GPU 0, so the safest reproducible procedure is to run
the four commands sequentially:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa \
  --config experiments/qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa/configs/release/e1.yaml \
  --phase train

CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa \
  --config experiments/qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa/configs/release/e2.yaml \
  --phase train

CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa \
  --config experiments/qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa/configs/release/e4.yaml \
  --phase train

CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa \
  --config experiments/qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa/configs/release/o2.yaml \
  --phase train
```

Test is run at epoch 1, every 5 epochs, and the final epoch. The selected file
is `runs/lgvq_<variant>/best_observed_test_checkpoint.pt`; the selection metric
is `mean(SRCC_spatial, SRCC_temporal)`. This is test-guided model selection by
explicit request, not a held-out generalization estimate.

All four release configs use `alpha∈[0.01,0.49]` (low-optical regime). They do
not cover `alpha>0.5` or optical-off; add those only as separate fusion
ablations, never by silently changing one row of the router table.

## 5. Re-evaluate a selected checkpoint and export predictions

Example for O2:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa \
  --config experiments/qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa/configs/release/o2.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa/runs/lgvq_o2/best_observed_test_checkpoint.pt
```

Important outputs are:

- `metrics_best_observed_test.json`: best periodically observed test metrics;
- `training_summary.json`: best epoch and selection policy;
- `test_metrics.json`: explicit re-evaluation metrics;
- `test_predictions.csv`: path-keyed spatial/temporal target and prediction;
- `fusion_diagnostics.json`: test-set batch-weighted four-layer alpha/E/O RMS diagnostics;
- `router_diagnostics.json`: test-set expert probability, selection share, entropy and optical capture;
- `parameter_breakdown.json`: phase/router/other parameter counts;
- `resolved_config.json` and `preflight.json`: auditable experiment contract.

## 6. Fields that may be edited for a new run

Copy a release YAML rather than editing `common.yaml` in place. Normally change
only `output_dir`, `training.epochs`, `training.batch_size`, and the explicit
cache/manifest paths. For the fair router table, do not change geometry,
detector intervals, fusion bounds, prompt, loss, or learning rates between
E1/E2/E4/O2.
