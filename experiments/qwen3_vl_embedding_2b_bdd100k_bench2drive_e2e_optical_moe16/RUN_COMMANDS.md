# Commands

All commands are run from the `2026OpticsMoE` repository root.

## Dependency-free smoke

```bash
python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive_smoke.yaml --phase smoke
```

## Tests

```bash
pytest experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/tests -q
```

## Bench2Drive Base scratch baseline (no BDD100K)

Install the offline behavior-cloning dependencies:

```bash
python -m pip install -r experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/requirements.txt
```

Download the official Base repository sequentially. The downloader is resumable
and retains only `camera/rgb_front` and `anno`; completed tar archives are deleted:

```bash
HF_ENDPOINT=https://hf-mirror.com python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16.prepare_bench2drive_base --archive-list ../third_party/Bench2Drive/docs/bench2drive_base_1000.json --output-dir data/bench2drive/base --archive-dir data/bench2drive/_downloads/base --workers 4
```

Validate/index the extracted data:

```bash
python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive_base_scratch.yaml --phase prepare_bench2drive
```

Run behavior cloning from zero raw phase. Optical core, CCD recombiner, and Actor
are trainable starting in stage 1; native Qwen parameters remain frozen:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive_base_scratch.yaml --phase bc_all
```

The formal scratch config uses batch size 8 and deterministic rotating windows
of at most 2,000 frames per navigation command per epoch. It writes a resumable
step checkpoint every 250 batches. Re-running the same command resumes an
interrupted stage; delete that experiment's checkpoints for a deliberate fresh run.

For the downloaded Base data, command counts are
`5436/2891/3557/34053/631/767`. The cap therefore produces 9,398 samples per
epoch and complete coverage after 18 rotating epochs. Four epochs do not cover
the dominant 34,053-sample command.

Validate the real-data forward/backward path with only 24 training records and
one epoch per BC stage:

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive_base_scratch_runtime_smoke.yaml --phase bc_all
```

Equivalent explicit phases (useful for resuming/debugging):

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive_base_scratch.yaml --phase bc_stage1
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive_base_scratch.yaml --phase bc_stage2
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive_base_scratch.yaml --phase bc_evaluate
```

## Data validation

```bash
python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive.yaml --phase prepare_data
```

## BDD100K Optical Backbone pretraining

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive.yaml --phase fit_pca
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive.yaml --phase bdd_pretrain
```

## Bench2Drive behavior cloning

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive.yaml --phase bc_all
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive.yaml --phase bc_evaluate
```

## Closed-loop SAC

The Python-3.8 CARLA service and Python-3.11 Qwen/SAC learner are connected by
the implemented localhost bridge. After BC has produced
`checkpoints/bc_policy_best.pt`, run the whole lifecycle with one command:

```bash
TRAINING_CUDA=2 CARLA_GRAPHICS_ADAPTER=1 \
  experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/run_sac_closed_loop.sh \
  experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive_base_scratch.yaml
```

This starts CARLA, waits for readiness, starts the route-environment service,
runs SAC in `xml`, and stops both services on exit. `TRAINING_CUDA` controls
PyTorch; `CARLA_GRAPHICS_ADAPTER` separately controls Vulkan rendering.

The equivalent manual three-step workflow is:

```bash
CARLA_GRAPHICS_ADAPTER=1 \
  experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/start_carla_bridge_rfl.sh
```

```bash
CUDA_VISIBLE_DEVICES=2 /home/guest3/miniconda3/envs/xml/bin/python -m \
  experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 \
  --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive_base_scratch.yaml \
  --phase sac_train
```

```bash
experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/stop_carla_bridge_rfl.sh
```

The bridge uses `24615`; CARLA reserves `24515` and `24516` itself.

Server layout used by this experiment:

```text
/DATA/DATA1/guest3/third_party/CARLA_0.9.15
/DATA/DATA1/guest3/third_party/Bench2Drive
/home/guest3/miniconda3/envs/RFL       # Python 3.8 CARLA runtime
```

`carla_env_server.py` owns the simulator and sensors in RFL;
`carla_bridge.py` exposes the Gymnasium-style proxy in xml. This is the
closed-loop route-training environment. Official Bench2Drive scenario and
leaderboard evaluation remains a separate final evaluation protocol.
