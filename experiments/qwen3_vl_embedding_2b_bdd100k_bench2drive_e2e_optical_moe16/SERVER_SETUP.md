# Server setup

## Central data layout

The experiment intentionally uses one repository-level copy of each dataset:

```text
2026OpticsMoE/
└── data/
    ├── bdd100k/
    │   ├── images/100k/train/*.jpg
    │   ├── images/100k/val/*.jpg
    │   └── labels/
    │       ├── bdd100k_labels_images_train.json
    │       └── bdd100k_labels_images_val.json
    └── bench2drive/
        └── base/
            └── <clip>/
                ├── camera/rgb_front/*.jpg
                └── anno/*.json.gz
```

Do not copy either dataset into the experiment directory. If a legacy
experiment requires its old path, point that path to the central dataset with
a symbolic link.

The BDD pretraining configuration requires usable auxiliary supervision.
Detection annotations provide road-participant targets. Drivable-area and lane
targets must be present either as supported polygon annotations or through the
configured official PNG mask roots.

## Output layout

Generated artifacts are stored inside the experiment:

```text
experiments/
└── qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/
    └── runs/
        └── qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/
```

Reusable cache data remains in the repository-level `cache/` directory.

## Python environments

Offline BDD pretraining and Bench2Drive behavior cloning use the repository
environment with PyTorch, Transformers, Pillow, NumPy and PyYAML.

The no-BDD baseline uses `configs/bench2drive_base_scratch.yaml`. Download Base
with the official manifest from a Bench2Drive checkout:

```bash
HF_ENDPOINT=https://hf-mirror.com python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16.prepare_bench2drive_base --archive-list ../third_party/Bench2Drive/docs/bench2drive_base_1000.json --output-dir data/bench2drive/base --archive-dir data/bench2drive/_downloads/base --workers 4
```

The downloader keeps only the modalities consumed by this experiment and can
be safely rerun: each successfully extracted archive has an independent marker.

Closed-loop SAC additionally requires:

1. CARLA 0.9.15 with a Python client matching its supported Python ABI;
2. the official Bench2Drive repository and routes;
3. a Gymnasium-style adapter factory configured through `sac.env_factory`.

Installing only the PyPI `carla` name is not considered a valid CARLA setup.
The simulator distribution, Python egg/wheel and server binary must belong to
the same CARLA release.

On the lab server the intended isolated locations are:

```text
/DATA/DATA1/guest3/third_party/Bench2Drive
/DATA/DATA1/guest3/third_party/CARLA_0.9.15
/home/guest3/miniconda3/envs/RFL           # Python 3.8 CARLA/SAC environment
```

Qwen/optical offline training remains in the `xml` environment. Do not move it
into the legacy Python 3.8 environment merely to run behavior cloning.

On the configured server, start and verify the simulator with:

```bash
conda activate RFL
CARLA_GRAPHICS_ADAPTER=5 bash experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/start_carla_rfl.sh 24515
python experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/carla_runtime_check.py --port 24515 --timeout 60
```

CARLA 0.9.15's Python client is built for Python 3.7/3.8, whereas the current
Qwen3-VL Transformers stack requires the repository's newer `xml` Python.
Consequently a closed-loop SAC run needs a Gymnasium bridge process: CARLA and
Bench2Drive run in `RFL`, while Qwen/optical inference and SAC optimization run
in `xml`. Merely setting `sac.env_factory` to an in-process CARLA factory would
load the incompatible CARLA extension into Python 3.11 and is unsupported.

The process bridge is implemented. CARLA reserves its RPC port and `RPC+1`, so
the configured ports are:

```text
CARLA RPC:       24515
CARLA streaming: 24516
Python bridge:   24615
```

Use the complete managed launch rather than importing CARLA into `xml`:

```bash
TRAINING_CUDA=2 CARLA_GRAPHICS_ADAPTER=1 \
  experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/run_sac_closed_loop.sh \
  experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive_base_scratch.yaml
```
