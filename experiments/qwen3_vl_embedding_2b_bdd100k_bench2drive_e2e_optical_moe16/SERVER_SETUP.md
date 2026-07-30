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

Closed-loop SAC additionally requires:

1. CARLA 0.9.15 with a Python client matching its supported Python ABI;
2. the official Bench2Drive repository and routes;
3. a Gymnasium-style adapter factory configured through `sac.env_factory`.

Installing only the PyPI `carla` name is not considered a valid CARLA setup.
The simulator distribution, Python egg/wheel and server binary must belong to
the same CARLA release.

