# Caltech101 Robust Hybrid Optical Retrieval

This experiment reuses the robust hybrid optical Qwen3-VL Student for a
two-stage category-level retrieval route: 40 epochs on all 101 Caltech101
object categories, followed by 20 epochs on a fixed target set of ten classes.
`BACKGROUND_Google` is excluded.

Caltech101 has no official retrieval split or standard gallery. The checked-in
protocol creates a deterministic, disjoint split using seed 42: three gallery
references, up to 30 training images and up to 20 test images per class. It
caps large classes without duplicating short classes. On the official archive
this produces 303 gallery, 3,024 train and 1,585 held-out test images. The
manifest and its SHA-256 digest are persisted with every run.

The model still produces 64-dimensional normalized embeddings. Training uses
the frozen Qwen3-VL Teacher losses, supervised retrieval loss, gallery-aligned
losses, robust optical residual/refiner blocks, phase dropout, k-space limiting
and alignment perturbations from the robust Grocery experiment.

Teacher embeddings live under the experiment-level `cache/` directory rather
than a run directory. The target-10 cache is sliced from the compatible
all-class cache by image path, so changing `output_dir` never repeats Qwen
forward passes.

The target-10 model also has a four-plane deployment pipeline adapted from the
Grocery10 release. It exports amplitude/phase BMPs, registers real CCD files,
runs the electronic bridges between planes, and evaluates measured embeddings.

## Commands

All copyable training, evaluation and physical SLM/CCD commands are kept in
`RUN_COMMANDS.md`; no shell wrapper is required.

The default AdamW learning rates are `1e-4` for the electronic adapter/readout,
`5e-5` for routers and `2e-5` for optical phase parameters.
