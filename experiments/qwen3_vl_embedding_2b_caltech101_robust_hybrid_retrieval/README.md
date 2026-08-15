# Caltech101 Robust Hybrid Optical Retrieval

This experiment reuses the robust hybrid optical Qwen3-VL Student but replaces
the GroceryStore protocol with category-level image-to-image retrieval on all
101 Caltech101 object categories. `BACKGROUND_Google` is excluded.

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

## Run

From the repository root:

```bash
GPU_ID=1 bash \
  experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/commands/01_train_caltech101.sh
```

The command downloads the official 137.4 MB CaltechDATA archive, verifies its
MD5, creates the manifest, caches Teacher embeddings, trains for 40 natural
epochs, and evaluates only the fixed final EMA checkpoint. Existing training
checkpoints are never overwritten.

The default AdamW learning rates are `1e-4` for the electronic adapter/readout,
`5e-5` for routers and `2e-5` for optical phase parameters.
