# Eight-stage optical ImageNet-large pretraining

This is an independent, manifest-locked ImageNet-21K/22K pretraining project.
It never edits or resumes the running ImageNet-1K formal trainer. It reads only
the immutable P11 asset at `FixedFeedbackSFT/runs/_assets/8stage`, creates a new
10,450/11,221/21,841-way readout, and asserts that the source checkpoint
contains no 1,000-way readout tensors.

## What is ready now

- Original Fall11 template: 21,841 classes, 14,197,122 images, 90 epochs,
  evaluation disabled because this contract has no official validation split.
- MIIL-P Fall11 template: 11,221 classes, 11,797,632 train images and 561,052
  validation images, 80 epochs, with separately manifest-locked indexes.
- A 100-batch GPU-5 plumbing smoke using the existing ImageNet-1K cache and a
  real 21,841-way head. Every manifest/result calls it non-publishable and it
  must never be reported as ImageNet-22K accuracy.

The server currently has no ImageNet-21K/22K data and no usable download token.
Therefore no formal large-data training is started. `03_launch_imagenet_large.sh`
hard fails in a CPU-only preflight before creating an output directory or
starting CUDA/NCCL when its reviewed index is absent.

## Data contract

The data owner supplies `authorized-class-folder-source-v1` JSON with exact
release/variant/split IDs, class/sample counts, access acknowledgement and a
fixed WNID list. Index construction writes `samples.tsv`, `offsets.u64`,
`class_to_idx.json`, per-class counts and SHA-256 identities. Runtime workers
memory-map the TSV and offsets; they do not hold 14 million Python tuples.

The training sampler applies a deterministic, bijective global affine
permutation before rank sharding. This avoids class-major 4,096-image blocks
while requiring O(1) shuffle memory.

Formal preflight re-hashes the complete sample TSV and uint64 offset file,
requires each indexed `source_root` to remain a live directory, and locks each
split ID. Train/validation additionally require identical WNID-list and
`class_to_idx` hashes; their common taxonomy digest is stored in every run and
checkpoint identity.

## Optimization and checkpoint contract

- AdamW with independent optical-phase LR, electronic LR, head LR and
  layer-wise decay.
- Mixup/CutMix with soft-target cross-entropy. Dense BCE is not used.
- AMP, EMA, DDP, exact per-rank RNG resume and config/data/code/asset identity
  locks.
- A short final gradient-accumulation window uses its actual micro-batch count.
  AMP overflow does not advance the scheduler, EMA, or optimizer-step counter.
- Without validation, only `backbone_last_raw.pt` and
  `backbone_last_ema.pt` are exported; neither is called “best”.
- With reviewed validation, raw/EMA selection is explicit and a single
  `backbone_best_validated.pt` is exported.

## Gate to the next stage

1. Obtain the authorized corpus and review its exact version/licence scope.
2. Build and checksum-audit indexes; verify class/sample counts exactly.
3. Run the 100-batch plumbing smoke, checking all eight phase gradients,
   resume identity and raw/EMA exports.
4. Run a short 1-epoch real-data pilot before the 80/90-epoch schedule.
5. After large-taxonomy pretraining, replace the temporary head and fine-tune
   on ImageNet-1K for 30 epochs. The transfer run must record the source
   backbone SHA, reset the 1,000-way readout, use a lower backbone LR than head
   LR, and report IN-1K validation Top-1/Top-5. This downstream contract belongs
   in a separate project/checkpoint format and is not silently folded into this
   trainer.

See `commands/COMMANDS.md` for all entry points.
