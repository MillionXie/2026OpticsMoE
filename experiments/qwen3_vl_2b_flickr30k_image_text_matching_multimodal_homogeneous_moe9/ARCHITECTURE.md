# Architecture and protocol

## Fixed example construction

```text
official Flickr30k image split
  -> stable caption-index selection per (seed, split, image_id)
  -> positive: target image + its selected human caption
  -> negative: target image + selected caption from a deterministic deranged image
  -> immutable JSONL pair manifest + SHA256
```

Default pair counts are 59,566 for train and 2,000 for test. The official 1,000-image validation split is only checked for leakage and intentionally unused. Train and test share neither image IDs nor pair manifests.

## Dynamic multimodal preprocessing

`preprocess_image_text(processor, images, prompts)` receives one prompt per image. Every prompt passes independently through the original Qwen chat template and processor. This replaces SPAQ's single global prompt interface; it does not concatenate captions across samples.

Qwen may internally pack visual tokens across a batch. Optical processing still recovers per-image token boundaries from `cu_seqlens`, constructs one 120-row field per sample, and then batches those independent fields. Batch size therefore changes parallelism, not the optical field of an individual pair.

## Electronic teacher

The full Qwen3-VL-2B backbone remains frozen and in evaluation mode. The last valid language hidden vector is passed to `NormalizedBinaryClassificationHead`:

```text
LayerNorm(2048) -> Linear(2048,1) -> raw logit
```

The head trains on all retained train pairs. By explicit experiment policy, every epoch is evaluated on test and the checkpoint with highest test AUROC is retained. This is not a conventional held-out final-test protocol, so output reports label the selection bias.

## Optical students

Vision optical input maps each image's packed hidden group `[T,1024]` directly to `[T,120]`, zero-pads token rows, and processes the resulting 120×120 field with the verified 480×480 homogeneous MoE9 system. It preserves Qwen's native three vision DeepStack taps at optical stages `[1,3,4]`.

The main mode also replaces the language decoder stack with an independent optical MoE9. It maps each valid sequence `[S,2048]` to `[S,120]`, preserves padding boundaries, incorporates the three native visual injections, and restores `[S,2048]`. The diagnostic mode keeps the original frozen electronic language layers.

Common physical structure:

- 480×480 canvas and 450×450 active area
- 3×3 experts, 120×120 each, pitch 150
- input-dependent top-3 routing
- five phase layers per expert
- per-expert square detection, non-affine LayerNorm, ReLU, routing-weight reapplication, and hard zeroing between stages
- global phase, propagation and full-plane detector readout
- no crop or fallback resize when a token limit is exceeded

## Caches

Teacher cache shards store labels, grid/token lengths, final answer hidden, and four vision targets. Teacher raw logits are generated later by the trained teacher head and stored separately. Processor cache shards store the exact per-pair chat-template/processor outputs used by student training.

All cache identities bind to the pair-manifest SHA256. This guarantees that teacher, electronic-language diagnostic, and vision+language optical runs cannot accidentally train against targets generated from different caption pairs.

## Metrics

Both models report BCE, accuracy, balanced accuracy, AUROC, average precision/AUPRC, precision, recall, F1, confusion matrix, and positive/negative counts at threshold 0.5. Prediction CSV files retain the pair ID, target/source image IDs, filename, caption, raw logit, probability, prediction, and correctness.
