# Architecture and protocol

## Fixed example construction

```text
official Flickr30k image split
  -> stable caption-index selection per (seed, split, image_id)
  -> positive: target image + its selected human caption
  -> negative: target image + selected caption from a deterministic deranged image
  -> immutable JSONL pair manifest + SHA256
```

For the requested 31,783-image profile, default pair counts are 59,566 for train and 2,000 for test. For the currently published `nlphuji/flickr30k` Karpathy profile, they are 58,000 and 2,000. The official validation images are only checked for leakage and intentionally unused. Train and test share neither image IDs nor pair manifests.

## Dynamic multimodal preprocessing

`preprocess_image_text(processor, images, prompts)` receives one prompt per image. Every prompt passes independently through the original Qwen chat template and processor. This replaces SPAQ's single global prompt interface; it does not concatenate captions across samples.

Qwen may internally pack visual tokens across a batch. Optical processing still recovers per-image token boundaries from `cu_seqlens`, constructs one 120-row field per sample, and then batches those independent fields. Batch size therefore changes parallelism, not the optical field of an individual pair.

The fixed 20,480-pixel processor budget was selected by scanning every persisted pair. The default manifest has maximum combined sequence length 117 at this budget, versus 123 and four over-limit pairs at 25,600. The 120-row limit remains strict: there is no caption truncation, hidden cropping, or fallback remapping. Processor preprocessing is completed and validated before teacher feature extraction.

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

### Transformer-aligned attention and residual

The optical replacement follows Qwen's original pre-norm Transformer equation at stack level:

```text
A = X + NativeAttention(Norm1(X))
Y = A + OpticalMoE(Norm2(A))
```

Vision copies `norm1`, attention, and `norm2` from the configured native vision source block. Language does the same with `input_layernorm`, self-attention, and `post_attention_layernorm`. Only one attention sub-layer precedes each complete optical stack; the electronic MLP is not copied. The copied attention defaults to frozen evaluation mode, so it preserves Qwen's learned RoPE-aware mixing without becoming a large trainable electronic substitute for the optical branch.

The residual coefficient is exactly the fixed value `1.0`, matching Qwen3-VL. There is no learned alpha/beta gate and no activation after the residual addition. Optical square detection, per-expert LayerNorm, and ReLU already provide the branch's internal nonlinearity. On the language side, native DeepStack image injections are accumulated into the residual baseline before every later optical stage; they are not discarded or counted as optical output.

These choices are configured under `student.transformer_block_alignment`. An old student checkpoint without matching alignment metadata is rejected, while teacher caches and the trained teacher head remain reusable because the electronic teacher path and cache targets are unchanged.

## Caches

Teacher cache shards store labels, grid/token lengths, final answer hidden, and four vision targets. Teacher raw logits are generated later by the trained teacher head and stored separately. Processor cache shards store the exact per-pair chat-template/processor outputs used by student training.

Teacher precompute reads the same processor-cache shards instead of repeating JPEG decoding and Qwen image/text preprocessing. This removes a duplicated CPU-heavy pass and permits safe batched frozen-teacher inference.

All cache identities bind to the pair-manifest SHA256. This guarantees that teacher, electronic-language diagnostic, and vision+language optical runs cannot accidentally train against targets generated from different caption pairs.

## Metrics

Both models report BCE, accuracy, balanced accuracy, AUROC, average precision/AUPRC, precision, recall, F1, confusion matrix, and positive/negative counts at threshold 0.5. Prediction CSV files retain the pair ID, target/source image IDs, filename, caption, raw logit, probability, prediction, and correctness.
