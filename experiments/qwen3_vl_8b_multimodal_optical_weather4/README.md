# Qwen3-VL-8B Multimodal Optical Weather-4

This experiment uses the full Qwen3-VL multimodal pipeline for BDD100K weather classification.
The teacher is the original electronic Qwen3-VL + MLP classifier. The student keeps the original
tokenizer, processor, LLM, and answer-position feature extraction, but replaces the last 20
Qwen3-VL vision transformer blocks with five trainable optical conversions. Each conversion
replaces four consecutive electronic blocks and contains exactly one phase mask. The optical surrogates and MLP are
trained jointly using hidden-state distillation, logit KD, and hard-label CE.

## Task and data

Only four visually distinct BDD100K weather classes are used:

```text
clear, rainy, snowy, foggy
```

With `"download": true` (the default), a missing dataset is downloaded from the public Kaggle
BDD100K mirror `awsaf49/bdd100k-dataset`, and the labelled BDD100K validation split is used as this
experiment's test split. The
archives support resuming through `.part` files. Images are linked into the four class folders to
avoid storing a second copy. This downloads about 6.9 GB, so ensure that `data_root` has enough
space. You remain responsible for complying with the BDD100K dataset license.

The automatically generated (or manually prepared) ImageFolder layout is:

```text
data/bdd100k_weather4/
  train/{clear,rainy,snowy,foggy}/...
  test/{clear,rainy,snowy,foggy}/...
```

Other BDD100K labels such as overcast and undefined must not be placed in these directories.
`train_limit_per_class` and `test_limit_per_class` optionally create deterministic balanced
subsets. Labels are remapped to the explicit class order above rather than ImageFolder's
alphabetical order.

To prepare data without loading a model or requiring a GPU, run `--phase prepare_data`. Set
`"download": false` only when the ImageFolder tree is already prepared manually.

The full processor/chat-template prompt is:

```text
Classify this driving scene into one of the following weather conditions: clear, rainy, snowy, foggy. Answer:
```

## Teacher and student

```text
Teacher:
image + prompt -> original Qwen3-VL vision blocks -> merger -> LLM
               -> answer-position hidden state -> teacher MLP

Student:
image + prompt -> original vision blocks[0:7]
               -> optical 1 replaces electronic blocks[7:11]
               -> optical 2 replaces electronic blocks[11:15]
               -> optical 3 replaces electronic blocks[15:19]
               -> optical 4 replaces electronic blocks[19:23]
               -> optical 5 replaces electronic blocks[23:27]
               -> original merger -> original LLM
               -> answer-position hidden state -> trainable student MLP
```

The inclusive Python-indexed distillation groups are `[7,10]`, `[11,14]`, `[15,18]`,
`[19,22]`, and `[23,26]`. In each group, the student bypasses the four electronic blocks and places
one optical surrogate at the group's endpoint. Qwen's 27-entry block list is retained so
index-dependent model control flow remains stable.

Transformers packs visual tokens as `[total_tokens, 1152]`; for a fixed-resolution batch this is
logically equivalent to `[batch, tokens_per_image, 1152]`. The surrogate uses `cu_seqlens` to
preserve image boundaries and always returns exactly the original packed shape.

The surrogate is:

```text
LayerNorm(1152) -> Linear(1152, 256) -> ReLU -> amplitude normalization
-> differentiable OpticalGroup -> Linear(256, 1152) -> residual
```

Each OpticalGroup contains exactly one angular-spectrum propagation, one trainable phase/amplitude
mask, one square-law detection, normalization, a ReLU-like detector nonlinearity, and amplitude
re-encoding. Token/optical channels are interpolated to the configured 2-D optical field and read
back at the original token resolution.

The optical simulator always computes in FP32/complex64 because PyTorch cannot construct complex
tensors from BF16 components. When a BF16 backbone is selected, the surrogate casts its block
input to FP32 internally and casts the block output back to the backbone dtype. This boundary is
recorded in `model.json`.

Only the five surrogates (LayerNorm, adapters, and one optical mask each) and student MLP are trainable. The patch
embedding, all unreplaced vision blocks, merger, LLM, and teacher MLP remain frozen. A single Qwen
backbone is shared in memory: the controller switches groups `[7,10]` through `[23,26]` between
the frozen electronic teacher blocks and five optical student conversions.

## Joint distillation

```text
L_hidden = mean over 5 groups of MSE(LN(H_student_group), LN(H_teacher_group))

L = 1.0 * L_hidden
  + 0.5 * T^2 * KL(student_logits/T || teacher_logits/T)
  + 0.5 * CrossEntropy(student_logits, labels)
```

The default temperature is `T=2`. Teacher forward uses `torch.no_grad()`. Student forward does not,
so CE and KD gradients pass through the frozen merger and LLM back to the optical block and MLP.
The student MLP is initialized from `checkpoints/teacher_mlp.pt`; `student_train` fails clearly if
that checkpoint is missing.

Teacher feature caches use a strict versioned metadata fingerprint covering dataset path and class
order, sample limits, prompt, image sizing, processor pixel bounds, model, dtype, and attention
implementation. A mismatch is printed and forces re-extraction instead of silently reusing stale
features.

## Numerical configuration

The server config uses `dtype=bfloat16` and `attn_implementation=sdpa` so Qwen3-VL-8B fits on a
24 GB RTX 4090. Optical propagation remains FP32/complex64 for numerical compatibility. The code
does not use `torch.compile`, FlashAttention, quantization, ONNX, TensorRT, or optical latency
estimates.

## Outputs

The run writes:

```text
config_resolved.json
environment.json
dataset.json
model.json
features/{train,test}.pt
metrics/teacher_inference.json
metrics/student_inference.json
metrics/student_training_history.csv
metrics/student_training.json
metrics/comparison.json
checkpoints/teacher_mlp.pt
checkpoints/student_mlp.pt
checkpoints/optical_surrogate.pt
```

No optical inference-time estimate is produced. See [RUN_COMMANDS.md](RUN_COMMANDS.md) for commands.
