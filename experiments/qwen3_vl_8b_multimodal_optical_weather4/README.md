# Qwen3-VL-8B Multimodal Optical Weather-4

This experiment uses the full Qwen3-VL multimodal pipeline for BDD100K weather classification.
The teacher is the original electronic Qwen3-VL + MLP classifier. The student keeps the original
tokenizer, processor, LLM, and answer-position feature extraction, but replaces the last Qwen3-VL
vision transformer block with a trainable optical surrogate. The optical surrogate and MLP are
trained jointly using hidden-state distillation, logit KD, and hard-label CE.

## Task and data

Only four visually distinct BDD100K weather classes are used:

```text
clear, rainy, snowy, foggy
```

With `"download": true` (the default), a missing dataset is downloaded from the public BDD100K
archive, and the labelled BDD100K validation split is used as this experiment's test split. The
archives support resuming through `.part` files. Images are linked into the four class folders to
avoid storing a second copy. This downloads more than 4 GB, so ensure that `data_root` has enough
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
image + prompt -> original vision blocks[0:26] -> optical surrogate at block[26]
               -> original merger -> original LLM
               -> answer-position hidden state -> trainable student MLP
```

Transformers packs visual tokens as `[total_tokens, 1152]`; for a fixed-resolution batch this is
logically equivalent to `[batch, tokens_per_image, 1152]`. The surrogate uses `cu_seqlens` to
preserve image boundaries and always returns exactly the original packed shape.

The surrogate is:

```text
LayerNorm(1152) -> Linear(1152, 256) -> ReLU -> amplitude normalization
-> differentiable OpticalGroup -> Linear(256, 1152) -> residual
```

Each OpticalGroup layer contains angular-spectrum propagation, trainable phase and amplitude
masks, square-law detection, normalization, a ReLU-like detector nonlinearity, and amplitude
re-encoding. Token/optical channels are interpolated to the configured 2-D optical field and read
back at the original token resolution.

Only the surrogate (LayerNorm, adapters, optical masks) and student MLP are trainable. The patch
embedding, all unreplaced vision blocks, merger, LLM, and teacher MLP remain frozen. A single Qwen
backbone is shared in memory: the controller switches block 26 between the frozen electronic block
for teacher forward and the optical surrogate for student forward.

## Joint distillation

```text
L = 1.0 * MSE(LN(H_student), LN(H_teacher))
  + 0.5 * T^2 * KL(student_logits/T || teacher_logits/T)
  + 0.5 * CrossEntropy(student_logits, labels)
```

The default temperature is `T=2`. Teacher forward uses `torch.no_grad()`. Student forward does not,
so CE and KD gradients pass through the frozen merger and LLM back to the optical block and MLP.
The student MLP is initialized from `checkpoints/teacher_mlp.pt`; `student_train` fails clearly if
that checkpoint is missing.

## Conservative baseline

Defaults are `dtype=float32` and `attn_implementation=eager`. TF32, cuDNN TF32, autocast, SDPA,
FlashAttention, `torch.compile`, quantization, ONNX, TensorRT, and optical latency estimates are not
used. FP32 Qwen3-VL-8B generally requires more than a 24 GB RTX 4090.

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
