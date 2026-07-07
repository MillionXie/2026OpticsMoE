# Qwen3-VL-2B CIFAR-10 optical fullstack4 token64 residual

This is a Qwen-based teacher/student distillation experiment, not a standalone optical classifier. It evaluates CIFAR-10 with the complete electronic Qwen3-VL-2B + MLP as teacher and a student that replaces both the full vision Transformer stack and the full language Transformer stack with independent four-conversion optical surrogates.

## Model paths

```text
Teacher
image + prompt -> complete electronic Qwen3-VL-2B -> answer hidden -> teacher MLP -> 10 logits

Student
image + prompt
  -> frozen vision patch embedding
  -> vision optical4 token64 residual surrogate
  -> frozen vision merger and multimodal injection
  -> language optical4 token64 residual surrogate
  -> frozen final RMSNorm
  -> answer hidden
  -> trainable student MLP
  -> 10 logits
```

Teacher Qwen inference is performed once by `teacher_precompute`. Student training reads cached teacher vision-stack outputs, answer-position hidden features, and teacher logits; it never runs the teacher online.

## CIFAR-10

The standard torchvision CIFAR-10 train and test splits are used. The 50,000-image train split is divided stratifiably into student training and validation subsets; the 10,000-image test split is reserved for final evaluation. `download=true` enables torchvision's automatic download.

Classes are `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`. The fixed short prompt is:

```text
Classify: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Answer:
```

## Token64 optical adapter

The processor pixel budget is fixed to 16,384 pixels by default. Each token is projected with `Linear -> LayerNorm -> Softplus` to 64 non-negative optical channels. The `[T, 64]` representation is written directly into the first `T` rows of a zero-initialized `[64, 64]` field. There is no bilinear interpolation, crop, truncation, pooling fallback, or multi-field fallback.

After four optical conversions, only valid token rows are read. The vision surrogate projects them back to 1,024 features for the frozen Qwen vision merger. The language surrogate projects them back to 2,048 features. Visual token count or language sequence length above 64 causes an explicit error.

Each surrogate uses an independently configurable residual branch:

```text
Y = beta * X + alpha * Delta
```

By default `beta_v=beta_l=1.0` are fixed and `alpha_v=alpha_l=0.1` are trainable. These values are saved in `model.json`, epoch metrics, inference metrics, and checkpoint metadata.

See [ARCHITECTURE.md](ARCHITECTURE.md) for shapes and losses, and [RUN_COMMANDS.md](RUN_COMMANDS.md) for single-line commands.
