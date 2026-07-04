# Qwen3-VL-2B Multimodal Optical Weather-4

This experiment uses **Qwen3-VL-2B**, not the 8B model, for BDD100K Weather-4
classification. The teacher is the complete electronic Qwen3-VL-2B multimodal model plus an MLP
classifier. The student preserves the processor, chat template, tokenizer, vision-language merger,
LLM, answer-position hidden extraction, and MLP path, but replaces the last 20 vision transformer
blocks with five optical conversions.

## Dataset and prompt

The ImageFolder layout is:

```text
data/bdd100k_weather4/
  train/{clear,rainy,snowy,foggy}/...
  test/{clear,rainy,snowy,foggy}/...
```

The full prepared dataset currently contains:

| Class | Full train | Test |
|---|---:|---:|
| clear | 37,344 | 5,346 |
| rainy | 5,070 | 738 |
| snowy | 5,549 | 769 |
| foggy | 130 | 13 |
| total | 48,093 | 6,866 |

With `validation_fraction=0.1`, the deterministic stratified split is approximately 43,284 student
training samples and 4,809 validation samples. `dataset.json` records split sizes, per-class counts,
and imbalance ratios. The dataset is extremely imbalanced, especially for `foggy`; report macro-F1
and per-class accuracy in addition to overall accuracy. The balanced config caps each class at 130
training and 13 test samples.

Every image is paired with this prompt through the full Qwen processor/chat-template/tokenizer path:

```text
Classify this driving scene into one of the following weather conditions: clear, rainy, snowy, foggy. Answer:
```

## Dynamic vision replacement

No vision depth or hidden dimension is hardcoded. Runtime values come from:

```python
model.config.vision_config.depth
model.config.vision_config.hidden_size
model.config.text_config.hidden_size
```

The replacement starts at `vision_depth - replace_last_n_vision_blocks`. For the expected 24-block
Qwen3-VL-2B vision transformer, the five inclusive teacher groups are:

```text
[4, 7], [8, 11], [12, 15], [16, 19], [20, 23]
```

Each four-block electronic teacher transformation is distilled into one optical conversion. The
original block-list structure is preserved with identity placeholders. The optical conversion is
placed at each group entrance so Qwen deep-stack taps inside a replaced group observe transformed
rather than untransformed features.

## One physical conversion per group

Each optical conversion contains exactly:

```text
LayerNorm -> InputAdapter -> ReLU amplitude encoding
-> per-image token-to-field interpolation
-> one angular-spectrum propagation
-> one trainable phase/amplitude mask
-> one square-law detection and normalization
-> per-image field-to-token interpolation
-> OutputAdapter
```

`optical_layers` is enforced to equal `1`. There is **no electronic residual bypass**: the output is
`OutputAdapter(optical_tokens)`, never `packed + OutputAdapter(...)`.

Packed hidden states must provide `cu_seqlens`. They are split into individual images before field
construction. Batch size controls only parallelism; it never merges `[sum(T_i), optical_dim]` into a
single optical field. Missing or inconsistent boundaries raise an error. The first training batch
records per-image token counts, packed shapes, group shapes, field shapes, and logits shapes in
`metrics/student_training.json`.

## Cached electronic teacher targets

`--phase teacher_cache` runs the full electronic teacher once and caches:

- labels, teacher logits, and teacher answer hidden states;
- input and output hidden states for every four-block teacher group;
- per-image token counts and `image_grid_thw` summaries.

`teacher_cache/train.pt` and `teacher_cache/test.pt` are small manifests. Tensor data is stored in
shards below `teacher_cache/train_shards/` and `test_shards/`, because full-resolution intermediate
features can require hundreds of gigabytes. Student training lazily reads these shards and does not
instantiate or run an additional teacher network.

Cache metadata includes model ID, prompt, dataset root, class names, processor pixel bounds,
runtime vision/text dimensions, group list, dtype, attention implementation, image-grid summary,
teacher-MLP SHA-256, model revision, and schema version. A mismatch invalidates the cache with an
explicit message.

## Optimization

Only the five optical surrogates and student MLP are trainable. All original Qwen parameters and the
teacher MLP are frozen. The student MLP is initialized from `teacher_mlp.pt`.

```text
L_hidden = mean_g MSE(LN(H_student_g), LN(H_teacher_g))
L_KD = T^2 KL(log_softmax(student/T), softmax(teacher/T))
L_total = 1.0 L_hidden + 0.5 L_KD + 0.5 CrossEntropy(student, label)
```

The default temperature is 2.0. Optical FFT propagation computes in FP32/complex64 even when the
Qwen backbone uses BF16.

## Phases and outputs

Supported phases are `prepare_data`, `teacher_train`, `teacher_cache`, `teacher_inference`,
`student_train`, `student_inference`, `compare`, and `all`. `all` executes them in the required
teacher-before-student order.

Outputs include:

```text
config_resolved.json
environment.json
dataset.json
model.json
features/{train,test}.pt
teacher_cache/{train,test}.pt
teacher_cache/{train,test}_shards/
metrics/teacher_inference.json
metrics/student_inference.json
metrics/student_training.json
metrics/student_training_history.csv
metrics/comparison.json
checkpoints/teacher_mlp.pt
checkpoints/student_mlp.pt
checkpoints/optical_surrogate.pt
```

See [RUN_COMMANDS.md](RUN_COMMANDS.md) for commands.
