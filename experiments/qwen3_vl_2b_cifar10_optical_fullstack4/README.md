# Qwen3-VL-2B CIFAR-10 Optical Fullstack4

This experiment uses Qwen3-VL-2B and CIFAR-10. The teacher is the complete electronic Qwen3-VL-2B multimodal model plus an MLP classifier. The student keeps the Qwen processor, chat template, tokenizer, vision patch embedding and merger, token embeddings, final language norm, answer-position extraction, and MLP interface.

The student simultaneously replaces **all vision transformer blocks** and **all language decoder blocks**. The complete vision stack is compressed into four optical conversions, and the complete language stack is independently compressed into four optical conversions. This is stack-level distillation; it no longer distills each four-block group. This directory implements only the vision-and-language both-optical4 experiment and no ablations.

Each surrogate contains one LayerNorm/input adapter, four consecutive optical conversions, and one final output adapter. There are no electronic adapters between conversions and no electronic residual bypass. Optical conversions pass detected, normalized intensity directly to the next conversion. The training forward never takes `sqrt(intensity)`.

Teacher precomputation stores only sample metadata, the full teacher vision-stack output, and the teacher answer-position hidden representation. It does not store stack inputs, group inputs, or intermediate transformer block outputs. Cache metadata is validated and stale caches must be deleted explicitly.

The main CIFAR-10 config keeps `train_limit_per_class=null` and `train_samples_per_class_per_epoch=null`, so every epoch covers the complete stratified training split. Student batches use an epoch-aware class-mixed sampler: indices are reshuffled every epoch, classes are interleaved within batches, and cache-shard locality is retained. A multi-shard LRU prevents mixed batches from repeatedly reloading the same teacher-cache files.

`teacher_train` reads cached answer hidden states and never reruns Qwen. `teacher_logits` derives lightweight logits from the same cache. Student training reads all teacher targets from disk and never runs an online teacher. Training history, latest status, best validation metrics, validation predictions, and checkpoints are written immediately after every epoch to avoid opaque long runs.

The default CIFAR-10 location is `experiments/qwen3_vl_2b_cifar10_optical_fullstack4/data/cifar10`, and torchvision can download it automatically.
