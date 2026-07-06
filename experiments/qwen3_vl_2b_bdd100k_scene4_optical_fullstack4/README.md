# Qwen3-VL-2B BDD100K Scene-4 Optical Fullstack4

This experiment adapts the Qwen3-VL-2B full vision-and-language optical4 distillation pipeline to BDD100K scene classification:

```text
highway
city_street
residential
other
```

The teacher is the full electronic Qwen3-VL-2B multimodal model plus an MLP classifier. The student preserves the processor, chat template, tokenizer, patch embedding, vision merger, token embedding, final language norm, answer-position feature, and MLP, while replacing the complete vision transformer stack and complete language decoder stack with four optical conversions each.

The optical configuration uses 64×64 effective fields, 128×128 padded propagation, 8 µm pixel pitch, 532 nm wavelength, 5 cm propagation distance, and zero-initialized phase masks. There is no electronic residual bypass through either replaced transformer stack.

Data preparation reads `attributes.scene`, creates an ImageFolder tree with links to the existing BDD100K images, writes `scene4_manifest.json`, and generates `scene4_dataset_report.md`. See [BDD100K_SCENE4_DATASET_REPORT.md](BDD100K_SCENE4_DATASET_REPORT.md) for the audited mapping and counts.

The `other` class is extremely small. The main configuration uses balanced per-epoch sampling with replacement for minority classes. Validation and test retain their natural class distributions, and conclusions should prioritize macro-F1, balanced accuracy, and per-class recall.

Teacher stack outputs are cached once. Student training reads cached teacher vision outputs, answer hidden states, and teacher logits; it does not run the teacher online.
