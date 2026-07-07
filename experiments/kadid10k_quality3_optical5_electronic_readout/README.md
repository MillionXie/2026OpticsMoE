# KADID-10k Quality-3 optical5 + electronic readout baseline

This is the non-Qwen baseline for `qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual`.

```text
RGB image -> grayscale -> RMS normalization -> 64 x 64 optical field
          -> five optical intensity conversions
          -> class-region detector
          -> two small electronic convolutions + average pooling + MLP
          -> high_quality / medium_quality / low_quality
```

Every optical layer applies a trainable phase mask, padded angular-spectrum propagation, square-law detection, per-sample intensity normalization, and detector ReLU. Detected intensity is passed directly to the next layer; it is not square-rooted back to amplitude.

The electronic tail is deliberately small: two convolutions with channels `[16,32]`, ordinary adaptive average pooling, and one hidden MLP layer. It does not use Qwen, a tokenizer, a Transformer, teacher/student distillation, or cached features.

The experiment imports the same KADID CSV parser, score-tertile labels, and reference-disjoint train/validation/test split as the Qwen experiment, so accuracy is directly comparable. Default optical geometry also matches the Qwen experiment: active field 64, padded field 128, pixel pitch 8 micrometres, zero phase initialization, and no amplitude masks.

Loss:

```text
L = L_classification + 1.0 * L_detector_region + 0.1 * L_energy_concentration
```

Outputs include training history, best/last checkpoints, full quality metrics, prediction metadata, confusion matrix, phase masks, per-layer light fields, detector fields, and fixed region layout.
