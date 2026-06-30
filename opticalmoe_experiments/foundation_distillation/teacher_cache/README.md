# Teacher Cache

Generated CLIP or DINOv2 image features are stored here and should normally remain uncommitted. Each dataset cache contains `train_features.pt`, `val_features.pt`, `test_features.pt`, and `metadata.json`.

The teacher input is always the student's grayscale image replicated to three channels. No RGB-only information, text features, or classification logits are cached. Both teacher types use the same split payload format and store L2-normalized image features.
