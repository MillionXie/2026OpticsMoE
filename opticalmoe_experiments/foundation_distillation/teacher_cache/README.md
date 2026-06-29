# Teacher Cache

Generated CLIP image features are stored here and should normally remain uncommitted. Each dataset cache contains `train_features.pt`, `val_features.pt`, `test_features.pt`, and `metadata.json`.

The teacher input is always the student's grayscale image replicated to three channels. No RGB-only information and no CLIP text features are cached.

