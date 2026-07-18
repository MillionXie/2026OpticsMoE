# Configuration

The main config is grouped by dataset, Qwen runtime, batching, teacher cache, optical MoE, loss, optimizer, training, regularization, and visualization.

- `spaq_mos.json`: full experiment; phase dropout starts at epoch 5.
- `spaq_mos_nodropout.json`: same experiment without phase dropout.
- `spaq_mos_batch1.json`: conservative one-image-per-batch variant.
- `spaq_mos_smoke.json`: 24 train / 12 test images and one epoch.

The three derivative configs use `base_config` so only intentional differences are repeated.

`train_samples_per_epoch=null` traverses the full retained train set each epoch. Setting an integer uses deterministic rotating windows: the stored train set is not deleted, and later epochs rotate through it.

`classification_head.output_activation` supports `sigmoid` and `linear`. It must be identical for teacher and student; changing it requires a new output directory and retraining the teacher head.
