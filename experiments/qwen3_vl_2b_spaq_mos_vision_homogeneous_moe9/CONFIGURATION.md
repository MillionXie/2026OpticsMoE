# Configuration

The main config is grouped by dataset, Qwen runtime, batching, teacher cache, optical MoE, loss, optimizer, training, regularization, and visualization.

- `spaq_mos.json`: full experiment; phase dropout starts at epoch 5.
- `spaq_mos_nodropout.json`: same experiment without phase dropout.
- `spaq_mos_batch1.json`: conservative one-image-per-batch variant.
- `spaq_mos_smoke.json`: 24 train / 12 test images and one epoch.

The three derivative configs use `base_config` so only intentional differences are repeated.

`train_samples_per_epoch=null` traverses the full retained train set each epoch. Setting an integer uses deterministic rotating windows: the stored train set is not deleted, and later epochs rotate through it.

`classification_head.output_activation` controls the teacher and remains `sigmoid` for compatibility with the trained teacher checkpoint.

`classification_head.student_output_activation` independently controls the student and defaults to `linear`. `classification_head.student_zero_initialize_regressor=true` copies the teacher LayerNorm but zeros only the student's final Linear weight and bias. It deliberately does not zero LayerNorm. `optimizer.student_head_learning_rate` defaults to `0.001`, separate from the optical surrogate and router learning rates.

Changing only these student fields does not require rebuilding the teacher feature cache or retraining `teacher_head.pt`. Old Sigmoid student checkpoints are rejected rather than silently interpreted as linear checkpoints.
