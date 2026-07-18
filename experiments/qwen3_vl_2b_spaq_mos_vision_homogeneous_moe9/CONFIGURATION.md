# Configuration

The main config is grouped by dataset, Qwen runtime, batching, teacher cache, optical MoE, loss, optimizer, training, regularization, and visualization.

- `spaq_mos.json`: full experiment; phase dropout starts at epoch 5.
- `spaq_mos_nodropout.json`: same experiment without phase dropout.
- `spaq_mos_batch1.json`: conservative one-image-per-batch variant.
- `spaq_mos_smoke.json`: 24 train / 12 test images and one epoch.

The three derivative configs use `base_config` so only intentional differences are repeated.

`train_samples_per_epoch=null` traverses the full retained train set each epoch. Setting an integer uses deterministic rotating windows: the stored train set is not deleted, and later epochs rotate through it.

`classification_head.output_activation` is the single activation setting shared by the teacher and student structures; it defaults to `linear`. Teacher and student parameters are trained independently, and the student head is always freshly initialized. `optimizer.student_head_learning_rate` defaults to `0.001`, separate from the optical surrogate and router learning rates.

Changing an old Sigmoid run to linear does not require rebuilding the frozen Qwen teacher feature cache. It does require rerunning `teacher_train` and `teacher_predictions`. Prediction-cache metadata prevents silently reusing scores produced by an incompatible head.

`training.logging.interval_batches=1500` limits rolling console output. Each rolling line includes raw router balance/importance losses and their weighted contributions to the total loss.
