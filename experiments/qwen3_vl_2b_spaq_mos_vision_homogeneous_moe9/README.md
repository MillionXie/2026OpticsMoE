# Qwen3-VL-2B SPAQ MOS Vision Homogeneous Optical MoE

This experiment transfers the verified CIFAR-10 homogeneous optical-MoE vision replacement to single-task SPAQ MOS regression. It does not use the Qwen language model, tokenizer, chat template, or prompt.

本实验只预测 SPAQ 的整体质量分数 `MOS`。选择 MOS 而不是 Brightness，是因为 MOS 是最标准的整体 IQA 指标，适合先验证完整 Qwen 电子视觉栈被光学 MoE 替换后的性能。

## Teacher and student

Teacher:

```text
RGB image
-> Qwen image processor
-> frozen Qwen vision patch embedding
-> complete frozen electronic Qwen vision transformer stack
-> valid-token mean pooling
-> LayerNorm(1024) -> Linear(1024,1)
-> linear normalized MOS prediction
```

Student:

```text
RGB image
-> same Qwen image processor and frozen patch embedding
-> Linear(1024,120) -> LayerNorm -> Softplus
-> one 120x120 field per image
-> input-dependent top-3 homogeneous optical MoE (9 experts, 5 phase layers)
-> global phase -> 20 cm propagation -> 480x480 detector
-> AvgPool 480->120 -> non-affine LayerNorm -> ReLU
-> valid token rows -> Linear(120,1024)
-> valid-token mean pooling
-> a freshly initialized LayerNorm(1024) -> Linear(1024,1)
-> linear normalized MOS prediction
```

The source images remain RGB. No grayscale conversion is performed. Batch samples are split using Qwen `cu_seqlens`, so one image always maps to one independent optical field.

## Training order

1. `teacher_precompute` caches the complete electronic vision-stack output for every retained train/test image.
2. `teacher_train` trains only the small MOS regression head on cached electronic features.
3. `teacher_predictions` caches the teacher MOS prediction used for score distillation.
4. `student_train` creates a fresh student head and jointly trains it with the optical MoE, adapters, and router. No teacher head parameters are copied.
5. `student_inference` evaluates the best optical student.

For CLI compatibility with the CIFAR classification source, `--phase teacher_logits` is accepted as an alias of `teacher_predictions`; the cached value is a scalar MOS prediction, not categorical logits.

Student loss:

```text
L = hidden_weight * LayerNorm-hidden MSE
  + prediction_distill_weight * SmoothL1(student_score, teacher_score)
  + regression_weight * SmoothL1(student_score, true_MOS)
  + router_balance_weight * router_balance
  + router_importance_weight * router_importance
```

MOS labels are divided by 100 during training and restored to 0-100 for MAE/RMSE. Evaluation reports MAE, RMSE, SRCC, PLCC, and within-5/within-10 accuracy.

Teacher and student use the single `classification_head.output_activation` setting, which defaults to `linear`. Their structures are identical but their parameters are independently initialized and trained. Predictions are not silently clamped during training or evaluation; SmoothL1 directly pulls the linear output toward normalized targets. The student head uses its own lower learning rate. Switching an older Sigmoid run to this configuration requires rerunning only `teacher_train` and `teacher_predictions`; the expensive frozen Qwen feature cache remains reusable.

## Checkpoint selection warning

To remain consistent with the current CIFAR-10 source experiment, the student is evaluated on the test split after every epoch and the default best checkpoint maximizes test SRCC. This makes the reported best-test result selection-biased. The `last` checkpoint is also always saved and should be used for a strict fixed-epoch report.

SPAQ is automatically downloaded when `download=true`, or `data_root`, `annotations_file`, and `image_dir` can point to an existing layout.
