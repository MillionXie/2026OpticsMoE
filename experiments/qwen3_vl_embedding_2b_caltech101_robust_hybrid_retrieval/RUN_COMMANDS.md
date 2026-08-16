# Caltech101 Robust Hybrid Retrieval 命令

以下命令均从仓库根目录 `/DATA/DATA1/guest3/2026OpticsMoE` 执行。正式流程是
101 类预训练 40 epoch，再选择固定的 10 类微调 20 epoch。不要重新运行当前已经开始的
101 类训练；等待它产生 `ema_last_checkpoint.pt` 即可。

## 1. 路径约定

101 类共享 Teacher cache：

```text
experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/cache/caltech101_all101_seed42_g3_train30_test20/teacher_embeddings.pt
```

10 类共享 Teacher cache：

```text
experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/cache/caltech101_target10_seed42_g3_train30_test20/teacher_embeddings.pt
```

cache 与 `runs/` 解耦。改变输出目录或重新训练不会重新执行 Qwen forward；10 类 cache
会直接从101类 cache 按图片路径切出，并严格检查模型、instruction、像素预算和 embedding
维度是否一致。

## 2. 检查101类预训练结果

当前101类训练结束后，固定评估 epoch-40 EMA：

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/configs/release/caltech101_robust_hybrid_moe4.yaml --phase evaluate --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/runs/caltech101_robust_hybrid_moe4/ema_last_checkpoint.pt
```

这一步只把101类模型当作阶段一预训练结果，不根据101类 test 选择后续 checkpoint。

## 3. 生成/验证10类共享 Teacher cache

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/configs/release/caltech101_target10_finetune.yaml --phase cache_teacher_embeddings
```

正常日志应包含 `Derived reusable target-10 cache without Qwen forward`。如果10类 cache
已经存在且身份一致，则打印 `Teacher cache already valid`。

## 4. 从 epoch 40 微调固定10类

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/configs/release/caltech101_target10_finetune.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/runs/caltech101_robust_hybrid_moe4/ema_last_checkpoint.pt
```

该命令训练额外20 epoch，对应绝对 epoch 41–60。10类阶段会在日志中显示每个 epoch 的
`test_top1` 和 `ema_test_top1` 以诊断过拟合，但正式硬件模型仍固定使用 epoch-60
`ema_last_checkpoint.pt`，不使用 test-selected checkpoint。

10类为：`airplanes`、`Motorbikes`、`Faces`、`Leopards`、`accordion`、
`grand_piano`、`scorpion`、`sunflower`、`watch`、`yin_yang`。

## 5. 固定评估10类模型

```bash
CUDA_VISIBLE_DEVICES=5 python -m experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/configs/release/caltech101_target10_finetune.yaml --phase evaluate --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/runs/caltech101_target10_finetune_epoch60/ema_last_checkpoint.pt
```

```bash
CUDA_VISIBLE_DEVICES=5 python -m experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/configs/release/caltech101_target10_finetune.yaml --phase visualize
```

## 6. 先做四平面纯模拟回放

在上光路之前验证 checkpoint、四层桥接和10类检索是否贯通：

```bash
CUDA_VISIBLE_DEVICES=5 python -m experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/configs/hardware/caltech101_target10_hardware.yaml --phase all_simulation --use-simulation
```

## 7. 生成实际光路文件

生成固定 gallery/query manifest、逐层 amplitude BMP、四张共享 phase BMP 和理论 CCD：

```bash
CUDA_VISIBLE_DEVICES=5 python -m experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/configs/hardware/caltech101_target10_hardware.yaml --phase prepare --artifact-profile full
```

默认选择每类3张 gallery 和10张固定 test query，共130张；首次光路测试不必播放全部
test。输出目录：

```text
experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/hardware_sessions/caltech101_target10_epoch60
```

SLM播放与相机采集仍使用独立的 `experiments/hardware_sdk/`。每层必须严格按照
`00_manifest/play_order.csv` 播放，把同 basename 的无损单通道 CCD 文件放进当前层的
`ccd_captured/`；不要使用 JPEG。

## 8. 逐层处理真实 CCD

每完成一层采集后执行对应命令，电网络会产生下一层需要播放的 amplitude：

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/configs/hardware/caltech101_target10_hardware.yaml --phase process_vision_expert
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/configs/hardware/caltech101_target10_hardware.yaml --phase process_vision_global
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/configs/hardware/caltech101_target10_hardware.yaml --phase process_language_expert
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/configs/hardware/caltech101_target10_hardware.yaml --phase process_language_global
```

最终硬件指标：

```text
hardware_sessions/caltech101_target10_epoch60/05_retrieval/metrics.json
hardware_sessions/caltech101_target10_epoch60/05_retrieval/retrieval_results.csv
hardware_sessions/caltech101_target10_epoch60/05_retrieval/confusion_matrix.csv
```

## 9. 测试

```bash
python -m pytest -q experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/tests
```
