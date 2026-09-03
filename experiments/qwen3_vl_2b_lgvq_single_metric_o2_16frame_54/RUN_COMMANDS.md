# 从缓存到训练的唯一顺序

所有命令从仓库根目录执行。Spatial 与 Temporal 分开执行，不能互换 language
cache、checkpoint 或 mask。

## 0. 先确认路径

正式服务器数据集预期为：

```text
/DATA/DATA1/lixinyue/xyli/data/LGVQ
```

准备服务器本地的 `Qwen3-VL-2B-Instruct` 绝对目录。缓存命令通过
`--model-path` 显式接收该目录；不允许在正式任务中临时联网下载。

若使用主服务器的四卡并行方案，优先直接遵循
[SERVER_RUNBOOK.md](SERVER_RUNBOOK.md)。下面保留逐条命令，便于单步排错。

## 1. 静态与小尺寸数值检查

```bash
python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 \
  --config experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial.yaml \
  --phase smoke
```

## 2. 缓存 Spatial 的 Qwen 前端

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.cache_qwen_front \
  --config experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial.yaml \
  --model-path /ABSOLUTE/PATH/TO/Qwen3-VL-2B-Instruct \
  --batch-size 2 --chunk-rows 16 --device cuda
```

这一步生成共享 Vision+quality 缓存，以及 Spatial 专属 prompt 缓存。采用可恢复
分片；中断后重复相同命令会继续。共享 Vision 文件约 4.2 GiB，不要复制第二份。

## 3. 只生成 Temporal 专属文本缓存

重复缓存命令并换 Temporal 配置：

```bash
CUDA_VISIBLE_DEVICES=1 python -m \
  experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.cache_qwen_front \
  --config experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/temporal.yaml \
  --model-path /ABSOLUTE/PATH/TO/Qwen3-VL-2B-Instruct \
  --batch-size 2 --chunk-rows 16 --device cuda
```

程序会核验并复用完全相同的 Vision 缓存，只重建 target-specific language cache。

## 4. 正式 preflight

```bash
python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 \
  --config experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial.yaml \
  --phase preflight

python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 \
  --config experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/temporal.yaml \
  --phase preflight
```

只有两个报告均为 `status=ready` 才能训练。

## 5. Spatial 与 Temporal 独立训练

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 \
  --config experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial.yaml \
  --phase train

CUDA_VISIBLE_DEVICES=1 python -m \
  experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 \
  --config experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/temporal.yaml \
  --phase train
```

后台运行时请分别重定向日志，不要让两个目标写入同一 output directory。

主服务器还提供 Spatial seed 43 和 Spatial 强鲁棒候选，但它们只是寻找更稳定
Spatial 权重的训练候选，不是新的推理结构，也不是 Top-k 消融。四项并行启动命令见
`SERVER_RUNBOOK.md`。

## 6. 结果文件

每个任务目录独立包含：

```text
best_observed_test_checkpoint.pt
last_checkpoint.pt
metrics_best_observed_test_optical_on.json
test_metrics_optical_on.json
test_metrics_optical_off.json
optical_contribution_same_checkpoint.json
test_predictions_optical_on.csv
test_predictions_optical_off.csv
phase_training_diagnostics.json
parameter_breakdown.json
resolved_config.json
training_summary.json
```

其中 `test_metrics_optical_off.json` 是同一个最佳光电 checkpoint 旁路四层光得到的
公平消融，不是单独训练的纯电子模型。

训练结束后使用 [EXPORT_COMMANDS.md](EXPORT_COMMANDS.md) 导出六阶段相位预览、
1920×1200 相位 SLM BMP 和 1024×1024 振幅布局检查图。正式逐样本振幅不能用布局
检查图代替。
