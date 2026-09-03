# 从缓存到训练的唯一顺序

所有命令从仓库根目录执行。Spatial 与 Temporal 分开执行，不能互换 language
cache、checkpoint 或 mask。

## 0. 先确认路径

正式服务器数据集预期为：

```text
/DATA/DATA1/lixinyue/xyli/data/LGVQ
```

编辑对应 YAML 中的 `initialization.qwen_model_path`，令其指向服务器本地的
`Qwen3-VL-2B-Instruct` 目录；不允许在正式任务中临时联网下载。

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

## 5. 两张 GPU 独立训练

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
