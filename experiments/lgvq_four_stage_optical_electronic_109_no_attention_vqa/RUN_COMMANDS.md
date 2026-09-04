# 唯一正式运行流程

> **作用域：完整服务器训练仓库。** 本文件记录仿真训练来源，不是实验室 ZIP 的执行入口；
> 它需要原始 LGVQ 视频和训练期软目标。实验室导出、采集、校验和逐层微调必须改看
> 压缩包根目录 `README_FIRST.md` / 本工程 `LAB_DEPLOYMENT.md`。

从仓库根目录执行。正式工程与唯一推荐配置分别为：

```text
experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa
experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/configs/release/formal_alpha50_kd300_center100.yaml
```

本工程只训练一次正常光电融合模型。训练结束后，程序自动加载同一个最佳 checkpoint 做两次测试：

1. 保留全部光路的正常光电推理；
2. 旁路全部光路的同 checkpoint 去光推理。

第二项只用于衡量同一组已训练权重对光学分支的依赖程度。它没有单独训练、微调或重新优化电子分支，也不是另一份纯电子 checkpoint。本工程没有“去光训练”命令。

## 1. 代码自检

```bash
python -m unittest \
  experiments.lgvq_four_stage_optical_electronic_109_no_attention_vqa.tests.test_core \
  -q
```

## 2. 首次准备数据

服务器已有这两个文件时不要重复生成：

```text
experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv
experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_four_frames_center100_224_uint8.pt
```

缺少 manifest 时执行：

```bash
python -m experiments.lgvq_four_stage_optical_electronic_109_no_attention_vqa.prepare_manifest \
  --dataset-root /DATA/DATA1/lixinyue/xyli/data/LGVQ \
  --output experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv \
  --seed 42
```

缺少 4 帧缓存时执行：

```bash
python -m experiments.lgvq_four_stage_optical_electronic_109_no_attention_vqa.cache_frames \
  --manifest experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv \
  --output experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_four_frames_center100_224_uint8.pt \
  --frame-size 224 \
  --crop-fraction 1.0
```

推荐配置还使用仅含 2250 个训练样本的二维标量软目标：

```text
experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/training_only_teacher_predictions.pt
```

该文件只参与训练 loss。推理时不加载教师、Qwen、Transformer 或任何软目标文件。

## 3. 正式预检

```bash
python -m experiments.lgvq_four_stage_optical_electronic_109_no_attention_vqa \
  --config experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/configs/release/formal_alpha50_kd300_center100.yaml \
  --phase preflight
```

预检必须显示：零外部模型模块、零冻结参数、光路由 Top-2、109 px 专家，以及无 Qwen、Transformer、attention、mixer 或 block。

## 4. 唯一一次正式训练：正常光电融合

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -m experiments.lgvq_four_stage_optical_electronic_109_no_attention_vqa \
  --config experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/configs/release/formal_alpha50_kd300_center100.yaml \
  --phase train
```

固定协议为 2250 个训练视频、558 个测试视频、无验证集、100 epoch、batch 64；每 5 epoch 在 558 个测试视频上评估一次，按 Spatial/Temporal 平均 SRCC 选择最佳权重。训练期间的模型调用始终启用光学路径。

正式输出目录：

```text
experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/runs/lgvq_oeo109_alpha50_kd300_center100_v3
```

## 5. 重算同一最佳 checkpoint 的开光/去光结果

训练结束时会自动执行本步骤。需要手工复算时使用：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -m experiments.lgvq_four_stage_optical_electronic_109_no_attention_vqa \
  --config experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/configs/release/formal_alpha50_kd300_center100.yaml \
  --phase evaluate \
  --checkpoint experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/runs/lgvq_oeo109_alpha50_kd300_center100_v3/best_observed_test_checkpoint.pt
```

只看以下文件即可：

```text
test_metrics_optical_on.json
test_metrics_optical_off.json
optical_contribution_same_checkpoint.json
phase_training_diagnostics.json
test_predictions_optical_on.csv
test_predictions_optical_off.csv
```

`test_metrics_optical_off.json` 不是纯电子模型的训练结果，而是同一正常光电模型、同一最佳权重在推理时旁路光路的结果。不得把它写成“单独训练的电子基线”。当前正式指标及证据说明见 [RESULTS.md](RESULTS.md)。
