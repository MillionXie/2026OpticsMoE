# 从这里开始

给老师发送、解压后直接运行的完整目录是：

```text
teacher_release_final/lgvq_temporal_08044/
```

固定权重位于：

```text
teacher_release_final/lgvq_temporal_08044/weights/best_checkpoint.pt
```

其 SHA-256 为：

```text
5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c
```

发布 ZIP 位于：

```text
releases/LGVQ_Temporal_08044_teacher_final_20260923.zip
```

## 固定权重复现

进入 `teacher_release_final/lgvq_temporal_08044/` 后运行：

```bash
python verify_release.py
python -I simulate.py --device cuda --fields 1 --output runs/smoke.json
python -I simulate.py --device cuda --fields 0 --output runs/full_test.json
```

最后一条命令使用包内权重和 35 个固定测试 field，评测 558 个视频。目标结果为
SRCC 0.8043868643。它不需要原始视频、Qwen 下载或外部仓库。

## 训练入口

训练代码入口为根目录的 `train.py`：

```bash
python train.py --config configs/temporal_16x4_s163.yaml --phase smoke
python train.py --config configs/temporal_16x4_s163.yaml --phase preflight
python train.py --config configs/temporal_16x4_s163.yaml --phase train
```

已从训练服务器恢复并放入展开目录的资产包括固定划分 manifest、train-only
teacher soft targets 和 9x4 warm-start checkpoint。两个原始 Qwen-front 缓存已不在
服务器上；完整重训练前需要按 `TRAINING_ASSETS.md` 从原始 LGVQ 视频与
Qwen3-VL-2B 重新生成。固定权重推理不依赖这两个训练缓存。
