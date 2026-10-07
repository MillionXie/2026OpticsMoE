# LGVQ Temporal 0.8044

本目录包含固定权重、35 个测试输入、独立推理代码和训练代码。

## 固定权重推理

环境：Python 3.11、PyTorch 2.6.0。

```bash
python -m pip install -r requirements.txt
python verify_release.py
python -I simulate.py --device cuda --fields 1 --output runs/smoke.json
python -I simulate.py --device cuda --fields 0 --output runs/full_test.json
```

完整测试使用 558 个视频，预期结果：

```text
SRCC  0.8022806420
KRCC  0.5931156844
PLCC  0.8154774079
RMSE  8.0496788
MAE   6.0566697
```

固定权重为 `weights/best_checkpoint.pt`。推理不需要原始 LGVQ 视频或 Qwen。

## 训练

```bash
python train.py --config configs/temporal_16x4_s163.yaml --phase smoke
python train.py --config configs/temporal_16x4_s163.yaml --phase preflight
python train.py --config configs/temporal_16x4_s163.yaml --phase train
```

训练配置包含可选的 CCD 噪声（偏置高斯和 shot），
正式 0.8044 配置中默认关闭。`router.noise_std` 只扰动训练期 Top-2 路由 logits，
不是 CCD 噪声，评测时关闭。完整重训练还需要从原始 LGVQ 和 Qwen3-VL-2B
生成以下两个缓存：

```text
assets/cache/qwen3vl_front_4f_49x1024_quality14.pt
assets/cache/qwen3vl_front_temporal_prompt_2048.pt
```
