# 10 cm / 17 µm 四层光电检索结果

## 结果口径

- 数据：Caltech101 固定 10 类；train 2625、gallery 30、test query 200。
- 训练：Qwen3-VL-Embedding-2B 主干冻结；电子 Mixer、四组光学 phase、两个
  MoE4 router、CCD readout、四个有下限融合门和 64 维检索头联合训练。
- 光路：532 nm、传播 10 cm、逻辑像素 17 µm、数值面 518×518、有效 CCD
  478×478；输入/phase/CCD 三路独立 ±16 pixel 扰动，并启用 k-space 限带。
- 正式 checkpoint：`ema_best_train_loss_checkpoint.pt`，epoch 27；只按最小训练
  loss 选择，test 没有参与选模。

固定 checkpoint 的仿真结果：

| 指标 | 数值 |
|---|---:|
| Top-1 | 0.6050 |
| Top-3 | 0.8450 |
| MRR | 0.7389 |
| Query / gallery images | 200 / 30 |
| Trainable parameters | 2,683,709 |

训练过程中观察到的 live-weight 最高 Top-1 为 0.6150（epoch 11），但该值使用
test 反复观察，具有选模偏差，只作诊断上界，不作为正式结果。

## 光学训练审计

epoch 27 的汇总 phase std 为 0.10824 rad；相对 epoch 7 恢复点的 RMS 位移为
0.08358 rad。所有 10 个 raw phase plane 均有有限梯度，无 saturation、NaN、Inf、
OOM 或 missing gradient。

| 物理阶段 | phase std (rad) | 相对恢复点 RMS (rad) |
|---|---:|---:|
| Vision expert | 0.12538 | 0.09816 |
| Vision global | 0.12475 | 0.09761 |
| Language expert | 0.08536 | 0.06477 |
| Language global | 0.09079 | 0.06758 |

最终四个融合系数约为 0.1999、0.1998、0.1993、0.1993；配置中的 0.10 是
`electronic + alpha * optical_delta` 的系数下限，不等同于光能量占比。最终训练轮
Vision / Language router 均覆盖 4/4 专家。

## 硬件产物

正式 checkpoint SHA-256：

```text
5bf4337d7a173aada16fe9011e591ee13983d7c834db02b90cb46e58f2d67d5b
```

四张 1920×1200 phase BMP 位于服务器 run 目录的
`hardware_phase_export/phase_bmp/`。它们使用 phase SLM 中心 `(980,590)`，并在
栅格化前执行竖直翻转。

| 文件 | SHA-256 |
|---|---|
| `vision_expert.bmp` | `681e469970ed64050983982cdb6181e6d32bd1491b41930d63bf339806b202b4` |
| `vision_global.bmp` | `e13f05b43f742818d3d1dd09a8f8094572d3f716c8e57d001e7ef7760b602bd1` |
| `language_expert.bmp` | `27c970a6ec93589b7b2a5e9f414cb250fafa09e65145a93cfc1925b601b1ef34` |
| `language_global.bmp` | `d8d13f1cba2c269ff17bc52a33368e7f2201a5cc8e84526bd75f6e2489754bc2` |

快速第四层 session 已导出 210 张 478×478 uint8 紧凑振幅（每类 10 train、
10 test、1 gallery）以及唯一的 `language_global.bmp`。它只代表“前三层仿真 +
第四层实测”，不能替代四层正式实测。

实验室 ZIP：

```text
runs/caltech101_four_layer_moe4_joint_17um_10cm_robust/
qwen_caltech101_10cm_quick210_lab_bundle.zip
```

- 大小：81,002,308 bytes
- 文件：385 个
- ZIP SHA-256：
  `3268ea40e70410ac8f069dbaa029fbbb504f5504e6ea1afef6a51af4b32a4cf3`
- `unzip -t`：通过，无损坏文件。
- 包含：唯一正式 checkpoint、四层 phase、quick210、硬件代码、TUCam 与
  Meadowlark vendor SDK、最小环境、训练指标/日志白名单和逐文件 SHA manifest。
- 排除：Caltech101 原图、模型 cache、CCD、1024×1024 中间振幅、额外 checkpoint。

完整播放、采集、上传和逐层/快速微调步骤见 [LAB_BUNDLE.md](LAB_BUNDLE.md) 与
[RUN_COMMANDS.md](RUN_COMMANDS.md)。
