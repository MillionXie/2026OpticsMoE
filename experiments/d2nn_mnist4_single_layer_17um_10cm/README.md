# MNIST-4 单层 10 cm 光学识别

这是与新双 SLM 光路一致的独立手写数字工程，只识别 MNIST `0/1/2/3`。它没有 Qwen、MoE、token mixer 或电子分类头；唯一可训练参数是 `478×478` 的相位张量。

## 物理结构

```text
MNIST 振幅 400×400
→ 居中放入 478×478 有效区域
→ 17 µm 振幅 SLM（255=白/透光，0=黑/遮光）
→ 4F 共面映射到相位 SLM
→ phase = 2π·sigmoid(raw_phase)，raw_phase 从全 0 开始
→ 10 cm 角谱传播（532 nm，17 µm 逻辑采样）
→ 中心 478×478 CCD ROI
→ 四个 49×49 积分探测区
→ argmax 得到 0/1/2/3
```

真实物理画布是 `518×518`。正式仿真在外部补零到 `1024×1024` 后传播，再裁回物理画布，用于减轻 10 cm FFT 周期回绕；这不会改变导出的 `478×478` 有效相位或硬件 ROI。

## 损失审核

参考文件为 `opticalmoe/notebooks/github_D2NN_mnist4.ipynb`。其正式训练代码是：

```text
out_img = |E_detector|²
det_labels = 正确类别探测区为1、其余位置为0的目标图
loss = 100 × mean((out_img - det_labels)²)
```

本工程默认使用同一类整幅探测面 MSE：

```text
loss = 1.0 × template_mse + 0.0 × detector_ce
template_mse = 100 × MSE(CCD intensity, binary detector template)
```

`detector_ce` 只作为诊断量记录，不参与正式训练。没有教师蒸馏，也没有额外的电子分类损失。

## 训练与硬件合同

- 波长：532 nm。
- 相位到 CCD：10 cm。
- 相位参数化：`2π·sigmoid(raw_phase)`；初始 `raw_phase=0`，实际均匀相位为 π。
- 优化器：Adam；初始相位学习率 `0.01`，与 notebook 一致。
- 振幅：1024×1024、17 µm，中心 `(512,512)`，不反相。
- 相位：1920×1200、8 µm，中心 `(980,590)`，导出前纵向翻转。
- CCD：由四焦点标定得到物理 ROI，再保存成 478×478、8-bit 灰度图；不做虚构的背景扣除。

硬件导出分为两组：

- `demo_topk`：每类 10 张容易且边界清晰的样本，仅用于演示，不得汇报为无偏准确率。
- `formal_fixed_random_100_per_class`：每类固定随机 100 张，选择过程不读取预测与正确性，用于正式硬件准确率。

完整命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)，实验室采集说明见 [HARDWARE_LAB.md](HARDWARE_LAB.md)。
