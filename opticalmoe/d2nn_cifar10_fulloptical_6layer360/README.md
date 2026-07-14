# CIFAR-10 全光 6 层 D2NN baseline（4 类 / 10 类）

## 20 cm 层间光电转换对照

`configs/cifar10_4class_optoelectronic_interlayers_20cm.yaml` 保留六张360×360相位板，但在前五张相位板后依次执行：传播20 cm、平方探测、无仿射空间 LayerNorm、ReLU、将非负实数作为下一张相位板的输入振幅。第六张相位板后传播20 cm并直接进行最终平方探测和原 detector-plane loss。该对照不增加可训练电子参数。

这是与 OpticalMoE 对照的独立纯光实验。模型中没有 router、prompt、专家路由、CNN、MLP 或其他电子分类层；探测器区域积分就是最终分类输出。

## 光路

```text
CIFAR-10 RGB
-> grayscale amplitude
-> bicubic resize 32x32 到 300x300
-> 居中 zero pad 到 360x360
-> 360x360 相位板 1
-> 5 cm 自由空间传播
-> 360x360 相位板 2
-> ...
-> 360x360 相位板 6
-> 10 cm 传播到 detector
-> 4 或 10 个方形 detector 区域积分
-> 分类结果
```

默认输入直接加载到第一张相位板，即 `input_to_layer_distance_m: 0.0`。所有相位板使用 `2π * sigmoid(raw_phase)` 约束，默认零初始化、无 phase dropout。波长为 532 nm，仿真像素尺寸为 16 μm。

可训练参数只有 6 张完整相位板：

```text
6 x 360 x 360 = 777,600 optical phase parameters
electronic trainable parameters = 0
```

## 数据采样与 batch

`train_samples_per_class` 和 `test_samples_per_class` 是整个数据集的每类数量上限。设为 `null` 时保留所选类别的所有数据。它们不能用来控制单个 batch。

`batch_size` 才是每次 optimizer step 使用的样本数。训练 DataLoader 会打乱当轮样本。提高 batch size 通常能提高 GPU 吞吐率，但不会直接减少每个 epoch 的总样本数，也不一定按相同比例减少光学 FFT 总计算量。

若希望缩短单个 epoch，同时长期覆盖完整数据集，使用 `train_samples_per_class_per_epoch`。例如十分类配置默认每类每轮取 1000 张，即每轮 10000 张；采样器按类别在完整 5000 张中轮转，约 5 个 epoch 覆盖全部数据，再重新打乱。各类别当轮样本合并后还会再次全局打乱，不会按类别连续组成 batch。

## Detector 布局

- 四分类：中心附近紧凑的 2×2、每区 40×40。
- 十分类：中心附近紧凑的三行 3/3/4、每区 30×30；不同行使用独立 x 起点，确保每行水平居中。

两种布局的几何中心都是 `(179.5, 179.5)`，与 360×360 光场中心一致。

训练输出保存在本工程的 `runs/` 下，包括配置、环境、数据统计、模型参数报告、best/last checkpoint、训练曲线、混淆矩阵、预测 CSV、相位图、逐层光强和 detector 柱状图。
