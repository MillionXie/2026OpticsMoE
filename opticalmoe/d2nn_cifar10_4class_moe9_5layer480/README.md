# CIFAR-10 四分类纯光 OpticalMoE（9 experts × 5 layers）

本实验以 `d2nn_mnist4_amp_1phase400` 的角谱传播、相位约束、探测积分和训练方式为物理基线，只把中间光路组织成 3×3 OpticalMoE。任务使用 CIFAR-10 的前四类：airplane、automobile、bird、cat。

## 光路结构

```text
灰度振幅图像 100×100
-> zero pad 到 120×120，并置于 480×480 中心
-> 3×3 routing amplitude + 单张连续 global prompt phase
-> 4f/global convolution 产生 9 路专家入口光场
-> 9 个专家，每个专家 120×120、5 层相位调制，pitch=150
-> 450×450 global FC phase
-> 4 个 50×50 detector 区域
-> 四分类
```

除输入相关 top-k router 外，不含电子分类 readout。router 为 `AdaptiveAvgPool2d(10×10) -> Linear(100,9) -> softmax -> top-k`，共有 909 个电子参数。纯光相位参数为：

```text
experts: 9 × 5 × 120 × 120 = 648,000
global FC: 450 × 450 = 202,500
total optical phase parameters = 850,500
```

## Prompt 的唯一正确含义

Prompt 只有两张物理图：

1. `prompt_amplitude`：450×450 有效面被划分为 3×3 个 150×150 区域。每个区域内是一个严格均匀的 routing amplitude；未选专家区域为 0。
2. `prompt_phase`：在完整 480×480 全局坐标上只计算一次的连续二次相位，有效范围为中心 450×450。它不是九张局部透镜/光栅相位的拼接，不随 routing weights 改变。

全局二次相位在任意一个 cell 中以该 cell 中心重写时，可以严格分解为“局部二次透镜项 + 线性载频/光栅项 + 常数项”。因此代码不需要、也不允许为九个 cell 重设九次坐标原点；此前相位图中的明显 3×3 拼接正是这种错误做法造成的。

专家入口采用：

```python
fftshift(ifft2(fft2(flip(input)) * fft2(prompt_transmission)))
```

即全局 4f/convolution fan-out，而不是 `ASM -> 中心入射场逐点乘 prompt -> ASM`。后者无法让九个空间振幅区域成为九条独立路由。离散网格标定后的默认焦距为 `f=7.22 cm`，匹配卷积距离为 `2f=14.44 cm`。名义 `f=7.5 cm` 会让外围复制像产生约 11 像素的径向位置误差；7.22 cm 将多样本实测的九路相对落点误差压到 0.14 像素以内，同时保持单路目标能量大于 97%、九路等权能量差小于 0.01。

训练和验证只把下面两张图称为 prompt：

```text
prompt_amplitude.png
prompt_phase.png
```

不会再生成或使用 `combined_router_transmission_amplitude`。复数 transmission 只是上述振幅和相位在计算中的组合，不是第三张可加载的光学图。

## 几何参数

| 项目 | 数值 |
|---|---:|
| simulation canvas | 480×480 |
| prompt/expert active size | 450×450 |
| outer zero padding | 15 pixels |
| input | 100×100 -> zero pad 120×120 |
| experts | 3×3 |
| expert size | 120×120 |
| expert pitch | 150 pixels |
| wavelength | 532 nm |
| simulation pixel pitch | 16 µm |
| global fan-out focal length | 7.22 cm |
| global convolution distance | 14.44 cm |
| expert inter-layer distance | 5 cm |
| global FC to detector | 10 cm |

## Routing 验证

训练前运行：

```bash
CUDA_VISIBLE_DEVICES=0 python validate_routing.py --config configs/config.yaml --device cuda --smoke-test --output-dir routing_validation
```

脚本绕过电子 router，依次强制 one-hot、multi-hot 和 9 路等权 routing，保存每种情况下唯一的 amplitude/phase 图、专家入口光强和能量柱状图。`routing_validation*/` 已加入 `.gitignore`。

验证报告还会记录九个复制像的局部质心对齐误差，以及从专家入口到第五层的逐层质心漂移。默认要求复制像位置范围不超过 1 pixel、五层累计漂移不超过 1 pixel；这样可以同时验收能量、空间落点和近似平行传播。

## 相位训练诊断量

Phase diagnostics 现在默认完全关闭：

```yaml
training:
  phase_diagnostics_enabled: false
```

关闭时不会 clone 相位参数，不会拼接首 batch 梯度，也不会计算或打印 `phase_std / phase_delta_mean / grad_mean`。如临时排查相位不更新，可改为 `true`；这些量只用于诊断，不参与 loss，也不会改变 Adam 学习率：

- `phase_std`：所有有效相位的标准差，单位 rad；描述相位板的空间起伏，不代表训练速度。
- `phase_delta_mean`：本 epoch 内 `raw_phase` 的平均绝对改变量。`1.115e-02` 表明参数确实在更新。
- `grad_mean`：第一个 batch 中全部 `raw_phase` 梯度绝对值的平均值。模型共有 850,500 个相位像素，loss 还会对像素求平均，因此 `1e-6` 量级不异常，应结合 `phase_delta_mean`、梯度非零比例和准确率曲线判断。

默认关闭后，上述统计没有任何训练时计算开销。

注意：7.22 cm 标定会改变固定 `prompt_phase`。因此默认 run name 已改为 `cifar10_4class_moe9x5_top3_globalfc_f722_seed7`；不要把 7.5 cm 几何下训练的 expert/global-FC checkpoint 与新几何混用，应从新 run 重新训练。

## 训练

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/config.yaml
```

默认 `phase_init=zeros`、`weight_decay=0`、phase dropout 关闭、k-space cutoff 关闭。CIFAR 输入从 32×32 缩放到 100×100 时使用带 antialias 的 bicubic 插值，随后 clamp 到 `[0,1]`，再 zero pad 到 120×120。可通过 `dataset.resize_interpolation` 改为 `bilinear` 或 `nearest`。

## Router importance 均衡约束

Router 的 softmax 平均概率记为 `importance`。除了原有 hard top-k load balance，代码还支持：

```text
importance_loss = num_experts * sum(importance ** 2) - 1
```

九个专家均匀时该值为 `0`，所有概率集中到一个专家时为 `8`。它完全基于 softmax 概率，因此未进入 top-k 的专家也能获得梯度。配置项为 `loss.router_importance_weight`；旧配置默认 `0.0`，`configs/config_importance_adamw.yaml` 提供 AdamW、importance 和较强 load balance 的独立实验配置。

训练 CSV 和终端会同时记录 balance loss、importance loss、归一化 router entropy、选择率和平均权重。

## 层间光电转换实验

`configs/config_optoelectronic_interlayers_20cm.yaml` 启用五次层间转换：

```text
expert phase k
-> 20 cm coherent propagation
-> square-law detection |E|^2
-> non-affine LayerNorm over the complete spatial plane
-> ReLU
-> use the nonnegative real result directly as the next optical amplitude
-> expert phase k+1
```

第五个 expert phase 后同样传播20 cm并完成检测、LayerNorm、ReLU和振幅重载，然后作用 global phase；global phase 后传播20 cm，最终平方探测并继续使用原 detector-plane target loss。中间 LayerNorm 没有 affine 参数，因此不会额外引入可训练电子参数。

## SLM BMP 导出

最好 checkpoint 会按光学平面导出 BMP。每个专家平面是一张 450×450 mosaic，不会为 45 个专家层单独输出 45 张 BMP。16 µm 仿真像素用 nearest-neighbor 放大到 8 µm，再居中 zero pad 到 1920×1200。
