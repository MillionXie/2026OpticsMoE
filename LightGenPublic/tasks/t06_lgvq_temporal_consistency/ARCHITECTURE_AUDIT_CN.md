# LGVQ Temporal 0.8044 物理与代码架构审计

本文记录 `multivideo16x4_rank_s163` 固定权重的实际计算契约。它的目的
是防止把实验台参数、鲁棒性增强和相机噪声混为一谈，并给后续代码整理提供
不改变 0.8044 数值结果的边界。

## 结论先行

1. **本权重对应 10 cm，不是 15 cm。** 配置、角谱传播实现和 2026-09-14
   硬件记录均使用 `distance_m = 0.10`。若改成 15 cm，传递函数发生变化，必须
   作为新实验重新评测，不能继续引用 0.8044。
2. **17 µm 是仿真逻辑采样间距，8 µm 是两块 SLM 的物理像素间距。** 当前
   映射保持物理孔径尺寸，而不是一一对应逻辑像素。
3. **产生 0.8044 的原配置没有启用相机传感器噪声。** 当前代码已补充训练期可选
   的 CCD 噪声（偏置高斯和 photon shot noise），但正式配置保持
   `enabled: false`，因此固定权重结果仍使用理想强度探测。
4. **调制网格映射已进入训练图。** 17 µm 逻辑场在每次传播前映射到 8 µm
   调制网格，传播后的强度再映射回固定读出网格。
5. **不应原地重构冻结 runtime。** 应保留兼容层复现 0.8044，并在旁边建立
   清晰的新实现，以逐模块/逐样本等价测试迁移。

## 固定物理契约

| 参数 | 固定值 | 含义 |
|---|---:|---|
| 波长 | 532 nm | 单色光角谱传播 |
| 传播距离 | 0.10 m | 每次光学传播的距离 |
| 仿真采样间距 | 17 µm/pixel | 518×518 角谱网格的逻辑间距 |
| 有效平面 | 478×478 pixels | 16 个视频共同占用的有效区域 |
| 仿真画布 | 518×518 pixels | 有效区四周各有 20 个逻辑像素 guard |
| 空间频率截断 | 0.5° | 额外的圆形 k-space 低通 |
| 相位量化 | 关闭（0 levels） | 0.8044 仿真使用连续相位 |
| 评测零级光功率比例 | 0.20 | 以相干场形式与调制场相加 |

角谱传播使用

```text
H(fx, fy) = exp(i z sqrt(k² - (2πfx)² - (2πfy)²))
Uout = IFFT2(FFT2(Uin) · H)
```

并将倏逝分量及超过 0.5° 截止角的频率置零。FFT 没有额外零填充或吸收边界，
所以 518 画布仍具有离散 FFT 的周期边界假设；训练中的 guard-energy loss 只能
抑制能量跑到视频支持区之外，不能替代更大的传播画布。

17 µm 采样的 Nyquist 角约为 0.896°，因此 0.5° 截止是主动增加的低通约束，
不是由采样极限自动产生的。

## 17 µm 与 8 µm 如何匹配

模型有效孔径的物理宽度为

```text
478 × 17 µm = 8126 µm = 8.126 mm
```

实验台导出代码按下面的规则计算 SLM 边长：

```text
Ndevice = round(478 × 17 / 8) = 1016 pixels
Wdevice = 1016 × 8 µm = 8128 µm = 8.128 mm
```

因此总宽度只多 2 µm，相对误差约为 0.0246%。这个映射保持的是**物理孔径**。
缩放倍率 `17 / 8 = 2.125` 不是整数，所以一个 17 µm 逻辑像素会覆盖 2 或 3 个
8 µm 设备像素；不能把两套网格描述为逐像素完全等价。

当前设备导出还包含以下处理：

- 相位：连续相位先映射到 0--255，再用 nearest-neighbor 放大；相位 SLM 额外
  做水平、垂直翻转和 `255 - gray` 编码。
- 振幅：每个场单独以正振幅 p99.5 为尺度，截断到 `[0, 1]`，转 8 bit，再用
  bilinear 放大。
- 1016×1016 有效图居中嵌入 1920×1080 振幅面板和 1920×1200 相位面板。
- 仿真的 20-pixel guard 没有作为一张 518×518 图整体导出；设备有效图之外用
  零振幅/零相位背景填充，物理面板本身提供了更大的外围区域。

这里有两个需要优先修正的 sim-to-real 风险：

1. 连续相位训练与 8 bit 相位播放不一致，而且 17→8 µm 的 nearest-neighbor
   缩放会形成不等面积的逻辑单元。
2. 振幅的逐样本 p99.5 归一化是一个样本相关非线性，它没有出现在模型前向中。
   正式系统更适合使用通过标定得到的固定映射，并记录实际光功率和饱和状态。

## 实际输入与特殊预处理

每个物理场装入 16 个互不相关的视频，每个视频使用 4 帧。视频帧不是从首尾
取样，而是在总时长的 10%、36.67%、63.33%、90% 附近取固定帧；这样会跳过
部分不稳定的首尾解码帧。每帧执行以下操作：

1. 取短边的中心 65% 正方形区域；
2. 使用 area interpolation 缩放到 448×448；
3. 一路进入冻结的 Qwen3-VL patch embedding + position embedding，最后池化为
   7×7×1024；不运行 Qwen Vision block、merger 或 attention；
4. 另一路构造 7×7×14 的固定质量特征。

14 个质量通道依次是：RGB 三通道、亮度、Sobel-x、Sobel-y、梯度幅值、绝对
Laplacian、5×5 局部标准差、RGB 最大最小差、相邻采样帧亮度绝对差、x 坐标、
y 坐标和时间坐标。第一帧的相邻帧差固定为零。全部通道做 7×7 自适应平均池化
后以 float16 缓存。

模型输入还含固定 Temporal prompt 的 Qwen embedding。学生网络自身没有
Transformer/attention，但它依赖冻结 Qwen-front 特征和显式质量特征，不能描述
成纯光学或纯原始像素模型。

## 六次光学传播与电子支路

```text
Qwen-front + quality14 + prompt conditioning
                  │
                  ├─ frame optical router ─ frame Top-2 expert ─ frame global
                  │       ╲ electronic route + RMS convex fusion ╱
                  │
                  └─ 4 frame summaries + prompt tokens
                          │
                          └─ video optical router ─ video Top-2 expert ─ video global
                                  ╲ electronic route + RMS convex fusion ╱
                                              │
                                       temporal readout → MOS
```

共有 frame router、frame expert、frame global、video router、video expert、
video global 六次传播。光学和电子分支在四个位置进行 RMS 归一化后的凸融合。
融合系数约为 0.56，但这不是“56% 性能来自光学”的归因比例。

时序读出对每帧 token 做 mean/max 汇聚，再使用核 3、5 的 depthwise temporal
convolution；最终汇总序列 mean/std/max、一阶差分和二阶差分统计后回归 MOS。

## 现有鲁棒性项与真正相机噪声的区别

| 项目 | 训练 | 固定权重评测 | 是否为相机噪声 |
|---|---:|---:|---|
| 输入平移 | 每轴整数均匀采样 `[-4, 4]` | 关闭 | 否，几何增强 |
| 相位平移 | 每轴整数均匀采样 `[-4, 4]` | 关闭 | 否，装调增强 |
| CCD 平移 | 每轴整数均匀采样 `[-4, 4]` | 关闭 | 否，ROI/装调增强 |
| 相位 dropout | 4×4 cell，概率 0.05 | 关闭 | 否，SLM 失效增强 |
| 零级光功率比例 | Uniform[0.20, 0.35] | 固定 0.20 | 否，相干漏光 |
| router 高斯噪声 | logits 标准差 0.06 | 关闭 | 否，路由正则 |
| 普通 dropout | 0.10 | 关闭 | 否，网络正则 |
| CCD 噪声 | 可选，正式配置关闭 | 关闭 | 偏置高斯和 photon shot noise |
| CCD 探测 | 理想 `abs(field)^2` 后接可选扰动 | 理想 `abs(field)^2` | 不含完整传感器模型 |

相位 dropout 后的单元变成零相位透射（复振幅 1），不是振幅遮挡。零级光使用
`sqrt(1-f) exp(iφ) + sqrt(f)` 的相干叠加，也不是加到图像上的高斯噪声。

正式 0.8044 profile **没有启用**上述简化噪声。可选分支仅包含偏置高斯和
photon shot noise，没有建模以下相机/器件因素：

- read noise、dark current、black level；
- PRNU、DSNU、坏点和固定图样噪声；
- ADC 量化、饱和曲线、gamma；
- 曝光/增益漂移、帧间抖动、滚动快门；
- 相机 MTF/PSF、散斑漂移、杂散光背景；
- 振幅 SLM 的 8 bit 量化与非线性 LUT；
- 相位 SLM 的 8 bit 量化、空间串扰和实测 LUT。

因此不能把 0.8044 原训练称为“包含完整相机噪声建模”。当前新增项是相对每个
干净 CCD 场均值的可选简化扰动，不等同于标定后的 sensor model。

## CCD 读出与实机特殊处理

仿真 CCD 强度裁出每个 lane/video patch 后执行：

```text
I = clamp(I, min=0)
Irel = clamp(I / mean(I), max=8)
feature = log1p(Irel)
```

随后做自适应平均池化、LayerNorm 和线性投影。逐 patch 除均值会移除绝对亮度
与统一增益，所以模型对曝光整体缩放较不敏感；但暗场、饱和、背景偏置和空间
非均匀性仍不会被正确模拟。

2026-09-14 的 SHS-202-M 实验使用 Mono8、400 µs、Gain_X4、100 fps，等待
240 ms 后取单帧。整幅相机图通过四点透视变换和已记录的水平镜像映射到
478×478，再 clip/round 为 uint8 PNG。六层统一乘 `1/255` 注入网络。

采集代码只拒绝明显异常帧：`p99 <= 8 且 std < 1.5` 的近暗帧，或 255 饱和
像素比例超过 1% 的帧。没有暗场扣除、平场校正、坏点修复或重复帧平均。正式
运行也没有保存 full-sensor raw frame，只保存透视校正后的 PNG，这限制了事后
重新标定和噪声统计。

实机六层结果为 SRCC 0.7977，不是 0.8044。第一层前三幅实测与仿真强度 PCC
约 0.30；第五层信号较弱，重复采集与原图 PCC 为 0.963/0.966。这说明最终
SRCC 接近不代表逐层光场已经匹配。

## 当前代码组织的问题

- 独立包虽然没有导入外部仓库，但为了 checkpoint 兼容仍保留了
  `runtime/LightGenV2/...` 和 `runtime/experiments/...` 两套历史命名空间；审阅者
  很容易选错 settings/modeling 入口。
- `configs/temporal_16x4_s163.yaml` 必须由
  `LightGenV2.tasks.t06_video_quality_assessment.multivideo_settings.load_settings`
  读取，而不是同包内 experiment 的通用 `settings.load_settings`。入口职责没有从
  目录名直观体现。
- 通用 `modeling.py` 和 `settings.py` 混有多代、多个未启用架构，文件过大；
  0.8044 实际依赖的路径没有形成最小模块边界。
- 传播、相位调制、鲁棒性增强、CCD 归一化和多视频布局耦合在模型实现中，难以
  单独验证单位、设备映射及传感器模型。
- 硬件代码另处维护，17→8 µm resize、相位编码、振幅 p99.5 归一化和相机 warp
  不在同一个可校验 contract 下，容易产生配置漂移。
- `MultiVideo9x4OpticalVQA` 等历史类名仍用于 16×4 模型，语义已经失真。
- “参考仿真”“训练鲁棒仿真”“设备真实仿真”没有成为显式 profile，导致增强项
  容易被误称为相机噪声。

## 建议的代码架构

冻结实现继续作为 `legacy_runtime/` 保留，仅负责复现 checkpoint；新代码使用
正常包名，不再把历史 `experiments/...` 和 `LightGenV2/...` 命名空间复制到发布
包。建议目标结构如下：

第一阶段已经新增 `runtime/lgvq_temporal/`：公开入口现在只依赖干净 facade，
facade 在载入 checkpoint 后强制核对 532 nm / 17 µm / 10 cm / 478 / 518 合同；
历史模块暂时仍作为 facade 下方的只读兼容后端。

```text
lgvq_temporal/
  contracts.py              # tensor shape、单位和物理不变量 dataclass
  features/
    video_sampling.py       # 10%--90%、65% center crop
    qwen_front.py           # 冻结前端边界
    quality14.py            # 14 通道定义
  model/
    network.py              # 顶层六 pass 编排
    electronic.py           # 四条电子支路
    routing.py              # router 与 Top-2
    fusion.py               # RMS convex fusion
    readout.py              # temporal readout
  optics/
    propagation.py          # AngularSpectrum，只接收 SI 单位 contract
    modulation.py           # phase、leakage、quantization
    geometry.py             # 16×4 布局及 support mask
  robustness/
    alignment.py            # train-only shifts/dropout
    sensor.py               # 可标定 sensor forward model
  hardware/
    coordinates.py          # 17 µm ↔ 8 µm 物理坐标映射
    rasterize.py            # 固定 LUT/量化/方向，不做隐式样本归一化
    camera.py               # raw capture 与元数据
    calibration.py          # dark/flat/MTF/响应曲线
  train/
    losses.py
    loop.py
  eval/
    metrics.py
    reproduce.py
legacy_runtime/             # 当前冻结 checkpoint 兼容实现，只读
configs/
  model_08044.yaml
  simulation_reference.yaml
  simulation_robust.yaml
  simulation_device_realistic.yaml
  hardware_shs202m_8um.yaml
tests/
  equivalence/
  physics/
  hardware/
```

三个仿真 profile 必须明确分开：

- `reference`：严格复现 0.8044；连续相位、理想 CCD、eval leakage 0.20。
- `robust`：当前训练增强；位移、phase dropout、leakage 区间和 router noise。
- `device_realistic`：使用 8 µm 物理栅格、SLM LUT/量化和经过标定的 sensor
  model。它是新实验，不自动继承 0.8044 指标。

## 推荐迁移顺序

1. 冻结当前 runtime、checkpoint、35 个输入场和逐样本预测作为 golden reference。
2. 先抽出只读 `PhysicalContract`、`TensorContract` 和单位检查，不改变前向。
3. 逐个迁移 propagation、modulation、geometry，并对每个 tap 做误差比较。
4. 再迁移 router、fusion、readout，要求 558 条预测和指标在既定容差内一致。
5. 独立加入设备坐标映射，采用物理坐标采样而不是隐含的图片 resize。
6. 采集 dark/flat/repeated frames 后拟合 sensor 参数；没有标定数据时不填写任意
   “经验噪声值”。
7. device-realistic profile 重新训练/评测，结果与 0.8044 分开报告。

迁移过程中任何改变传播距离、采样间距、相位量化、振幅归一化、帧采样或 CCD
归一化的修改都属于新实验，而不是代码整理。
