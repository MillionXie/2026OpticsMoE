# 四专家光学 Prompt：当前架构、推荐参数与后续规划

本文总结当前四专家光学 Prompt 几何验证的实现、推荐参数、已验证结果、已知限制和后续优化方向。它可以直接作为后续方案讨论或交给 GPT 继续规划时的背景文档。

## 1. 当前目标

当前阶段只验证光学几何，不进行分类训练。

需要确认：

1. 中心输入光场能否被复制到四个专家入口。
2. 微透镜是否负责成像，局部光栅是否能修正像的位置和主光线角度。
3. 四个专家的幅度权重是否能独立控制。
4. 光场经过五层恒等专家和一层恒等全局 FC 后，是否仍能保持四路布局。
5. 是否存在明显的方向错误、镜像、转置、持续漂移、边缘回卷或串扰。

当前脚本：

```text
scripts/test_four_expert_prompt_geometry.py
```

该脚本不训练参数，也不会修改现有左右双专家 OpticalMoE。

## 2. 当前光路架构

完整流程如下：

```text
中心输入光场
    ↓ 自由空间传播 0.20 m
四分区微透镜阵列 Prompt
    ├─ 局部薄透镜相位：完成成像
    ├─ 局部光栅相位：修正像的位置和传播角度
    └─ 标量幅度：控制每个专家的光能权重
    ↓ 自由空间传播 0.20 m
四个专家入口平面
    ↓
5 层恒等专家层，每层间传播 0.05 m
    ↓
恒等全局 FC，相距 0.05 m
    ↓
最终探测器平面，相距 0.05 m
```

本测试中的专家相位全部为 0。它验证的是 Prompt 和空间布局，不代表已经完成四专家分类网络。

## 3. 推荐基准参数

| 参数 | 推荐值 | 说明 |
|---|---:|---|
| 波长 | 532 nm | 当前衍射传播和相位计算基准 |
| 像素尺寸 | 8 um | 与已有 OpticalMoE 仿真一致 |
| 画布 | 700 × 700 | 完整传播网格 |
| 输入尺寸 | 200 × 200 | 放在画布中心 |
| 专家尺寸 | 200 × 200 | 每个专家的有效入口区域 |
| 专家间距 | 100 px | 当前专家 aperture 之间的间隔 |
| 输入到 Prompt | 0.20 m | 物距 `s` |
| Prompt 到专家 1 | 0.20 m | 像距 `s'` |
| 微透镜焦距 | 0.10 m | 满足 `1/f = 1/s + 1/s'` |
| 放大率 | 1 | `M = s'/s` |
| 层间传播 | 0.05 m | 五层恒等专家之间 |
| Layer 5 到 FC | 0.05 m | 当前几何基准 |
| FC 到探测器 | 0.05 m | 当前几何基准 |
| aperture 模式 | hard | 几何验证时阻止杂散光绕过专家 |
| 设备 | CUDA | CPU 也可运行，但 CUDA 更适合后续扫描 |

四个专家中心：

| 专家 | y 范围 | x 范围 | 中心 |
|---|---|---|---|
| E0 | 100:300 | 100:300 | (200, 200) |
| E1 | 100:300 | 400:600 | (200, 500) |
| E2 | 400:600 | 100:300 | (500, 200) |
| E3 | 400:600 | 400:600 | (500, 500) |

## 4. Prompt 相位结构

每个分区的透射函数为：

```text
T_k = mask_k × a_k × exp(i × (phi_lens + phi_grating + b_k))
```

其中：

- `mask_k`：第 k 个 Prompt cell 的空间掩膜。
- `a_k`：控制专家能量的标量幅度。
- `phi_lens`：局部薄透镜相位，负责形成图像副本。
- `phi_grating`：局部线性相位，负责修正像的位置和出射角。
- `b_k`：可选的整体相位偏置。

默认距离下，各 cell 相对光轴偏移为 `±150 px = ±1.2 mm`。

计算得到：

- x/y 偏转角绝对值约 `0.3438°`。
- x/y 光栅周期约 `11.08 px`。
- 光栅周期高于当前设置的 `8 px` 采样警戒线。

微透镜和光栅缺一不可：

- `lens_only`：能够成像，但离轴图像会越过专家中心。
- `grating_only`：能改变传播方向，但不能正确形成图像副本。
- `lens_plus_grating`：同时完成成像、位置修正和主光线角度修正。

## 5. 当前验证结果

完整结果目录：

```text
runs/four_expert_prompt_geometry_verified
```

关键结果：

| 指标 | 结果 |
|---|---:|
| 点源 lens + grating 平均对准误差 | 1.30 px |
| 五层恒等专家平均逐层质心漂移 | 8.45 px |
| 四专家总能量比例，方形输入专家 1 平面 | 74.14% |
| 方形输入专家区域外能量比例 | 25.86% |
| all-on 四专家是否都有能量 | 是 |
| 四个 one-hot 路由是否都正确 | 是 |
| 总体几何状态 | PASS |

one-hot 幅度路由结果：

| 路由模式 | 目标专家能量比例 | 最大非目标专家能量比例 |
|---|---:|---:|
| onehot E0 | 71.25% | 1.22% |
| onehot E1 | 71.43% | 1.24% |
| onehot E2 | 71.43% | 1.24% |
| onehot E3 | 71.61% | 1.21% |

点源位置误差是严格的几何判据。方形或 MNIST 属于扩展相干目标，背景衍射会把普通能量质心拉向画布中心，因此不能只用全区域质心判断图像是否对准，还需要结合目标区域能量和输出图像观察。

## 6. 关于 Prompt cell 之间的 gap

### 6.1 当前代码的真实行为

当前 Prompt cell 与专家 aperture 使用同样的 `200 × 200` 范围，cell 之间有 `100 px` 的中央十字间隔，画布外围还有 `100 px` margin。

当前复透射率先初始化为 0，随后只在四个 cell 内写入微透镜和光栅：

```text
cell 内：a_k × exp(i × phase_k)
cell 外：0
```

因此 gap 中不是“相位为 0 并透明通过”，而是：

```text
复振幅透射率 = 0
```

也就是完全遮光。

四个 cell 总面积为：

```text
4 × 200 × 200 = 160000 pixels
```

整个画布面积为：

```text
700 × 700 = 490000 pixels
```

当前有效 Prompt 填充率只有约：

```text
160000 / 490000 = 32.65%
```

其余约 67.35% 的 Prompt 平面被阻挡。

### 6.2 这种设置是否正确

对于当前“验证四个互不重叠分区能否独立成像和路由”的实验，它是正确且保守的：

- 能保证每个像素只属于一个 Prompt cell。
- 不会发生多个局部相位函数在同一像素直接叠加。
- 能清楚验证 one-hot 幅度路由。
- 能阻止未调制光直接穿过 Prompt 并形成中心背景。

但它不适合直接作为最终高效率 OpticalMoE Prompt：

- 100 px 的硬 gap 会损失大量光能。
- 二值硬边缘会产生明显衍射条纹。
- 每个局部波前被突然截断，输出图像会有明显的裁切感。
- 光场在 gap 附近的能量不会被任何专家利用。
- 后续层可能出现更高的旁瓣、串扰和边缘敏感性。

因此你观察到的明显截断不是显示问题，而是当前硬掩膜结构的真实结果。

### 6.3 不建议直接把 gap 改成透明

最简单的修改是让 gap 的透射率从 0 变成 1，但这通常不是正确方案。

透明 gap 会让大量没有经过微透镜和角度修正的光直接传播。这部分光可能：

- 留在画布中心；
- 落到多个专家之间；
- 在后续传播中进入错误专家；
- 提高背景能量和串扰；
- 让 amplitude routing 失去明确的物理意义。

因此应该提高有效填充率，而不是简单让 gap 透明。

## 7. 推荐的 Prompt cell 优化方案

### 方案 A：增大 cell，专家 aperture 保持不变

这是最推荐的下一步。

将 Prompt cell 尺寸与专家尺寸解耦：

```text
expert_size = 200
prompt_cell_size = 280 或 300
```

Prompt cell 仍以四个专家中心为相位中心，但覆盖更大的 Prompt 平面。

推荐扫描：

```text
prompt_cell_size = 200, 240, 280, 300
```

当 cell 为 `300 × 300` 时，四个 cell 可以在中心边界处刚好相接：

```text
C0: y=50:350,   x=50:350
C1: y=50:350,   x=350:650
C2: y=350:650,  x=50:350
C3: y=350:650,  x=350:650
```

此时：

- 中央 gap 从 100 px 降为 0。
- 外圈只保留 50 px 安全边界。
- Prompt 填充率提高到约 73.47%。
- 专家 aperture 仍保持原来的 `200 × 200`，不会改变后续专家网络几何。

需要重新检查：

- 点源对准误差；
- 四专家能量均匀性；
- outside energy；
- one-hot 串扰；
- 五层传播漂移；
- 相位边界产生的衍射。

### 方案 B：对 cell 边缘进行平滑加窗

在 cell 边缘加入 cosine/Tukey 过渡，避免从 1 突然跳到 0。

推荐扫描：

```text
edge_taper_pixels = 0, 5, 10, 20
```

优点：

- 减少硬边缘高频分量；
- 降低明显条纹和振铃；
- 输出图像通常更平滑。

代价：

- 会进一步损失一部分能量；
- taper 过宽会降低有效孔径和分辨率。

因此建议先增大 cell，再使用 5 到 10 px 的轻度 taper。

### 方案 C：高填充率 Voronoi/象限分区

将 Prompt 平面的每个像素分配给最近的专家中心，使内部没有零 gap。

优点：

- Prompt 平面利用率高；
- 没有中央遮光十字。

问题：

- cell 形状和局部透镜有效孔径不再完全对称；
- 相位边界仍然不连续；
- 外圈像素距离相位中心较远，可能增加采样压力；
- 需要更严格的波前和串扰验证。

该方案适合作为后续对照，不建议直接替换当前基准。

### 方案 D：共享相位面上的连续优化

最终可以不使用严格的二值 cell，而是在整个 Prompt 平面优化连续复透射函数：

```text
T(x,y) = A(x,y) × exp(i × phi(x,y))
```

用损失函数同时约束：

- 四专家目标能量；
- 非目标专家串扰；
- 输出图像保真度；
- 相位平滑性；
- 最小可制造光栅周期；
- gap/边界能量；
- 不同任务的路由权重。

这属于后续可训练 Prompt 阶段，不应在当前几何基准尚未稳定前直接开展。

## 8. 推荐的下一轮实验

建议保持专家 aperture 和传播距离不变，只扫描 Prompt 有效孔径：

| 实验变量 | 建议取值 |
|---|---|
| prompt cell size | 200、240、280、300 |
| edge taper | 0、5、10 px |
| 输入 | delta、方形、MNIST 5 |
| 幅度 | all-on、四个 one-hot |
| aperture mode | 专家层继续 hard |

每组至少记录：

```text
point_source_centroid_error
expert_energy_ratio
outside_energy_ratio
onehot_target_to_wrong_ratio
identity_stack_drift
edge_energy_ratio
image_similarity
```

推荐选择标准：

1. 点源平均对准误差小于 5 px。
2. all-on 时四专家能量尽量接近。
3. one-hot 目标专家能量至少为最大错误专家的 10 倍。
4. 专家平面 outside energy 尽量低于 20%。
5. 五层平均逐层漂移小于 10 px。
6. 方形和 MNIST 输出不出现严重裁切、镜像或转置。
7. 最小局部相位周期不低于 8 px。

基于当前结果，建议优先测试：

```text
prompt_cell_size = 300
edge_taper_pixels = 5
s = s' = 0.20 m
f = 0.10 m
```

## 9. 距离扫描结论

当前距离 sweep 使用点源进行严格位置测试：

| s=s' | f | 光栅周期 | 平均误差 | 结论 |
|---:|---:|---:|---:|---|
| 0.10 m | 0.05 m | 5.54 px | 0.66 px | 几何正确，但光栅采样过密 |
| 0.15 m | 0.075 m | 8.31 px | 91.32 px | 当前离散 ASM 下未对准 |
| 0.20 m | 0.10 m | 11.08 px | 1.30 px | 推荐基准 |

因此当前不要仅依据薄透镜公式任意改变距离。传播距离、画布、像素尺寸、光栅周期和离散 ASM 采样相互约束，改变其中一个参数后必须重新执行完整几何验证。

## 10. 运行命令

默认 CUDA 验证：

```powershell
python scripts/test_four_expert_prompt_geometry.py --device cuda --out-dir runs/four_expert_prompt_geometry_test
```

包含距离扫描：

```powershell
python scripts/test_four_expert_prompt_geometry.py --device cuda --sweep-distances --out-dir runs/four_expert_prompt_geometry_sweep
```

one-hot E0：

```powershell
python scripts/test_four_expert_prompt_geometry.py --device cuda --amplitudes 1 0 0 0 --out-dir runs/four_expert_prompt_onehot_e0
```

MNIST 样本：

```powershell
python scripts/test_four_expert_prompt_geometry.py --device cuda --input-type mnist_sample --mnist-index 5 --data-root ./data --out-dir runs/four_expert_prompt_mnist
```

## 11. 后续开发边界

下一阶段推荐按以下顺序推进：

1. 将 `prompt_cell_size` 与 `expert_size` 解耦。
2. 增加 cell size 和 edge taper sweep。
3. 加入图像保真度指标，而不只测能量和质心。
4. 确定高填充率 Prompt 几何。
5. 用真实恒等以外的四专家相位替换当前 identity expert。
6. 验证单专家 checkpoint 在四专家画布上的迁移。
7. 最后再训练幅度权重、残差相位或输入相关 router。

在高填充率几何通过之前，不建议直接训练四专家 Prompt。否则优化器可能主要在补偿硬截断和错误采样，而不是学习真正的任务路由。
