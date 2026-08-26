# Caltech101 四层 10 cm 鲁棒光电检索

这个工程是 `qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval` 的独立正式版本。它面向新的双 SLM 光路，从头联合训练紧凑电子网络、MoE4 router、四个光学阶段、CCD readout、四个融合门和最终 64 维检索头，不加载旧纯电子或旧光学 checkpoint。

当前采样语义对应checkpoint architecture `..._v2`：输入场、phase map和CCD ROI在518面阵上独立错位。修复前的本工程`..._v1` checkpoint没有这一物理语义，禁止作为正式初始化或与v2结果混用；正式实验仍从零训练。

原始 Qwen3-VL-Embedding-2B 主体始终冻结，DeepStack 关闭，训练目标为 Caltech101 中固定的 10 类图像检索。

## 固定实验合同

| 项目 | 正式设置 |
|---|---|
| 波长 | 532 nm |
| 单次衍射距离 | 10 cm |
| 逻辑光场 | 478×478，17 µm/pixel |
| 数值传播画布 | 518×518 |
| 振幅 SLM | 1024×1024，17 µm/pixel，中心 `(512,512)` |
| 振幅极性 | `255=白色/透光`，`0=黑色/遮光`，禁止反相 |
| 相位 SLM | 1920×1200，8 µm/pixel，默认中心 `(980,590)` |
| 相位方向 | 导出前纵向翻转，不做横向翻转 |
| CCD 传输 | 478×478、8-bit 灰度 PNG |
| MoE | 2×2 共4专家，top-k=2 |
| 光学融合系数 | 初值0.20，硬下限0.10 |
| phase LR | 0.006 |
| phase dropout | 8×8块旁路，概率0.08 |
| 错位扰动 | 每个物理阶段在518面阵上独立采样输入场、phase map、CCD ROI三种±16逻辑像素平移；输入与phase最坏相对错位为±32像素 |
| k-space | 开启，`theta_max=0.65°` |

融合写成：

```text
Y = E + alpha * O
alpha = 0.10 + 0.90 * sigmoid(raw_gate)
```

因此四个门的光学残差系数始终不低于0.10。这个下限是分支系数下限，不等同于“输出能量中严格有10%来自光路”。

## 四个硬件阶段

```text
1. vision_expert
2. vision_global
3. language_expert
4. language_global
```

每个阶段都是“振幅输入 → 相位 SLM → 10 cm传播 → CCD → 电子非线性/readout → 与电子 Mixer 融合”。第一和第三阶段使用 MoE4 expert 相位拼图；第二和第四阶段使用 global 相位。阶段之间由 CCD 和电子网络重新加载，不是一条连续传播四张相位板的纯光级联。

支持两种硬件流程：

- 正式四层流程：严格按四层顺序执行 `export → capture → fine-tune`，下一层输入来自之前各层的实测 CCD。
- 最后一层快速流程：前三层使用训练仿真，只播放 `language_global` 的理论输入和相位，采集一层 CCD 后微调末端电子网络。该结果只能表述为“第四层单层实测”，不能冒充四层全部实测。

## 文档

- [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md)：架构、训练项、鲁棒性措施及限制。
- [DATA_PIPELINE.md](DATA_PIPELINE.md)：服务器和实验室之间的数据合同及目录结构。
- [RUN_COMMANDS.md](RUN_COMMANDS.md)：训练、评估、四层采集和快速验证的完整命令。

正式配置：

```text
configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml
```

训练输出：

```text
runs/caltech101_four_layer_moe4_joint_17um_10cm_robust/
```

先在服务器完成训练和仿真评估，再进入硬件阶段。配置和 checkpoint 必须来自这个新工程，不能用旧工程的 checkpoint 交叉加载。
