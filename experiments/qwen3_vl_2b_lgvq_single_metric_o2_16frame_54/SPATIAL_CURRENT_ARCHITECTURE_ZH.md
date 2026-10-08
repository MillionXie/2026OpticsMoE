# LGVQ Spatial 当前合规架构

更新日期：2026-09-10

## 结论

SRCC 0.666503 的版本含冻结预训练 ResNet18，已被否决，只保留为非合规容量
上界。当前正式候选不加载 ResNet、VGG、Attention 或 Transformer block。
在新的微型 RGB 前端完成复测前，最后一个合规结果仍是 s643：SRCC
0.636592、PLCC 0.667525。

## 输入前端

每个视频均匀采样 4 帧，每帧 224x224 RGB。输入同时产生三类信息，但它们
不会形成三个独立评分分支：

1. 冻结 Qwen3-VL 前端只执行 `patch_embed` 和官方位置插值，不执行 Vision
   Transformer、Attention、merger 或语言模型 block。每帧得到 14x14 个
   1024 维 token，四帧张量为 `[B,4,196,1024]`；随后由
   `LayerNorm + Linear(1024,192)` 变为 `[B,4,196,192]`。
2. 固定的 Spatial prompt 经过 Qwen tokenizer 和冻结的 `embed_tokens`，得到
   `[B,38,2048]`，再投影为 `[B,38,192]`。其均值只对视觉 token 施加最多
   约 10% 的 scale/shift；完整的 38 个 token 随后进入语言阶段。
3. 已有的小型质量头从 RGB、亮度、Sobel、Laplacian、局部标准差、饱和度、
   帧差和坐标等 14 个确定性通道提取 `[B,4,196,192]`。它由 5 层普通卷积
   组成，共 339,312 个参数，使用本任务旧 checkpoint 训练后冻结，没有
   ImageNet 预训练，也没有自己的 MOS 输出。

正在复测的 ResNet 替代是一个从 RGB 直接输入的微型深度可分离卷积前端：

```text
RGB 224x224
  -> Conv3x3, 3->16, stride 2
  -> DWConv3x3 + PWConv1x1, 16->24, stride 2
  -> DWConv3x3 + PWConv1x1, 24->32, stride 2
  -> DWConv3x3 + PWConv1x1, 32->48, stride 2
  -> parallel DWConv5x5 / dilated DWConv3x3
  -> zero-start Conv1x1, 96->192
  -> [B,4,196,192]
```

该模块只有 24,264 个随机初始化参数，不加载任何预训练权重，不输出分数，
只作为 E1 内部的有界修正。旧 Conv5 加上该模块共 363,576 个参数；相比被
否决的 2,782,784 参数 ResNet 前端约小 7.65 倍。

## 四个光电阶段

当前推理图只有“一条电子残差支路”和“一条光支路”：

```text
Qwen visual tokens + prompt conditioning
        |
        +-> E1: 视觉二维电子残差
        |       + 冻结小型 Conv5 质量特征
        |       + 可选 24,264 参数 Tiny-RGB 修正
        +-> O1: 光 Router 四专家能量读出 Top-2 + 109x109 相位专家
        -> E1/O1 先做 RMS 同尺度化，再 (1-alpha1)E1 + alpha1 O1
        |
        +-> E2: 第二个视觉二维电子残差
        +-> O2: 全局相位传播
        -> 第二次同尺度融合
        |
        -> 每帧 196 token 做 mean/max 后得到 4 个图像 token
        -> 与 38 个 prompt token 拼成 [B,42,192]
        |
        +-> E3: 一维序列电子残差
        +-> O3: 光 Router 四专家能量读出 Top-2 + 相位专家
        -> 第三次同尺度融合
        |
        +-> E4: 第二个序列电子残差
        +-> O4: 全局相位传播
        -> 第四次同尺度融合
        -> 单一 Spatial MOS 读出头 -> 一个连续分数
```

四个融合系数沿用 s643 的 `[0.45, 0.55, 0.40, 0.775]`。物理设置为
532 nm、17 um、10 cm，视觉专家 109x109，光 Router 固定 Top-2。当前仿真
连续相位、不做 8-bit 直通量化、不做 k 空间滤波，像素/相位/CCD 平移扰动为
0，未调制分量固定为 20%。

## ResNet 原来加载在哪里

被否决的 ResNet 不在 Qwen 内，也不在最终 MOS 头。它在离线阶段读取同样的
4 帧 RGB，运行 `stem + layer1 + layer2 + layer3` 后缓存
`[B,4,196,256]`。推理图中的 173,120 参数适配器将其变成 192 维，并执行：

```text
E1 = ElectronicResidual1(QwenVisual)
E1 = E1 + Conv5Quality
E1 = E1 + ResNetAdapter(ResNetTokens)
```

然后才进入第一次光电融合。它虽然不是直接评分旁路，但 278 万参数的额外
预训练视觉前端仍不符合最终架构，因此整个 0.666503 checkpoint 已降级。

## 当前复测配置

- 仅训练 Tiny-RGB：`configs/release/spatial_tiny_rgb_e1_s710.yaml`
- Tiny-RGB 与唯一 MOS 头联合：`configs/release/spatial_tiny_rgb_joint_s713.yaml`
- Tiny-RGB、原电子路径与唯一 MOS 头联合：
  `configs/release/spatial_tiny_rgb_path_s714.yaml`

所有配置都显式设置 `resnet_feature_cache: null`。训练仍每个 epoch 测试一次，
按 test SRCC 保存 best，并且只保留 best 与 last。
