# BDD100K → Bench2Drive Optical MoE16 端到端驾驶

这是一个独立实验，不修改现有商品检索、分割或 SPAQ 工程。它分成两个离线阶段和一个闭环阶段：

1. 用 BDD100K 前视 RGB 对 Optical Vision Backbone 做道路场景预训练；
2. 用 Bench2Drive 官方专家数据做两阶段行为克隆；
3. 以行为克隆策略初始化 SAC，在 CARLA/Bench2Drive 闭环环境中微调。

## 光学 Backbone

部署时的数据流固定为：

```text
224×224 RGB
→ frozen Qwen3-VL-Embedding-2B patch / position embedding
→ trainable Linear(1024,224) + LayerNorm + Softplus
→ 4×4 Optical MoE16, electronic Top-4 router
→ one 224×224 phase-only expert plane
→ OEO square detection / per-expert LN / ReLU / route-weight reload
→ one 986×986 global phase
→ 10 cm free-space propagation
→ physical CCD square-law intensity over the 986×986 active ROI
→ adaptive pooling to [224,224]
→ LayerNorm(224)
→ Linear(224,224)
→ signed spatial token features
```

物理画布为 1026×1026，16 个 224×224 专家按 4×4 排列，pitch=254，active footprint=986。Expert→OEO/global 和 global→CCD 均沿用 10 cm 配置。最终 CCD 后不再附加 ReLU；`LayerNorm→Linear` 可以恢复正负特征。

在 Qwen Vision hidden=1024 的默认配置下，部署 Backbone 的可训练参数拆分为：

| 部分 | 参数量 |
|---|---:|
| input adapter + norm | 230,048 |
| electronic Top-4 router | 3,152 |
| 16 个 expert phase masks | 802,816 |
| global phase | 972,196 |
| CCD LayerNorm + Linear(224,224) | 50,848 |
| 合计 | 2,059,060 |

BDD 辅助头另有 70,339 个训练期参数，但不会进入部署 checkpoint。BC Actor 有
93,653 个有效确定性参数；SAC 额外训练 3 个全局 `log_std`。

## BDD100K 预训练

冻结 Qwen 教师，hook 最后一个原生 Vision block、merger 之前的 `[T,1024]` 空间 hidden。一次离线 PCA 将教师坐标降到 224 维；PCA 只生成监督目标，不进入学生、不进入部署。

总损失包含：

```text
token-wise normalized cosine + SmoothL1 feature distillation
+ drivable-area BCE/Dice
+ lane-line BCE/Dice
+ road-participant BCE/Dice
+ router balance
```

训练完成后，导出的 checkpoint 只含 Optical core、global phase、CCD LayerNorm 和 `Linear(224,224)`。BDD 辅助头和 PCA 都被移除。

BDD100K 默认目录：

```text
data/bdd100k/
├── images/100k/train/*.jpg
├── images/100k/val/*.jpg
└── labels/
    ├── bdd100k_labels_images_train.json
    └── bdd100k_labels_images_val.json
```

loader 同时支持 official poly2d JSON 和可配置 PNG drivable/lane mask 目录。Mask 始终用 nearest-neighbor resize。

## Bench2Drive 行为克隆

loader 递归读取官方结构：

```text
<clip>/
├── camera/rgb_front/00000.jpg
└── anno/00000.json.gz
```

使用标注中的 `speed / next_command / x_target / y_target / x / y / theta / steer / throttle / brake`。目标点按 ego 朝向旋转为局部坐标；导航命令映射成 6 维 one-hot。划分单位是完整 route/clip，而不是单帧，避免相邻帧泄漏。

配置中的 `train_fraction` 只生成 route-disjoint 的离线 validation，用于选择 BC
checkpoint；它不是 Bench2Drive 官方闭环测试。最终驾驶结论必须在官方 220 条闭环
route 上单独评测。

Actor 输入：

```text
mean-pooled optical token [224]
+ normalized speed [1]
+ navigation one-hot [6]
+ local target point [2]
→ MLP 233→256→128→3
→ steer / throttle / brake
```

- Stage 1：冻结全部 Optical Backbone，仅训练 Actor；
- Stage 2：加载 Stage 1 best，使用小学习率联合微调 CCD Linear、Optical core 和 Actor。

## SAC 闭环微调

仓库实现了 twin-Q SAC、BC Actor 初始化、自动 entropy coefficient、replay buffer、软更新和奖励分解。奖励严格要求环境 `info` 提供：

```text
route_progress, speed, target_speed, lane_offset,
collision, offroad, red_light
```

其中 `route_progress` 必须是本 step 的路线进度增量，而不是整条 route 的累计百分比；
三个违规字段应为当前 step/event 的布尔标志。

默认先全程冻结 Backbone。若设置 `linear_step`，只解冻 CCD 后的
`LayerNorm+Linear`；若设置 `phase_step`，只解冻 16 个 expert phase mask 和
global phase，input adapter、router 与 OEO 仍保持冻结。启用任一解冻阶段时必须同时开启
`replay_store_images=true`，以便从 replay 中重新执行视觉前向并保留梯度。

Bench2Drive 官方仓库是 CARLA route benchmark，不是统一的 Gym 环境。因此配置项 `sac.env_factory` 必须指向用户安装环境的工厂函数：

```text
package.module:function
```

工厂返回 Gymnasium 风格环境；observation 至少含 `rgb_front, speed, command, target_point`。本工程不会假装在没有 CARLA 0.9.15 的机器上完成真实闭环训练。

## 验证范围

`--phase smoke` 不下载 Qwen、不需要 BDD100K/CARLA，实际执行 feature loss、辅助头、BC Actor、Critic 和 reward 的 forward/backward。它用于验证代码路径，不代表正式驾驶结果。
