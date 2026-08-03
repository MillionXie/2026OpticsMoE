# BDD100K → Bench2Drive Optical MoE16 端到端驾驶

## Bench2Drive Base scratch baseline

`configs/bench2drive_base_scratch.yaml` 是当前不使用 BDD100K 的对照配置。它使用
Bench2Drive Base（名义上 1,000 clips），把所有 expert/global `raw_phase` 初始化为
零，并从行为克隆第一阶段开始联合训练 optical core。原生 Qwen patch/position
embedding 仍作为冻结的图像 stem；该模式没有 BDD PCA target，也不加载预训练光学
checkpoint。

BC 两阶段都训练 Actor、CCD `LayerNorm+Linear`、input adapter、router、expert/global
phase 和 OEO 参数。Stage 1 使用较大的 scratch optical LR；Stage 2 加载完整的
Stage-1 policy，再用较小 LR 联合细化。两阶段都显式保持原生 Qwen 参数冻结。

为控制磁盘占用，`prepare_bench2drive_base.py` 每次只下载一个官方归档，只解出
`camera/rgb_front` 和 `anno`，记录可恢复进度后删除已完成归档。离线行为克隆不需要
CARLA；只有后续官方闭环评测/SAC 才需要 CARLA 0.9.15 及兼容的 Python 3.7/3.8
环境。VISTA 是另一套模拟器，切换后不能视为等价 Bench2Drive 结果。

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

- Stage 1：是否冻结 Optical Backbone 由 `train_optical_from_stage1` 控制；当前
  `bench2drive.yaml` 和 scratch 正式配置都设为 `true`，因此从第一阶段就联合训练。
- Stage 2：加载 Stage 1 best，使用更小学习率联合微调 CCD Linear、Optical core 和 Actor。

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

## BC 稳定性与轮转采样

正式配置默认 `batch_size=8`，并按六种导航命令各取最多 2,000 帧组成一个
epoch。每种命令内部使用固定种子的循环窗口，不会永远重复固定子集；启动日志会打印
真实命令分布、每轮样本数和覆盖完整训练集需要的 epoch 数。

BC 在 backward 后检查 loss 和每个梯度是否有限，再以 `max_norm=1.0` 裁剪；NaN
出现时会列出样本 ID 和首批异常参数，且不会执行该次 optimizer step。每 250 batch
覆盖保存 `bc_stage{stage}_step_last.pt`，重新执行相同 phase 会从该 epoch/batch 继续。
训练 history 还会记录 raw phase RMS 和相对零初始化的物理相位 RMS 变化，便于判断
mask 是否真正更新。

解析后的数万条 `anno/*.json.gz` 会缓存在：

```text
cache/qwen3_vl_embedding_2b_bench2drive_base_scratch_moe16/
  manifests/bench2drive_records_v1.json.gz
```

修改或替换源 Bench2Drive 数据后应删除此索引，让程序重新构建。
