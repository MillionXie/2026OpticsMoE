# T22 时间一致性参考实现

当前唯一模型由 `configs/selected.json` 指定，全部结果与图位于 `runs/simulation/current/`。

## 当前版本

本轮先优化视觉 router 与四个线性专家，使 Top2 选择份额接近均衡，再固定这部分参数，继续训练两级 global 相位。原始初始化沿用 `raw=0 → 2π·sigmoid(raw)=π`，本轮相位阶段从上一版已学习的 mask 继续优化。

相位阶段完成 260 轮，选中第 86 轮。视觉 router 训练使用真实 Top2 选择次数反馈调整现有输出偏置，并用 TRAIN 上的专家混合输出和视频评分进行蒸馏。模型权重、初始化和逐张量核验保存在 `selection.json`，前序训练记录在 `training_stages/`。

| 558 条原 TEST 开发评价条件 | SRCC |
|---|---|
| 当前学习到的两级相位 | 0.7991007779 |
| 视觉相位替换为常相位 π | 0.7940065649 |
| 图文融合相位替换为常相位 π | 0.7908598884 |
| 两级相位同时替换为 π | 0.7891170378 |
| 两级相位同时替换为各 tile 的照明区域圆均值 | 0.7889792107 |

原仿真参考 SRCC 为 0.8043868643，本轮绝对差为 0.0052860864，达到本轮 SRCC ≥0.79 的要求。

## 先看这些文件

| 文件 | 用途 |
|---|---|
| `configs/selected.json` | 唯一模型入口、SHA256、选中轮次和评价条件 |
| `configs/train.json` | 当前相位训练的实际配置 |
| `runs/simulation/current/LightGen.pt` | 独立推理权重 |
| `runs/simulation/current/figures/` | 当前模型的全部图 |
| `runs/simulation/current/test_predictions.csv` | 558 条逐视频目标与预测 |
| `runs/simulation/current/phase_effect/` | 常相位消融、逐视频结果与相位原始数组 |
| `runs/simulation/current/robustness/` | 26 项数值扰动测试 |
| `runs/simulation/current/routing/` | 专家选择次数及逐槽位统计 |
| `runs/simulation/current/metadata.json` | 实际配置、训练源码、平台及数据身份 |
| `runs/simulation/current/history.json`、`logs/` | 训练曲线与原始日志 |
| `runs/simulation/current/training_stages/` | 当前模型前序 router／专家训练的配置、指标和日志 |
| `runs/simulation/current/validation/` | 复算、视频入口与初始化核验 |
| `assets/initialization/initialization.pt` | 本轮相位阶段的实际初始化，包含已均衡训练的 router 和专家 |

```text
configs/
  selected.json
  train.json
assets/
  front.pt
  vision.pt
  language.pt
  manifest.csv
  training_only_teacher_predictions.pt
  initialization/
  example/
runs/simulation/current/
  LightGen.pt
  best_checkpoint.pt
  last_checkpoint.pt
  metadata.json
  history.json
  metrics.json
  test_predictions.csv
  selection.json
  logs/
  training_stages/
  fields/
  routing/
  routing_example/
  phase_effect/
  robustness/
  validation/
  figures/
```

## 输入到输出

每个光场组织 16 个视频，每视频取 4 帧。输入顺序为 10%、36.67%、63.33%、90% 时间位置的四帧，取短边 65% 的中心正方形区域，转 RGB 并缩放至 448×448。

| 顺序 | 操作 | 每视频尺寸 |
|---|---|---|
| 1 | 冻结 Qwen patch 投影、位置特征及原空间均值汇聚 | `4 × 49 × 1024` |
| 2 | 视觉 LN1024、Linear1024→192 | `4 × 49 × 192` |
| 3 | 每帧 router，LN192、Flatten9408、Linear9408→128、ReLU、Linear128→4 | `4 × 4` 个 logits |
| 4 | 四专家 softmax、Top2、所选两项按和归一化 | 每帧两项非零 |
| 5 | 四个无偏置 Linear192→192 专家，按该帧权重汇合 | `4 × 49 × 192` |
| 6 | Linear192→27/ReLU，token 轴 Linear49→27/ReLU，RMS 规范化 | `4 × 27 × 27` 幅度场 |
| 7 | Vision global，相位调制、整场传播、强度探测、CCD 读取 | `4 × 49 × 192` |
| 8 | 原 Frame Merger，空间 mean/max、LN384、Linear384→192、ReLU，加帧位置参数 | `4 × 192` |
| 9 | 固定 prompt，LN2048、Linear2048→192，与视觉 token 拼接并加序列位置参数 | `42 × 192` |
| 10 | 图文 router，LN192、Flatten8064、Linear8064→128、ReLU、Linear128→4，softmax 与 Top2 | 4 个概率、2 个激活专家 |
| 11 | 独立的一组四个无偏置 Linear192→192 专家，按每视频权重汇合 | `42 × 192` |
| 12 | Linear192→56/ReLU，补零到 56 行，RMS 规范化 | `56 × 56` 幅度场 |
| 13 | Vision-language global，相位调制、整场传播、强度探测、CCD 读取 | `42 × 192` |
| 14 | Flatten8064、Linear8064→256、ReLU、Linear256→5，softmax 与五等级分数加权 | 1 个 MOS |

视觉 router 与四个视觉专家在所有帧和视频槽位间共享参数。每帧独立选择专家，该帧 49 个 token 使用同一组权重。图文 router 读取 4 个视觉 token 与 38 个文本 token，共 42 个 token，为每视频产生一组权重。两个阶段各有独立的 router 和专家。

两个 global 均使用 518×518 传播画布、中央 478×478 有效区，以及 16 块 111×111 相位。Vision global 中每视频四帧按 2×2 排列，整场形成 8×8 帧输入区。Vision-language global 中每视频对应一个输入区，形成 4×4 排列。传播参数为 532 nm 波长、17 μm 采样、10 cm 距离和 0.5° 截止。

## 相位训练与实际改变量

最后阶段优化两级相位，共 394,272 个原始相位参数。模型总参数为 5,732,678。此前视觉 router 与线性专家的优化记录在 `training_stages/`。`configs/train.json` 保存最后相位阶段的实际配置。

上一版采用 σ=6 像素的梯度平滑、σ=0.35 的参数平滑，以及 75% 的跨槽位梯度平均。这些设置偏向大块、相似的相位结构。本轮分别减弱至 2.5 像素、0.15 和 50%，同时保留相邻相位及曲率约束。视觉照明区由四个 27×27 帧区域构成，图文融合照明区为中央 42×56，局部训练纹理主要出现在这些区域。

| 统计，覆盖全部 16 个相位 tile | 视觉 | 图文融合 |
|---|---|---|
| 相对原始常相位 π 的相位 RMS 改变量，rad | 0.806523 | 0.382342 |
| 照明区域内相对 π 的 RMS 改变量，rad | 0.876972 | 0.673452 |
| 照明区域相邻像素相位差 RMS，rad | 0.141774 | 0.140192 |

`figures/phase_training_update.png` 比较本轮相位优化前后的 mask，并显示 wrapped 差值，原始数组在 `fields/phase_training_arrays.npz`。消融时固定电子权重、传播、探测与输入，替换指定阶段的相位。`pi_*` 表示物理相位 π，`zero_*` 表示物理相位 0，`flat_*` 表示各 tile 照明区域的圆均值。

## 运行

在本任务目录执行以下命令。

```powershell
pip install -r requirements.txt
python infer.py --weights runs/simulation/current/LightGen.pt --videos assets/example/native_test_video.mp4 --device cpu --output runs/simulation/current/validation/example_inference.json
```

`LightGen.pt` 包含推理所需的 patch 投影、位置编码、固定 prompt embedding、电子层和两级相位。输入为视频文件，输出 JSON 包含每个真实输入视频的 MOS。视频按给定顺序每 16 条组场，尾组从输入起始处循环补齐。

### 重绘

```powershell
pip install -r requirements-visualization.txt
python render_figures.py --paths paths.local.json --device cpu
```

`paths.local.json` 保存本机缓存路径。重绘脚本核对权重、数据清单和示例图片的身份，图输出到 `runs/simulation/current/figures/`。全部图使用 Arial，正文、坐标、刻度和图例为 7 pt，标题为 8 pt，PNG 为 300 dpi。相位通过 `atan2(sinφ, cosφ)` 显示在 −π 至 π，采用线性色标，刻度为 −π、0、π。

| 图 | 内容 |
|---|---|
| `network_overview.png`、`frame_dimensions.png` | 结构与尺寸 |
| `frame_global.png`、`video_global.png` | 两阶段的输入、调制后光场、探测强度及 CCD 处理 |
| `phase_masks.png`、`phase_training_update.png` | 当前相位与初始化对照 |
| `optical_expert_share.png` | 对应 Fig.2d 配色的专家选择份额 |
| `routing_example.png` | 同一视频四帧与图文融合的四专家 softmax 概率，边框标出 Top2 |
| `expert_matrix_and_tokens.png` | 专家矩阵、逐 token 输出及加权 RMS |
| `router_sensitivity.png` | 专家 softmax 概率对空间 token 的梯度 RMS |
| `global_layout.png` | 两级 8×8 与 4×4 输入区布局 |
| `robustness_curves.png` | 相位位移、CCD 位移和相位噪声测试 |

Optical expert share (%) 为某专家被 Top2 选中的次数除以全部选中次数。

| 阶段 | E1 | E2 | E3 | E4 |
|---|---|---|---|---|
| 视觉 | 26.79% | 25.09% | 23.34% | 24.78% |
| 图文融合 | 21.95% | 28.05% | 27.51% | 22.49% |

视觉阶段共 2232 次路由、4464 次 Top2 选择，各专家次数为 1196、1120、1042、1106。逐视频、逐帧及逐槽位记录保存在 `routing/`。

## 评价与日志

沿用 2250 TRAIN 与 558 条原 TEST。原 TEST 用于开发选模，完整 558 条主指标来自缓存特征复算。固定原输入顺序，每 16 视频组场，尾组有 2 条补齐输入，指标仅统计 558 条真实视频。

16 次循环换位的平均 SRCC 为 0.7901063063，最低为 0.7857706830。26 项鲁棒性条件为数值扰动测试。8 bit 相位量化的 SRCC 为 0.7989927886，检测面平移 2 像素条件中的最低值为 0.7823957337。

独立视频入口与缓存输入的逐视频核验记录保存在 `validation/`，包括权重和输入 SHA256、同一光场的预测对照及运行环境。

选定相位训练在 Windows、Python 3.11.9、PyTorch 2.11.0+cu128、RTX 4070 Laptop 上完成。训练源码提交、命令、缓存 SHA、初始化 SHA 和优化参数名单保存在 `metadata.json`，最佳轮次与独立 PT 的对应关系保存在 `selection.json`。

如需复训，可使用当前初始化及实际配置。

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
python run.py --config configs/train.json --paths paths.local.json --initialize assets/initialization/initialization.pt --train-soft-targets assets/training_only_teacher_predictions.pt --out runs/simulation/reproduce_finer_masks --device cuda --allow-shared-gpu
```

该示例会在运行元数据中记录桌面 GPU 的非独占资源模式。复算与可视化使用各自保存的完整分组及输入顺序。

## 归档

此前版本与其他训练尝试保存在 `E:\2026LightGen2Data\_internal_archives\t22_router_balance_20261112\`，包含上一版、此次训练尝试、逐文件 SHA256 与选择回执。活动结果统一位于 `runs/simulation/current/`。
