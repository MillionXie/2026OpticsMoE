# T22 时间一致性参考实现

当前唯一模型由 `configs/selected.json` 指定。权重、数值记录和图统一放在 `runs/simulation/current/`。本目录由当前对话维护。

## 先看这些文件

| 文件 | 用途 |
|---|---|
| `configs/selected.json` | 当前权重路径、SHA256、架构和评价条件 |
| `runs/simulation/current/LightGen.pt` | 独立推理权重，包含所需前端组件和固定 prompt |
| `runs/simulation/current/figures/` | 当前模型的全部可视化 |
| `runs/simulation/current/metrics.json` | 完整 558 条开发评价指标 |
| `runs/simulation/current/test_predictions.csv` | 逐视频预测、目标与样本 ID |
| `runs/simulation/current/logs/` | 本轮训练启动记录和原始日志 |
| `runs/simulation/current/metadata.json` | 原训练配置、运行平台、源码身份与数据身份 |
| `assets/initialization/initialization.pt` | 本轮训练实际使用的初始化权重 |
| `configs/train.json` | 与该初始化配套的完整训练配置 |

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
  logs/
  fields/
  routing/
  routing_example/
  phase_effect/
  robustness/
  validation/
  figures/
```

`assets` 内的缓存用于复算完整数据集。`LightGen.pt` 已包含独立视频推理所需的 patch 投影、位置编码和固定文本 embedding。

## 当前架构

每个完整光场组织 16 个视频，每视频取 4 帧。视觉与文本在 Frame Merger 后拼接。

| 顺序 | 操作 | 每视频尺寸 |
|---|---|---|
| 1 | 四帧中心裁剪、缩放，提取冻结 Qwen patch 与位置特征，固定空间均值汇聚 | `4 × 49 × 1024` |
| 2 | 视觉 LN1024、Linear1024→192 | `4 × 49 × 192` |
| 3 | 每帧 router，LN192、Flatten9408、Linear9408→128、ReLU、Linear128→4 | `4 × 4` 个 logits |
| 4 | 四专家 softmax、Top2、所选两项按和归一化 | 每帧两项非零 |
| 5 | 四个独立无偏置 Linear192→192 专家，按该帧权重汇合 | `4 × 49 × 192` |
| 6 | Linear192→27/ReLU，沿 token 轴 Linear49→27/ReLU，RMS 规范化 | `4 × 27 × 27` 幅度场 |
| 7 | Vision global，相位调制、整场传播、强度探测、CCD 区域读出 | `4 × 49 × 192` |
| 8 | 原 Frame Merger，空间 mean/max、LN384、Linear384→192、ReLU，加帧位置参数 | `4 × 192` |
| 9 | 固定 prompt，独立 LN2048、Linear2048→192，与视觉 token 拼接并加序列位置参数 | `42 × 192` |
| 10 | 图文 router，LN192、Flatten8064、Linear8064→128、ReLU、Linear128→4，softmax 与 Top2 | 4 个概率、2 个激活专家 |
| 11 | 独立于视觉阶段的四个无偏置 Linear192→192 专家，按每视频权重汇合 | `42 × 192` |
| 12 | Linear192→56/ReLU，补零到 56 行，RMS 规范化 | `56 × 56` 幅度场 |
| 13 | Vision-language global，相位调制、整场传播、强度探测、CCD 区域读出 | `42 × 192` |
| 14 | Flatten8064、Linear8064→256、ReLU、Linear256→5，softmax 与五等级分数加权 | 1 个 MOS |

视觉 router 与四个视觉专家在所有帧和视频槽位间共享参数。每帧独立选择专家，该帧 49 个 token 使用同一组路由权重。图文 router 读取完整 42 个 token，为每视频产生一组权重，应用于全部图文 token。两个阶段各自拥有独立的 router 和专家。

两个 global 均使用 518×518 传播画布、中央 478×478 有效区和 16 块 111×111 相位。Vision global 中每视频的四帧按 2×2 排列，形成整场 8×8 帧输入区。Vision-language global 中每视频对应一个输入区，形成 4×4 排列。传播参数为 532 nm 波长、17 μm 采样、10 cm 距离和 0.5° 截止。

## 相位纹路的来源

历史名称 Joint42 表示图文 router 读取 4 个视觉 token 加 38 个文本 token。当前图中统一使用 Vision 与 Vision-language 标识阶段。

当前模型采用透镜先验初始化后优化。视觉相位继承此前已优化的透镜先验，融合相位使用平滑透镜先验和小幅残差重新初始化。最终环状纹路保留了大部分先验结构。本轮选中权重相对实际初始化的 wrapped 相位 RMS 改变量，按全部 16 块相位统计，视觉为 0.016603 rad，图文融合为 0.056498 rad。潜在照明区域内分别为 0.024603 rad 和 0.025529 rad。

`figures/phase_training_update.png` 同列显示实际初始化、最终相位与 wrapped 训练改变量。原始数组在 `fields/phase_training_arrays.npz`，统计范围与数值在 `fields/phase_training_statistics.json`。

固定电子权重、传播及探测，只将对应相位替换成各 tile 潜在照明区域的常相位圆均值，558 条开发评价得到以下结果。

| 相位条件 | SRCC |
|---|---|
| 当前完整模型 | 0.793315 |
| 视觉相位替换为常相位 | 0.786747 |
| 图文融合相位替换为常相位 | 0.788084 |
| 两级同时替换为常相位 | 0.777637 |

逐条件结果和预测保存在 `phase_effect/`。该对照反映固定模型对当前相位变换的依赖。环状结构的初始化来源由初始化对照图单独记录。

## 运行

从本任务目录执行以下命令，使用 Python 3.11 与支持当前设备的 PyTorch。

```powershell
pip install -r requirements.txt
python infer.py --weights runs/simulation/current/LightGen.pt --videos assets/example/native_test_video.mp4 --device cpu --output runs/simulation/current/validation/example_inference.json
```

输入为视频文件，按 10%、36.67%、63.33%、90% 的位置抽取四帧。每帧取短边 65% 的中心正方形区域，转 RGB 并缩放至 448×448。输入按原顺序每 16 个视频组成一场，尾组循环补齐，输出仅保留真实输入视频的 MOS。

### 重绘当前全部图

```powershell
pip install -r requirements-visualization.txt
python render_figures.py --paths paths.local.json --device cpu
```

`paths.local.json` 保存本机缓存路径。脚本核对权重 SHA、数据清单与示例图片身份，使用保留的光场数组、完整路由统计及当前权重重绘，输出到 `runs/simulation/current/figures/`。

全部图使用 Arial，正文、坐标、刻度和图例为 7 pt，标题为 8 pt，PNG 为 300 dpi。运行环境需提供 Arial 字体。相位色标统一使用单调 asinh 映射，范围为 0–2π，仅标注 0、π、2π。原始相位数组保留弧度值。

| 图 | 内容 |
|---|---|
| `network_overview.png`、`frame_dimensions.png` | 网络顺序和尺寸 |
| `frame_global.png`、`video_global.png` | 两阶段的输入、调制后场、探测强度与 CCD 处理结果 |
| `phase_masks.png`、`phase_training_update.png` | 当前相位与初始化对照 |
| `optical_expert_share.png` | 按 Fig.2d 配色绘制的专家选择份额 |
| `routing_example.png` | 同一视频四帧及图文融合的 softmax 概率，边框标出 Top2 |
| `expert_matrix_and_tokens.png` | 专家矩阵、逐 token 输出和加权 RMS |
| `virtual_expert_activation.png` | 四专家在特征空间中的虚拟布局 |
| `router_sensitivity.png` | 各空间 token 对专家 softmax 概率的梯度 RMS |
| `global_layout.png` | 两阶段 8×8 与 4×4 输入区的组织方式 |
| `robustness_curves.png` | 相位位移、CCD 位移及相位噪声的数值测试 |

Optical expert share (%) 定义为某专家被 Top2 选中的次数除以全部选中次数。558 个视频对应 2232 次视觉路由和 558 次图文路由，每次贡献两次选择。

## 当前结果与基本日志

本轮在 Windows、Python 3.11.9、PyTorch 2.11.0+cu128、RTX 4070 Laptop 上完成 150 轮优化，选中 epoch 35。原运行标识及源码提交保存在 `metadata.json` 与 `logs/`。模型共有 5,732,678 个参数，本轮参与优化的参数为 3,531,049，独立权重为 27,880,416 字节。

| 评价 | 数值 |
|---|---|
| 原顺序、固定 16 视频分组，558 条开发 SRCC | 0.7933148397 |
| 16 次循环换位平均 SRCC | 0.7891963522 |
| 16 次循环换位最低 SRCC | 0.7864636515 |
| 视觉专家 E1/E2/E3/E4 选择份额 | 38.87% / 13.37% / 8.53% / 39.22% |
| 图文专家 E1/E2/E3/E4 选择份额 | 20.79% / 29.21% / 27.15% / 22.85% |

沿用 2250 条 TRAIN 与 558 条原 TEST，原 TEST 用于开发选模。完整 558 条指标由缓存特征复算。当前权重的原始视频 CLI 核验覆盖首个 16 视频光场，记录在 `validation/native_first_field/`。26 项鲁棒性条件为数值扰动测试，记录在 `robustness/`。当前视觉专家选择仍较偏向 E1/E4。

## 目录维护

本轮归档位于 `E:\2026LightGen2Data\_internal_archives\t22_cleanup_20261112\`。活动目录只保留当前模型、其初始化、所需缓存和相应证据。归档包含旧目录快照、逐文件 SHA256 与清理回执。
