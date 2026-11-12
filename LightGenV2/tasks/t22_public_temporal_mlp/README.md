# T22 时间一致性参考实现

当前唯一模型由 `configs/selected.json` 指定，全部结果与图位于 `runs/simulation/current/`。

## 当前版本

本轮沿用已训练的 T22 电子参数，将两级 `raw_global_phase` 全部置 0，再训练相位。相位映射为原来的 `2π·sigmoid(raw)`，初始物理相位为 π。

最终选中第 122 轮，完整预算为 200 轮。相对初始化，变化的模型张量为 `frame_global.raw_global_phase` 与 `video_global.raw_global_phase`。电子参数和前端 buffer 的逐张量核验保存在 `selection.json`。

| 558 条原 TEST 开发评价条件 | SRCC |
|---|---|
| 当前学习到的两级相位 | 0.7876236295 |
| 视觉相位替换为初始常相位 π | 0.7808204787 |
| 图文融合相位替换为初始常相位 π | 0.7819602196 |
| 两级相位同时替换为 π | 0.7776207227 |
| 两级相位同时替换为各 tile 的照明区域圆均值 | 0.7777879386 |

原仿真参考 SRCC 为 0.8043868643，本轮绝对差为 0.0167632348，处于既定 ±0.02 范围。电子参数全随机初始化的对照最好为 0.7615998407。对照及其他尝试已归档。

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
| `runs/simulation/current/validation/` | 复算、视频入口与初始化核验 |
| `assets/initialization/initialization.pt` | 两级 raw 参数归零后的实际初始化 |

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

本轮仅优化两级相位，共 394,272 个原始相位参数。模型总参数为 5,732,678。优化使用空间高斯梯度平滑、跨视频槽位的部分梯度平均，以及更新后的轻度原始参数平滑。完整数值与训练损失权重见 `configs/train.json`。

| 统计，覆盖全部 16 个相位 tile | 视觉 | 图文融合 |
|---|---|---|
| 相对初始 π 的相位 RMS 改变量，rad | 0.737768 | 0.309679 |
| 照明区域内的 RMS 改变量，rad | 0.780761 | 0.504561 |
| 照明区域相邻像素相位差 RMS，rad | 0.088143 | 0.058931 |

`figures/phase_training_update.png` 显示实际初始化、训练后的相位与 wrapped 差值，原始数组在 `fields/phase_training_arrays.npz`。消融时固定电子权重、传播、探测与输入，替换指定阶段的相位。`pi_*` 表示物理相位 π，`zero_*` 表示物理相位 0，`flat_*` 表示各 tile 照明区域的圆均值。

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

`paths.local.json` 保存本机缓存路径。重绘脚本核对权重、数据清单和示例图片的身份，图输出到 `runs/simulation/current/figures/`。全部图使用 Arial，正文、坐标、刻度和图例为 7 pt，标题为 8 pt，PNG 为 300 dpi。相位显示使用共同的单调 asinh 映射，范围为 0–2π，仅标注 0、π、2π。

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

Optical expert share (%) 为某专家被 Top2 选中的次数除以全部选中次数。视觉阶段 E1/E2/E3/E4 的份额为 38.87% / 13.37% / 8.53% / 39.22%，当前仍偏向 E1/E4。

## 评价与日志

沿用 2250 TRAIN 与 558 条原 TEST。原 TEST 用于开发选模，完整 558 条主指标来自缓存特征复算。固定原输入顺序，每 16 视频组场，尾组有 2 条补齐输入，指标仅统计 558 条真实视频。

16 次循环换位的平均 SRCC 为 0.7824810668，最低为 0.7778999435。26 项鲁棒性条件为数值扰动测试。

558 条原始视频已通过独立视频入口完整复算，SRCC 为 0.7875790110，记录保存在 `validation/native_full_558/`。同一服务器 CPU 的缓存输入复算为 0.7876337826，与训练平台保存的逐视频预测最大 MOS 差为 0.000031。视频输入与缓存输入的平均绝对 MOS 差为 0.006785，最大为 0.061352。首个 16 视频光场还通过了本机 `infer.py` 命令行入口核验，记录保存在 `validation/native_first_field/`。

选定相位训练在 Windows、Python 3.11.9、PyTorch 2.11.0+cu128、RTX 4070 Laptop 上完成。训练源码提交、命令、缓存 SHA、初始化 SHA 和优化参数名单保存在 `metadata.json`，最佳轮次与独立 PT 的对应关系保存在 `selection.json`。

如需复训，可使用当前初始化及实际配置。

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
python run.py --config configs/train.json --paths paths.local.json --initialize assets/initialization/initialization.pt --train-soft-targets assets/training_only_teacher_predictions.pt --out runs/simulation/reproduce_raw_zero --device cuda --allow-shared-gpu
```

该示例会在运行元数据中记录桌面 GPU 的非独占资源模式。复算与可视化使用各自保存的完整分组及输入顺序。

## 归档

此前版本与其他训练尝试保存在 `E:\2026LightGen2Data\_internal_archives\t22_raw_zero_20261112\`，包含 ZIP、逐文件 SHA256 与选择回执。活动目录统一使用当前模型及相应数据。
