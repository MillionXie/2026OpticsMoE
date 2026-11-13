# T22 时间一致性父模型研究记录

本文件保留20261113父模型的研究过程及当时指标。20261114空间集中度微调后的
选定版本、指标及发行流程以任务根目录`README.md`和`configs/selected.json`为准。

推理与发布说明见根目录 `README.md`。本文件记录作者侧训练与核验流程。
训练、缓存准备和完整分析依赖通过 `pip install -r tools/requirements.txt` 安装。

当前唯一模型由 `configs/selected.json` 指定，结果与图位于 `runs/simulation/current/`。

## 当前版本

本轮将末级 global 的探测后投影合入读出首层，将读出头收窄至 32 个 ReLU 单元，
并使用探测强度除以区域均值的处理方式。文本适配器、定制 mean/max frame merger、
两级 router、四个线性专家和已学习的相位均沿用父模型。

模型共有 3,777,350 个参数。最后阶段仅微调读出头的 129,221 个参数，使用
2250 条 TRAIN 的缓存特征和 TRAIN 教师预测进行训练。32 个隐藏单元由 TRAIN
激活贡献选取，完成 400 轮后，依据用户指定的开发 SRCC 0.790–0.799 区间选择
第 290 轮，完整模型复算指标如下。

| 558 条原 TEST 开发评价条件 | SRCC |
|---|---:|
| 选中权重通过完整模型、缓存前端特征复算 | 0.7988976462 |
| 原始视频入口 | 0.7992454773 |
| 16 次循环槽位换位平均值 | 0.7953460324 |
| 16 次循环槽位换位最低值 | 0.7885972250 |
| 两级常相位 π | 0.7945205413 |
| 两级恒等传播 | 0.6436202435 |

正式参考指标采用完整模型复算。head-only 评价与完整模型的 GPU 矩阵批形状不同，
两者有轻微浮点差异。原仿真参考 SRCC 为 0.8043868643。

## 先看这些文件

| 文件 | 用途 |
|---|---|
| `configs/selected.json` | 唯一模型入口、SHA256、选中轮次和评价条件 |
| `runs/simulation/current/LightGen.pt` | 独立推理权重 |
| `runs/simulation/current/figures/` | 当前模型的全部图 |
| `runs/simulation/current/test_predictions.csv` | 558 条逐视频目标与预测 |
| `runs/simulation/current/phase_effect/` | 当前权重的常相位控制及逐视频结果 |
| `runs/simulation/current/global_ablation/` | 保留接口的相位、传播控制及随机种子汇总 |
| `runs/simulation/current/robustness/` | 当前权重的数值扰动结果 |
| `runs/simulation/current/routing/` | 专家选择次数及逐槽位统计 |
| `runs/simulation/current/metadata.json` | 本轮读出训练配置、源码、平台及数据身份 |
| `runs/simulation/current/history.json`、`logs/` | 训练曲线与原始日志 |
| `runs/simulation/current/selection.json` | 选中轮次、参数核验及最终权重身份 |
| `runs/simulation/current/validation/` | 完整前向、视频入口、槽位及部署验证 |

## 输入到输出

每个光场组织 16 个视频，每视频取 4 帧。输入顺序为 10%、36.67%、63.33%、90%
时间位置的四帧，取短边 65% 的中心正方形区域，转 RGB 并缩放至 448×448。

| 顺序 | 操作 | 每视频尺寸 |
|---|---|---|
| 1 | 冻结 Qwen patch 投影、位置特征及项目空间均值汇聚 | `4 × 49 × 1024` |
| 2 | 视觉 LN1024、Linear1024→192 | `4 × 49 × 192` |
| 3 | 每帧 router，LN192、Flatten9408、Linear9408→128、ReLU、Linear128→4 | 每帧 4 个 logits |
| 4 | 四专家 softmax、Top2、所选两项按和归一化 | 每帧两项非零 |
| 5 | 四个无偏置 Linear192→192 专家，按该帧权重汇合 | `4 × 49 × 192` |
| 6 | Linear192→27/ReLU，token 轴 Linear49→27/ReLU，RMS 规范化 | `4 × 27 × 27` 幅度场 |
| 7 | Vision global，相位调制、整场传播、强度探测及探测后处理 | `4 × 49 × 192` |
| 8 | 继承 LG2 定制 frame merger，空间 mean/max、LN384、Linear384→192、ReLU，加帧位置参数 | `4 × 192` |
| 9 | prompt 经 LN2048、Linear2048→192，与视觉 token 拼接并加序列位置参数 | `42 × 192` |
| 10 | 图文 router，LN192、Flatten8064、Linear8064→128、ReLU、Linear128→4，softmax 与 Top2 | 4 个概率、2 个激活专家 |
| 11 | 独立的一组四个无偏置 Linear192→192 专家，按每视频权重汇合 | `42 × 192` |
| 12 | Linear192→56/ReLU，补零到 56 行，RMS 规范化 | `56 × 56` 幅度场 |
| 13 | 图文 global，相位调制、整场传播、强度探测及探测后处理 | `42 × 96` |
| 14 | Flatten4032、Linear4032→32、ReLU、Linear32→5，softmax 与五等级分数加权 | 1 个 MOS |

视觉 router 与四个视觉专家在所有帧和视频槽位间共享参数。每帧独立选择专家，该帧
49 个 token 使用同一组权重。图文 router 读取 4 个视觉 token 与 38 个文本 token，
共 42 个 token，为每视频产生一组权重。两个阶段各有独立的 router 和专家。

两个 global 均使用 518×518 传播画布、中央 478×478 有效区，以及 16 块 111×111
相位。视觉 global 中每视频四帧按 2×2 排列，整场形成 8×8 帧输入区。
图文 global 中每视频对应一个输入区，形成 4×4 排列。传播参数为 532 nm 波长、
17 μm 采样、10 cm 距离和 0.5° 截止。

## 池化、LN 与投影的位置

探测强度首先按每个区域的均值归一化，分母下限为 10⁻⁶。

\[
I_{\mathrm{relative}}=I/\max(\operatorname{mean}(I),10^{-6}).
\]

视觉阶段的每帧 56×56 探测 ROI 随后经过 AdaptiveAvgPool2d，得到 49×96，
再经 LN96、ReLU、Linear96→192，返回 49×192 token。

图文阶段的每视频 115×115 探测 ROI 经过相同的强度均值归一化，池化为 42×96，
再经 LN96 和 ReLU，直接交给两层读出头。原末级 96→192 投影与读出首层的权重、
偏置完成代数合并，使读出输入由 8064 维变为 4032 维。

本轮在探测后处理移除了强度上限截断和 `log1p`。保留的池化承担探测网格到 token
布局的固定映射，LN 作用于每个池化 token 的 96 个特征。文本适配器保留运行时
LN 和 Linear，以支持后续 prompt 变化时的同型投影。

## 相位与训练来源

两级相位共 394,272 个原始参数，沿用选定父模型。父模型先优化视觉 router 与线性
专家的选择份额，再进行相位优化。其最初 raw 相位参数从 0 开始，经原 sigmoid
映射得到物理相位 π。父模型最后相位阶段完成 260 轮，选中第 86 轮。

本轮读出训练使用固定光学与电子前级的特征缓存。训练标签、隐藏单元贡献排序、
教师蒸馏损失均取自 TRAIN。原 TEST 仅用于开发评价和轮次选择。读出外保留参数的
逐张量核验以及父权重 SHA256 由 `selection.json` 和训练元数据记录。

当前相位的可视化范围覆盖全部 16 块相位 tile，显示采用线性 −π 至 π 色标。
相位、常相位及随机相位的性能评价均需通过本轮完整模型重新计算。

## 运行与重绘

在本任务目录执行以下命令。

```powershell
pip install -r requirements.txt
python infer.py --weights runs/simulation/current/LightGen.pt --videos assets/example/native_test_video.mp4 --device cpu --output runs/simulation/current/validation/example_inference.json
```

`LightGen.pt` 包含 patch 投影、位置编码、prompt embedding、文本适配器、电子层
和两级相位。输入为视频文件，输出 JSON 包含每个真实输入视频的 MOS。视频按给定
顺序每 16 条组场，尾组从输入起始处循环补齐，补齐样本从指标中排除。

```powershell
pip install -r tools/requirements.txt
python tools/render_figures.py --paths paths.local.json --device cpu
```

`paths.local.json` 保存本机缓存路径。重绘脚本核对权重、数据清单和示例图片的身份，
图输出到 `runs/simulation/current/figures/`。全部图使用 Arial，正文、坐标、刻度和
图例为 7 pt，标题为 8 pt，PNG 为 300 dpi。相位通过 `atan2(sinφ, cosφ)` 显示在
−π 至 π，色标刻度为 −π、0、π。

| 图 | 内容 |
|---|---|
| `network_overview.png`、`frame_dimensions.png` | 结构与尺寸 |
| `frame_global.png`、`video_global.png` | 两阶段的输入、调制后光场、探测强度及处理 |
| `phase_masks.png`、`phase_training_update.png` | 当前相位与其训练来源对照 |
| `optical_expert_share.png` | 对应 Fig.2d 配色的专家选择份额 |
| `routing_example.png` | 同一视频四帧与图文融合的四专家 softmax 概率，边框标出 Top2 |
| `expert_matrix_and_tokens.png` | 专家矩阵、逐 token 输出及加权 RMS |
| `router_sensitivity.png` | 专家 softmax 概率对空间 token 的梯度 RMS |
| `global_layout.png` | 两级 8×8 与 4×4 输入区布局 |
| `robustness_curves.png` | 本轮选中权重的相位位移、探测位移和相位噪声测试 |

Optical expert share (%) 为某专家被 Top2 选中的次数除以全部选中次数。
本轮逐视频、逐帧及逐槽位统计由选中权重重新生成，记录保存在 `routing/`。

## 评价与日志

沿用 2250 TRAIN 与 558 条原 TEST。原 TEST 用于开发选模。固定原输入顺序，每
16 视频组场，尾组有 2 条补齐输入，指标仅统计 558 条真实视频。

原始视频入口SRCC为0.7992454773，8 bit相位量化为0.7993216603，两像素探测位移中的最低值为0.7747974614。当前常相位和传播控制已完成复算。
槽位循环换位的平均值为 0.7953460324，最低值为 0.7885972250。
对应逐视频输出与核验文件分别写入 `validation/`、`phase_effect/`、
`global_ablation/` 和 `robustness/`，并记录选中权重的 SHA256。

本轮读出训练在 Windows、Python 3.11.9、PyTorch 2.11.0+cu128、RTX 4070 Laptop
上完成，采用桌面 GPU 非独占资源模式。实际命令、源码 SHA、父权重 SHA、缓存
身份、种子、优化参数和选中轮次分别保存在元数据、历史日志与选中记录中。

作者侧训练入口为 `tools/refine_electronics.py`，参数说明可通过以下命令查看。

```powershell
python tools/refine_electronics.py --help
```

复训初始化保存为 `assets/initialization/simplification_parent.pt`。实际训练命令如下。

```powershell
python tools/refine_electronics.py --weights assets/initialization/simplification_parent.pt --paths paths.local.json --out runs/simulation/reproduce_readout --hidden 32 --epochs 400 --learning-rate 0.0003 --teacher-weight 10 --minimum-srcc 0.790 --maximum-srcc 0.799
```

## 从随机初始化训练的电子对照

本轮按用户指定协议，冻结 Qwen 来源的视觉 `4×49×1024` 和文本 `38×2048`
embedding，将各自全部元素展平为 200,704 和 77,824 个值，拼接为 278,528 维。
输入经过随机初始化的 `Linear278528→32→ReLU→Linear32→5`，随后通过 softmax
和五等级分数期望输出 MOS。完整 flatten 是用户确认的输入形式，保留
embedding 的全部元素。

模型仅学习两层 FC 的权重与偏置，共四个张量、8,913,093 个参数。种子为 713，
AdamW 初始学习率为 0.0001，batch size 为 128，完成 400 轮 TRAIN 标签训练。
MOS 均值、标准差、等级分数和边界由 2250 条 TRAIN 标签重新计算。原 TEST
开发评价选取已评价轮次中的最高 SRCC，选中第 30 轮。

| 指标 | 当前光学参考模型 | 随机初始化两 FC 对照 |
|---|---:|---:|
| SRCC | 0.7988976462 | 0.7409566246 |
| PLCC | 0.8005052856 | 0.7736740243 |
| KRCC | 0.5864579021 | 0.5310321863 |
| RMSE | 8.4795709019 | 8.6899832681 |
| MAE | 6.3281161469 | 6.7342148579 |

当前光学参考模型继承已训练的 adapters、merger、router、expert 和相位，最后
阶段仅优化 129,221 个读出参数，并使用 TRAIN 教师蒸馏。解释两组精度差时需要
同时交代输入表示、初始化、训练目标、选模规则和更新范围的差异。

从随机初始化复现电子对照的命令如下，输出目录应为尚未使用的新运行目录。

```powershell
python tools/embedding_readout_control.py --input-mode flatten_concat --paths paths.local.json --out runs/simulation/reproduce_embedding_readout --hidden 32 --epochs 400 --batch-size 128 --learning-rate 0.0001 --seed 713 --device cuda
```

当前记录保存在 `runs/simulation/embedding_readout_control_20261113/flatten_concat/`，
包含配置、源码与缓存 SHA、初始权重指纹、逐轮日志、best/last 权重和逐视频结果。

### 历史任务 warm-start 容量检查

此前协议保留了父模型已训练的视觉与文本 adapters、mean/max frame merger、
位置参数，以及 8064→32→5 读出头。优化全部 942,341 个保留电子参数，完成
400 轮后，第 305 轮开发 SRCC 为 0.7937765654。其 PLCC、RMSE 和 MAE 分别为
0.8074254715、8.2045499911 和 6.2657914521。

| 历史候选 | 开发 SRCC | 初始化与训练范围 |
|---|---:|---|
| 32 头任务 warm-start 电子检查 | 0.7937765654 | 父任务层初始化，全部保留电子层更新 |
| 128 头光学参考候选 | 0.7989651231 | 父任务模型初始化，读出头更新 |
| 128 头任务 warm-start 电子检查 | 约 0.79764 | 父任务层初始化，全部保留电子层更新 |

128 头的任务 warm-start 电子检查与光学候选表现接近。这些历史协议的训练日志、
原始预测和结构配置保存在内部归档，保留各自的初始化与评价身份。

## 相位与传播控制

图文 global 接收 42×192 token，输出 42×96 token。前向控制保持该接口，分别
替换相位或传播算子，并保留电子输入映射、强度探测和探测后处理。常相位、随机
相位及恒等传播结果用于评价选中模型对这些算子的依赖。

常相位控制保持其他权重、传播、探测和输入，替换指定阶段相位。`pi_*` 表示物理
相位 π，`zero_*` 表示物理相位 0，`flat_*` 表示各 tile 照明区域的圆均值。
随机条件的种子、分布、逐种子指标和汇总表由各自协议记录。

```powershell
python tools/analyze_phase_effect.py --checkpoints runs/simulation/current/LightGen.pt --labels Current --paths paths.local.json --out runs/simulation/phase_effect_replay --device cpu
```

## 归档与发布

活动模型与结果统一位于 `runs/simulation/current/`。父版本、训练尝试和原始核验
材料存入 `E:\2026LightGen2Data\_internal_archives\` 下的任务归档目录，并保留
SHA256 清单及选择回执。公开发布入口为 `releases/current/` 和同名 ZIP。

无区间约束的 head128 候选曾在完整模型复算中得到 SRCC 0.8050366302。
该候选及训练日志进入内部归档，当前入口采用满足指定区间的训练轮次。

先生成公开包，再从包内的合并汇总表重绘控制结果。

```powershell
python tools/build_release.py
python tools/plot_global_ablation.py --summary releases/current/results/global_ablation_summary.csv --output runs/simulation/current/figures/global_ablation.png
```

发布权重使用公开元数据，模型张量与当前研究权重逐个核对。文件白名单、权重指纹、
逐文件 SHA256、示例推理及参数一致性验证记录在发布清单中。

此前限定 128 单元候选池的任务 warm-start 检查 SRCC 0.7842540330 随其协议
保存在内部归档。
