# 光学 MoE 与 Qwen3-VL 计时合同 V2

## 1. 为什么必须保留 bridge

bridge 是串行神经网络计算，不是图片排版或文件处理：

- LGVQ 时间质量的 `frame_to_video_bridge` 把 `[1,16,4,49,192]` 的逐帧视觉特征与 38 个 prompt token 合并为每个视频 42 个序列 token，同时生成下一光学阶段使用的场。
- LGVQ 空间质量的 `frame_to_sequence_bridge` 把四帧视觉网格和 prompt 合并为 `[1,42,192]` 序列。
- OpenMoji 的 `language_to_vision_bridge` 从语言特征生成视觉条件，并送入最终交互头。

这些模块的输出是后续阶段的输入，存在数据依赖。在没有执行流重叠证据时，不能声称它们被光传播或 language 分支覆盖。因此正式关键路径计入整个 required bridge。并行电子残差仍允许由物理光传播覆盖，但只从延迟中隐藏，其增量能耗仍计入。

## 2. Ours 的正式关键路径

正式主表采用 CUDA Event 中位数：

```text
T_ours = N × 1.314 ms 物理光场
       + Σ CUDA-event median(CCD 强度→归一化/读出/融合)
       + Σ CUDA-event median(required serial bridge)
       + CUDA-event median(完整任务头)
```

所有电子组件均在同一 A100 上以 `100 warm-up + 1000 measured calls`、`torch.inference_mode()`、逐次同步的协议测量。原始数据位于 `evidence/ccd_fusion_only_a100.json`、`evidence/task_heads_a100.json` 及三个功率 JSON 中。

| 任务 | 物理光场 | 串行电子（CUDA） | 正式 CUDA 总时间 | 同图同步 Wall 诊断 |
|---|---:|---:|---:|---:|
| LGVQ 时间质量，16 视频并行 | 7.884 ms | 3.217 ms | **11.101 ms** | 11.207 ms |
| LGVQ 空间质量，单视频 4 帧 | 7.884 ms | 2.962 ms | **10.846 ms** | 10.949 ms |
| ABO 图搜文 | 7.884 ms | 2.057 ms | **9.941 ms** | 10.027 ms |
| LSP 关键点 | 3.942 ms | 1.919 ms | **5.861 ms** | 5.913 ms |
| SALICON 显著性 | 3.942 ms | 2.183 ms | **6.125 ms** | 6.178 ms |
| OpenMoji 语义交互 | 7.884 ms | 3.575 ms | **11.459 ms** | 11.562 ms |

LGVQ 时间质量的逐项复算：

```text
7.884000 ms  = 6 × 1.314 ms 物理光场
1.892352 ms  = 2 × frame CCD→fusion + 2 × video CCD→fusion
0.464896 ms  = required frame→video bridge
0.860160 ms  = 完整 16 视频任务头
------------------------------------------------------
11.101408 ms = 正式 CUDA 关键路径
```

## 3. CUDA Event 与 Wall 的区别

- `CUDA Event` 在同一 GPU stream 上记录起止事件，得到 GPU 已排队计算体执行的时间；不含 Python 调度、CPU 发起 kernel 的开销以及最终主机同步等待的额外开销。它适合比较两套 GPU 神经计算，也作为论文主表口径。
- `synchronized Wall` 用主机单调时钟包住同一个调用，并在结束点执行 `torch.cuda.synchronize()`；它包含 Python/dispatcher、kernel launch 和同步开销，更接近软件调用者观察到的延迟。

二者必须分别成列，不能把一个组件的 Wall 值加到其他组件的 CUDA 值中。同步 Wall 通常略大；它作为实现诊断，不替换 CUDA 主表。

## 4. bridge 能否更快

可以工程化缩短，但不能直接从表里删掉。合法方向包括：

1. 把 bridge 的 LayerNorm/Linear 与上一 CCD readout 融合成一个 GPU kernel；
2. 预分配输出张量，避免每次创建中间 canvas；
3. 在输入 shape 固定时单独评估 CUDA Graph 或 `torch.compile`；
4. 后续重新训练时让 CCD readout 直接输出下一阶段所需的 42-token 表示，消除额外投影。

前三项不改变物理光路，但属于新的“优化实现”，必须重新计时并与当前 eager 基线并列报告。当前 bridge 仅占时间质量总时间约 `4.19%`，先保持保守计入更利于学术审计。

## 5. 当前组合能耗（已计 required bridge）

```text
E_total = 80.388 W × T_CUDA
        + P_A100,idle × T_CUDA
        + Σ(P_component - P_idle) × t_component,CUDA
```

并行残差的增量 A100 能耗计入，但不增加延迟。结果为 composed estimate，不是同一墙插功率计的直接测量。

| 任务 | 正式时间 | 光学+A100 组合能耗 | 组合平均功率 | A100=250 W 理论上界 |
|---|---:|---:|---:|---:|
| LGVQ 时间质量 | 11.101 ms | **1.616 J** | 145.57 W | 3.668 J |
| LGVQ 空间质量 | 10.846 ms | **1.562 J** | 143.98 W | 3.584 J |
| ABO 图搜文 | 9.941 ms | **1.433 J** | 144.10 W | 3.284 J |
| LSP | 5.861 ms | **0.852 J** | 145.31 W | 1.936 J |
| SALICON | 6.125 ms | **0.908 J** | 148.23 W | 2.024 J |
| OpenMoji | 11.459 ms | **1.659 J** | 144.76 W | 3.786 J |

精确未四舍五入值和逐组件字段见 `summary.json` 与 `ours_combined_energy.csv`；运行 `python build_report.py` 可从 evidence 重新生成。

## 6. Qwen3-VL 新正式测试合同

为了获得更完整且仍规范的 baseline 模型时间，主边界改为：

```text
所有 processor 输出 tensor 已在 GPU 上
→ 完整 Qwen3-VL forward（包含 Vision patch embedding）
→ 原生完整 vocabulary projection
→ 五个质量词行与加权连续质量分数在 GPU 上就绪
```

旧的“第一个 Vision block→分数”时间保留为 secondary diagnostic。不会添加 sleep、重复无意义层或循环同一个视频来拉长时间。

正式测试固定条件：

- LGVQ 固定 test：558 个视频；batch=1 为 558 次前向，batch=2 为 279 次前向；
- 单进程、单次模型加载；无显式 warm-up；第一个 test batch 计入统计；
- 每视频固定抽 4 帧；每帧取短边 `65%` 的中心正方形，再用 OpenCV `INTER_AREA` 缩放成 `448×448 RGB`；
- Qwen AutoProcessor 固定 `min_pixels=max_pixels=448²`，不开启 processor 二次抽帧；
- MP4 random-seek 解码、裁剪缩放、processor/tokenizer 和 H2D 分别记录，但不混入 model-only 主延迟；
- 功率以 50 ms 间隔保存逐采样 `nvidia-smi` 原始记录；平均功率、峰值、idle-subtracted 能耗和 250 W 上界同时输出；
- 每个样本保存源视频尺寸、帧数、FPS、选中的四个帧号、中心裁剪尺寸和输出尺寸；每个 batch 保存所有输入 tensor 的 shape/dtype/device。

新的 batch=1、2 原始运行目录为：

```text
/DATA/DATA1/guest3/2026OpticsMoE_a100_measurements/
  20260908_temporal_batch1_batch2_full_gpu_boundary/
```

同一批原始数据也按 `evidence/full_boundary_batch1/`、`evidence/full_boundary_batch2/` 和 `evidence/full_boundary_run_meta/` 收入本报告目录。`batch_timing.csv` 是逐批次 CUDA/Wall/旧边界数据，`predictions.csv` 是逐样本预测，`sample_preprocessing.jsonl` 是逐样本图像处理合同，`telemetry.csv` 是逐采样功率，`report.json` 是汇总。`summary.json/source_files` 保存本地 evidence 的逐文件 SHA256 与字节数；服务器打包时生成的原始 `SHA256SUMS.txt` 也原样保留。

## 7. batch=1、2 正式重测结果

两组均已在物理 GPU 6（A100-PCIE-40GB）上顺序完成。运行前等待该卡无其他 compute process；工作树 clean，commit 为 3fab90b363e3d49f02fa68acdc6c363b991097e7。两组使用相同模型、checkpoint、manifest、448×448 输入和 558 条 test。

| 项目 | batch=1 | batch=2 |
|---|---:|---:|
| GPU 前向调用数 | 558 | 279 |
| SRCC / PLCC | 0.766343 / 0.776053 | 0.766415 / 0.776848 |
| KRCC | 0.565454 | 0.565905 |
| RMSE / MAE | 8.7905 / 6.8384 | 8.7734 / 6.8165 |
| 完整模型 CUDA mean / batch | **56.766 ms / 1** | **70.464 ms / 2** |
| 完整模型 CUDA median / P95 | 53.748 / 58.088 ms | 65.872 / 70.185 ms |
| 同边界同步 Wall mean / batch | 56.817 ms | 70.514 ms |
| 旧 block-0 CUDA mean / batch | 53.108 ms | 63.417 ms |
| 吞吐率 | 17.616 video/s | 28.383 video/s |
| 等效 16 视频完整模型时间 | **908.248 ms** | **563.716 ms** |
| A100 active mean / peak | 160.545 / 200.010 W | 198.540 / 266.890 W |
| A100 idle mean | 53.710 W | 41.062 W |
| 额定功率平均占比 | 64.22% | 79.42% |
| 等效 16 视频实测 active 能耗 | **145.815 J** | **111.920 J** |
| 等效 16 视频 idle-subtracted 能耗 | 97.033 J | 88.773 J |
| 等效 16 视频 250 W 上界 | 227.062 J | 140.929 J |


正式均值不丢弃首批，也不做显式 warm-up。batch=1 的最大值为 948.642 ms，batch=2 的最大值为 1165.165 ms，均出现在首批；batch=1 的 batch index 13 另有一个 426.127 ms 的异常 CUDA 批。它们保留在逐批数据和 mean 中，没有人为删除，也不对异常来源作没有证据的归因。median/P95 用来描述常规批次，但不替换“启动一次后顺序跑完整 test”的全样本 mean。

batch=2 的 `nvidia-smi` 遥测中有一个 266.890 W 的瞬时 peak sample，高于设备报告的 250 W power limit；这不是平均功率，也不用于能耗积分。正式能耗使用同一次完整推理期间的 active mean 198.540 W。该 peak 只作为原始遥测事实保留，不据此声称 A100 持续超过额定功率。

与 bridge-inclusive Ours（16 视频 11.101408 ms、1.616046 J）相比：

| Qwen 执行方式 | Qwen 16 视频时间 | 加速比 | Qwen 16 视频能耗 | 能耗降低倍数 |
|---|---:|---:|---:|---:|
| batch=1，连续调用 16 次 | 908.248 ms | **81.81×** | 145.815 J | **90.23×** |
| batch=2，连续调用 8 次 | 563.716 ms | **50.78×** | 111.920 J | **69.26×** |

这两个 batch 应同时保留：batch=1 是严格不并行的单视频模型基线；batch=2 是同模型的小批并行基线。不能把 batch=1 的时间与 batch=2 的功率或性能拼在一起。

正式主表若只能放一组 Qwen 数字，采用 batch=1 的 16 次顺序调用结果；batch=2 单列为吞吐效率对照。这样能得到较长但仍真实、可复现的模型时间，而不是通过额外循环、sleep 或把预处理/I/O 混入模型边界来拉长时间。

## 8. 输入张量和数据规模

每个视频固定四帧。原始 558 个 test 视频的容器尺寸分布为：512×512 有 186 个、256×256 有 186 个、1408×768 有 93 个、672×384 有 93 个；原视频帧数范围 8–96，FPS 范围 4–24。逐帧精确裁剪框和抽帧位置已写入 JSONL。

Qwen processor 后的典型张量为：

```text
batch=1:
  pixel_values_videos [1568,1536] float32
  video_grid_thw      [1,3]       int64
  input_ids/mask      [1,446]     int64
  vision block 0      [1568,1024]
  language block 0    [1,446,2048]

batch=2:
  pixel_values_videos [3136,1536] float32
  video_grid_thw      [2,3]       int64
  input_ids/mask      [2,446]     int64
  vision block 0      [3136,1024]
  language block 0    [2,446,2048]
```

因此这里不是把原始 MP4 直接按 448×448 张量送入 Transformer；四帧先经过明确的中心裁剪和 resize，再由 Qwen processor 转为视觉 patch/token 张量。原始输入尺寸、处理后尺寸和模型内部 shape 均已保留。
