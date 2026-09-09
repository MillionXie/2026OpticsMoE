# A100 全任务正式计时、性能与能耗报告

## 最终统一口径

- 光学 MoE：采用较短的 `CUDA Event` 口径。总时间为固定物理传播时间，加 CCD 输出后必须串行执行的电子 bridge、非线性/归一化、融合和完整任务头的 CUDA Event 时间。可与光传播并行的电子残差不叠加到延迟，但它的增量能耗仍计入。
- Qwen3-VL：采用 `Synchronized Wall`。在第一个原生 Transformer block 前同步 CUDA 并启动主机计时器，在任务结果已经生成于 GPU 后再次同步并停止计时器。
- 两列故意使用用户指定的不同计时器，表头和论文中必须明确写出，不能将 Qwen 的 CUDA Event、整段测试循环时间或预处理时间混入正式速度列。
- GPU 均为 `NVIDIA A100-PCIE-40GB`。Qwen 能耗使用同一次运行的 A100 板卡平均功率；Ours 是 80.388 W 实测光路功率与 A100 电子部分组合得到的代理值，不是整机插座功率。

`CUDA Event` 主要记录 GPU 队列中的执行时间；`Synchronized Wall` 还包含所选边界内的 Python/CUDA 调度与同步等待。二者不是同一种时钟，因此本报告同时保留原始字段名和边界说明，避免误写成同口径硬件比较。

## 可直接进入主表的正式结果

| 任务 | 工作量 | Ours Event | Qwen Wall | Ours 能耗 | Qwen 能耗 | 加速 | 节能 |
|---|---|---:|---:|---:|---:|---:|---:|
| LGVQ 时间，batch=1 | 等效 16 视频，每视频 4 帧 | 11.101408 ms | 850.555926 ms | 1.616046 J | 136.552869 J | 76.6169× | 84.4981× |
| LGVQ 时间，batch=2 | 8 次调用，等效 16 视频，每视频 4 帧 | 11.101408 ms | 507.724462 ms | 1.616046 J | 100.803728 J | 45.7351× | 62.3768× |
| LGVQ 空间 | 1 视频、4 帧 | 10.846432 ms | 81.117794 ms | 1.561662 J | 6.990707 J | 7.4788× | 4.4765× |
| ABO 图搜文 | 1 图查询 100 个预计算标题 | 9.941216 ms | 43.975579 ms | 1.432520 J | 3.793213 J | 4.4236× | 2.6479× |
| ABO 图搜图 | 1 图查询 120 个预计算商品中心 | — | 49.189481 ms | — | 4.080771 J | — | — |
| LSP 关键点 | 1 图 | 5.860976 ms | 18.027164 ms | 0.851649 J | 1.453343 J | 3.0758× | 1.7065× |
| SALICON 显著性 | 1 图 | 6.125168 ms | 19.151612 ms | 0.907962 J | 1.552301 J | 3.1267× | 1.7097× |
| OpenMoji 语义交互 | 1 场景图 + 1 编辑指令 | 11.458784 ms | 49.053312 ms | 1.658809 J | 4.476462 J | 4.2808× | 2.6986× |

ABO 图搜图没有可运行且通过审计的正式光学计算图，因此 Ours 保持为空；这不是漏测，不能拿其他 ABO 模型或跨任务平均值补齐。

## 性能

| 任务 | 指标 | Ours Sim. | Qwen3-VL |
|---|---|---:|---:|
| LGVQ 时间 | SRCC | 0.8044 | 0.7663（batch=1）/ 0.7664（batch=2） |
| LGVQ 空间 | SRCC | 0.6393 | 0.6908 |
| ABO 图搜文 | R@1 | 0.7983 | 0.7358 |
| ABO 图搜图 | R@1 | — | 0.9521 |
| LSP | PCK@0.2 | 0.5773 | 0.7226 |
| SALICON | CC | 0.8291 | 0.8810 |
| OpenMoji | changed-cell accuracy | 0.9800 | 0.5415 |

OpenMoji 的 Qwen 方法必须完整命名为 `Frozen Qwen3-VL-2B + 1.212M structured head`。它不是充分微调后的 Qwen 能力上限；相同任务的匹配电子对照及光学消融见 `tasks/t04_semantic_interaction/reports/a100_formal/AUDIT_20260908_CN.md`。

LSP 的 Ours PCK@0.2 以 `t02_keypoint_detection/reports/dc20_comparison/comparison.json` 中 1000 个测试样本的 0.577286 为准；旧汇总表中的 0.7983 是误抄了 ABO 图搜文的 R@1，已更正，不能继续引用。

## 数据量、输入和在线边界

| 任务 | 性能测试量 | 正式计时量 | 模型输入 | 在线计时不含 |
|---|---:|---:|---|---|
| LGVQ 时间 batch=1 | 558 视频 | 558 次调用 | 每视频 4 帧，RGB 448×448；Vision block 0 输入 `[1568,1024]`，Language block 0 输入 `[1,446,2048]` | MP4 打开/定位/解码、中心裁剪与缩放、processor/tokenizer、H2D、patch embedding、模型加载 |
| LGVQ 时间 batch=2 | 558 视频 | 279 次调用 | 每次 2 视频；Vision `[3136,1024]`，Language `[2,446,2048]` | 同上 |
| LGVQ 空间 | 558 视频 | 558 次调用 | 每视频 4 帧，中心 65% 裁剪后 RGB 448×448 | MP4/解码、裁剪缩放、processor/tokenizer、H2D、patch embedding、模型加载 |
| ABO 图搜文 | 2400 图 | 200 图 | RGB 224×224、2048 维查询向量、100 个离线标题向量 | 文件/解码、processor、patch embedding、标题库预计算、模型加载 |
| ABO 图搜图 | 480 图 | 200 图 | RGB 224×224、2048 维查询向量、120 个离线商品中心 | 文件/解码、processor、patch embedding、gallery 构建、模型加载 |
| LSP | 1000 图 | 200 图 | RGB 224×224，输出 14 个关键点 | 文件/解码、图像预处理、patch embedding、模型加载 |
| SALICON | 5000 图 | 200 图 | RGB 224×224，输出 224×224 显著图 | 文件/解码、图像预处理、patch embedding、模型加载 |
| OpenMoji | 1000 样本 | 1000 样本 | RGB 224×224 + 编辑文本，输出 6×6 category/edit grid | 文件/解码、processor、H2D、模型加载 |

LGVQ 空间复测采用单进程、模型只加载一次、无显式 warm-up，首个测试视频保留。Wall mean/median/P95 分别为 81.117794/79.372149/99.713603 ms；对应 CUDA Event mean 为 80.967204 ms，仅作审计。空间质量 SRCC/PLCC/KRCC 为 0.690773/0.706545/0.500492。

LGVQ 时间的正式数值取逐批原始 CSV 中 `legacy_vision_block0_to_score_synchronized_wall_ms` 列，以保持“首个原生 Transformer block 到任务输出”的共同边界；同文件中的 `full_gpu_input_to_score_*` 是更宽的审计边界，不进入主表。

其余单图任务按各原始报告的正式计时子集执行。光学电子组件的原始基准使用每个 primitive 的 CUDA Event 中位数，并在组合报告中记录调用次数、功率和并行/串行归属。

## 光学 MoE 组合时间

| 任务 | 物理 pass | 固定传播 | 串行电子 Event | 合计 Event |
|---|---:|---:|---:|---:|
| LGVQ 时间 | 6 | 7.884 ms | 3.217408 ms | 11.101408 ms |
| LGVQ 空间 | 6 | 7.884 ms | 2.962432 ms | 10.846432 ms |
| ABO 图搜文 | 6 | 7.884 ms | 2.057216 ms | 9.941216 ms |
| LSP | 3 | 3.942 ms | 1.918976 ms | 5.860976 ms |
| SALICON | 3 | 3.942 ms | 2.183168 ms | 6.125168 ms |
| OpenMoji | 6 | 7.884 ms | 3.574784 ms | 11.458784 ms |

这里的 bridge 是模型内部必要的数据变换，例如 LGVQ 的 frame-to-video bridge 或 OpenMoji 的 language-to-vision bridge；它不是文件排版、BMP 重建或普通图像预处理，所以正式时间中保留。`router post`、下一张 SLM 布局/重建和文件处理明确排除。并行残差只从延迟中隐藏，不能从能耗中删除。

## 能耗公式

Qwen 使用同次运行的板卡功率：

`E_qwen = active_mean_power × synchronized_wall_mean / 1000`

时间质量的 16 视频工作量：batch=1 为 16 次顺序调用；batch=2 为 8 次、每次 2 视频。因此时间和能耗都按对应调用数等效为 16 视频。

Ours 使用组合代理：

`E_ours = 80.388 W × T_event_total + P_idle,A100 × T_event_total + Σ[(P_component-P_idle,A100) × T_component,event]`

可并行残差的增量 A100 能量进入求和，但其执行时间不再次加入 `T_event_total`。详细分项和额定功率上界保存在 `../a100_temporal_batch3_batch4_20260908/summary.json`。这是一种可复算的组合估计，不应写成实验室功率计直接测得的整机能量。

## 原始证据与复现索引

Provenance note: the spatial rerun reports a dirty server worktree because the timing script was deployed with SCP. Its recorded script SHA256 (`cf5467fd...d1d773b`) matches the locally committed script byte-for-byte. The archive is identified by script, checkpoint, manifest, and evidence-file SHA256 values.

- 本次 LGVQ 空间完整复测：`evidence/lgvq_spatial_wall/`，含 558 行逐视频预测与 Event/Wall 时间、50 ms 间隔板卡功率采样、完整报告及 `SHA256SUMS.txt`。
- LGVQ 时间 batch=1/2：`../a100_temporal_batch3_batch4_20260908/evidence/full_boundary_batch1/` 与 `full_boundary_batch2/`，包含逐批 `batch_timing.csv`、558 视频预测、逐样本预处理 JSONL、功率遥测、stdout/stderr 和 SHA256。
- Ours Event、bridge/融合/任务头分项与能耗：`../a100_temporal_batch3_batch4_20260908/summary.json` 及其 `evidence/optical_*_a100.json`、`task_heads_a100.json`。
- ABO、LSP、SALICON 的 Qwen A100 原始报告：`../a100_audited_table_20260907/evidence/`。
- OpenMoji：`tasks/t04_semantic_interaction/reports/a100_formal/`。
- 机器可读总表：`formal_table.csv`。

空间复测三个核心文件的 SHA256：

- `report.json`: `9858954de368171e4736607cd8f2340b871191799e10d3068b3576b73cdd20e6`
- `power_samples.csv`: `bf0856c64ffe2f68948dd47ea4a0b30a788b51c2840bfbf9eeea99633c0e2feb`
- `per_video_predictions_and_timing.csv`: `0ad9916b485f6de30cc42fe46ad5a18c0ae53ee45a2530af1a77cebe2b6e6791`

正式表不得用 CUDA Event 替换 Qwen Wall，也不得用整段 test-loop wall、跨任务平均开销或缺少模型图的推测值填空。
