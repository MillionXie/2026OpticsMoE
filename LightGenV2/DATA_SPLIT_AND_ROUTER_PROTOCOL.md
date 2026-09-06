# LightGenV2 数据划分、选模和 Router 统一口径

更新时间：2026-09-06。ABO 的 T07/T08 按当前要求暂不纳入。

| 任务 | 训练集 | 测试集 | validation | checkpoint 选择 | 我们的方法 |
| --- | ---: | ---: | --- | --- | --- |
| T01 Caltech101 物品检索 | 2,625 | gallery 30 + query 200 | 无 | 每 5 epoch 测 test，按 Top-1 最高 | 光 Router，4 专家 Top-2 |
| T02 LSP 关键点检测 | 10,428（LSPET 9,428 + LSP 前 1,000） | LSP 后 1,000 | 无 | 每 5 epoch 测 test，按 PCK@0.2 最高（同分依次看 NME、loss、较早 epoch） | 光 Router，4 专家 Top-2 |
| T03 SALICON 显著性 | 官方 train 10,000 | 官方 val 5,000 作为 public test | 无 | 每 5 epoch 测 public test，按 CC 最高 | 光 Router，4 专家 Top-2 |
| T04 OpenMoji 语义交互 | 5,000 | 1,000 | 无 | 每 5 epoch 测 test，按 changed-cell accuracy 最高 | 视觉/语言光 Router，各 4 专家 Top-2 |
| T05 视频分类 | 尚未建立正式数据 | — | — | — | — |
| T06 LGVQ 视频质量评价 | 2,250 videos | 558 videos | 无 | Spatial/Temporal 各自按对应 test SRCC 选模；常规每 5 epoch，Spatial 平衡微调阶段每 epoch | 两级光 Router，各 4 专家 Top-2 |

这里的 “Top-2” 只描述主方法：每个光 Router 从四个专家中激活两个。普通 D2NN baseline 是固定 dense phase stack，没有 Router；冻结 Qwen baseline 也没有 MoE Router。因此不能写成“三组都是 Top-2”。

## 三组比较的统一定义

1. Ours：光学 Router Top-2 + MoE 光相位 + global phase + 同尺度光电凸融合 + 20%–30% 相干零级分量鲁棒训练。
2. D2NN baseline：移除 Router 和条件专家选择，用普通 dense phase；其相位参数量匹配 Ours 每样本激活的专家相位参数量。只在当前服务器测性能，不把 CUDA 仿真耗时当物理光路速度。
3. Frozen Qwen baseline：任务对应的 Qwen 模型不微调；性能、从第一个原生 Transformer block 到输出的速度、整卡功耗全部留到 RTX 5090 D 测量。

## 统一解释

- 当前协议明确允许用周期 test 选择最佳 checkpoint，不主张它是无偏泛化估计；所有表格都应注明 `no validation / test used for selection`。
- 三组必须共享相同的 train/test manifest、输入预处理、任务指标和随机种子。
- 正式 run 只保留 `best_checkpoint.pt`、`last_checkpoint.pt`、指标 JSON、配置、日志和必要可视化。
- 表格中 Ours 的 `9.084 ms (6层)` 是既定光计算时间；D2NN 当前只填性能；冻结大模型三栏必须等 5090 D 实测，不能跨 GPU 外推。
