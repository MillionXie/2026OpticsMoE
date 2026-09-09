# A100 全任务统一计时报告

## 正式口径

- 光学 MoE：`CUDA Event`，物理传播 + CCD 后串行融合 + 必要 bridge + 完整任务头；并行电子残差不加延迟但计能耗。
- Qwen3-VL：`Synchronized Wall`，从第一个原生 Transformer block 输入到任务输出；预处理、H2D、模型加载不计入。
- 时间、功率和能耗必须来自同一方法与同一 batch。旧 CUDA 数字只作审计，不进入 Qwen 正式列。

## 当前可直接进入主表的数据

| 任务 | Ours Event | Qwen Wall | Ours 能耗 | Qwen 能耗 | 加速 | 节能 |
|---|---:|---:|---:|---:|---:|---:|
| LGVQ 时间，batch=1 等效16视频 | 11.101408 ms | 850.555926 ms | 1.616046 J | 136.552869 J | 76.6169x | 84.4981x |
| LGVQ 时间，batch=2 等效16视频 | 11.101408 ms | 507.724462 ms | 1.616046 J | 100.803728 J | 45.7351x | 62.3768x |
| ABO 图搜文 | 9.941216 ms | 43.975579 ms | 1.432520 J | 3.793213 J | 4.4236x | 2.6479x |
| ABO 图搜图 | 无正式光学图 | 49.189481 ms | — | 4.080771 J | — | — |
| LSP | 5.860976 ms | 18.027164 ms | 0.851649 J | 1.453343 J | 3.0758x | 1.7065x |
| SALICON | 6.125168 ms | 19.151612 ms | 0.907962 J | 1.552301 J | 3.1267x | 1.7097x |
| OpenMoji | 11.458784 ms | 49.053312 ms | 1.658809 J | 4.476462 J | 4.2808x | 2.6986x |

LGVQ 空间质量的旧 A100 原始报告只保存了 CUDA Event，没有保存 Synchronized Wall；正式 Wall 复测完成前，该格必须留空，不能把 74.437647 ms 的 Event 冒充 Wall。其 Ours Event 为 10.846432 ms、Ours 组合能耗为 1.561662 J。

## 性能

| 任务 | 指标 | Ours Sim. | Qwen3-VL |
|---|---|---:|---:|
| LGVQ 时间 | SRCC | 0.8044 | 0.7663 |
| LGVQ 空间 | SRCC | 0.6393 | 0.6908 |
| ABO 图搜文 | R@1 | 0.7983 | 0.7358 |
| ABO 图搜图 | R@1 | — | 0.9521 |
| LSP | PCK@0.2 | 0.7983 | 0.7226 |
| SALICON | CC | 0.8291 | 0.8810 |
| OpenMoji | changed-cell accuracy | 0.9800 | 0.5415 |

OpenMoji 的 Qwen 行必须写成 `Frozen Qwen3-VL-2B + 1.212M structured head`，不能将 0.5415 描述成充分微调后大模型的能力上限。

## 计算公式

`Qwen energy = 同次运行 active_mean_w × Synchronized Wall mean_ms / 1000`。时间一致性 batch=1 是 16 次顺序调用；batch=2 是 8 次、每次两个视频。Ours 能耗继续采用 80.388 W 光路加 A100 串行电子与并行残差增量功耗的组合代理。

## 原始证据

- LGVQ 时间逐批 Wall：`../a100_temporal_batch3_batch4_20260908/evidence/full_boundary_batch1/` 与 `full_boundary_batch2/`。
- 其余 Qwen：`../a100_audited_table_20260907/evidence/`；OpenMoji 见 `tasks/t04_semantic_interaction/reports/a100_formal/`。
- Ours Event 与组合能耗：`../a100_temporal_batch3_batch4_20260908/summary.json`。

所有未重测字段必须写 `pending`，不得用 Event、整段 test-loop wall 或跨任务平均开销替代 Synchronized Wall。
