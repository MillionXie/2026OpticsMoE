# A100 nine-metric timing and energy table — 2026-09-28

## 2026-10-23 GPU前端、Bridge与Tokens/s核查

本次新增 `frontend_tokens_20261023/九任务_核心与含前端_Tokens_20261023.xlsx`。
前两个工作表分别是**核心口径历史参考**与**含GPU前端的待补测口径**；后者不是已完成的新测量。
旧数据完整保留。缺失分段不填0、不以不同日期的运行拼接完整耗时。

### 计时边界

当前实现的 tokenizer 在 CPU processor 阶段，不是 GPU tokenizer。服务器
Transformers 4.57.3 的实际前向先执行 text embedding，再进入视觉编码器；视觉
patch embedding、位置编码准备也在首视觉 block 之前。因此旧 Qwen 首block计时未覆盖
这些GPU前端。纯文本查询的 embedding 同样在首语言block计时之前。
视觉与语言之间、首视觉block之后执行的原生merger则已经在Qwen旧总计时中，不能再加一次。
服务器源码SHA、相关行号及只读GPU占用快照保存在该目录的
`runtime_frontend_source_and_gpu_audit.json`。

Ours旧计时从预成形的GPU CCD/特征开始，也没有完整覆盖patch/position、embedding、
输入输出adapter。还发现旧SLM编码整体排除时，把其中的Linear、LayerNorm、softplus
一起排除了。扩展版需按真实前向补测这些必要神经网络，不能把它们都当作排版。
下列旧Bridge数字必须重新解释：

- ABO图搜图：实际执行`frontend.merge`，旧计时脚本没有注册该分段，原0代表漏计。
- ABO图搜文：替换Qwen保留主merger，旧通用计时没有单测，不能按0解释。
- LSP/SALICON：无vision→language任务桥接；但不等于没有GPU前端、adapter或实际执行的兼容merger。
- LGVQ与OpenMoji、文生图：已有的局部Bridge时间保留，但不代表已计全部前端。

另一个计算图修正是：采用`.8600`的ABO文搜图`cc977b83…`模型，在线title query
是纯文本，只经过language router/expert/global三次；六次是gallery图像的完整编码。
证据在`tasks/t08_abo_image_text_retrieval/physical10cm/cache_physical_readout.py`
与`export_full_next_stage.py`的title跳过vision合同。旧六次通用参考不得当作该在线query成本。
新工作表已把这一行光学部分改为3次，即3.1341 ms；电子部分尚缺匹配测量，不能只减三次光时间得到完整结果。

### 已留存的前端时间

旧九任务没有可独立还原的GPU embedding计时。LGVQ旧`preprocessing_ms`混合了解码、
processor及H2D，不可拿来充当GPU前端。
另一批10月20日LGVQ cap100/B1原始时间戳可作**同一运行的wall边界对照**：
三次运行各前200视频，共600次，首block至GPU结果78.168550 ms/视频，模型调用入口
至GPU结果84.254847 ms/视频，差6.086297 ms（约7.79%）。折16视频分别为
1250.696800与1348.077558 ms。这个差包含前端调度，并不是独立CUDA Event的embedding时间，
不能加到9月表的74.274134 ms上。

### 吞吐、能耗与算量

输入吞吐定义为 `Tokens/s = sum(N_input,i) / sum(t_i,s)`，不是各样本
`N_i/t_i`的平均。本表分别注明LLM输入序列、视觉patch及Ours特征槽位；后续光层
重新表征不再次累加。它们不是自回归生成token，也不能直接用不同粒度的原生token速率
比值代替任务加速比。缺少mask时不冒称有效非padding token。

例如旧Qwen LGVQ时间质量的200视频共89,200个LLM输入序列token，计时总和
14.85482683032 s，得到6,004.7822 tokens/s。每视频另有1,568个视觉patch，
对应21,110.9832 patch/s。Ours同一历史16视频形状是16×4×49=3,136特征槽位，
按10.245569591 ms参考总时间得到306,083.5特征槽位/s；两种原生粒度不同，不能据此宣称50.97倍任务加速。
ABO等变长输入按200条实际形状累计，不统一套最大长度。

两个版本均采用以下同边界计算（时间单位ms）：

`T = t_opt + t_elec`；`E_ours = (121.746*t_opt + 104.230*t_elec)/1000`；
`E_Qwen = (338.2 + P_A100,task)*t_wall/1000`；
`TOPS = TOP/(T/1000)`；`TOPS/W = TOP/E`；
`任务/s = 匹配任务数/(T/1000)`；`任务/J = 匹配任务数/E`。

旧OP按1实数MAC=2OP，Ours当前使用的`raw_remote/ours_ops/operation_counts.json`
已排除并行残差。更早的1,125,138,432电子OP含残差，不能混用。
但当前旧OP仍只是历史形状的MAC主项/光等效估算，非当前每个PT的完整算量；图搜图的
旧OP用77语言槽、旧统一计时用71槽。此外部分baseline算量只跑首样本，不能当作变长
200样本的平均算量。新增前端时也必须补计对应OP，而不是只改分母。
Embedding查表没有Linear/Conv式MAC不意味着没有时间或能耗。

本次没有执行新的GPU推理。两次占用检查均发现A100上其他用户任务，保留了快照。
扩展版缺前端/漏项NN计时及相应算量，因此完整新速度、能耗、TOPS及倍数保持待测，
不以0或不同run的差值补齐。

复核入口：`python LightGenV2/reports/20260928_nine_task_a100_table/audit_frontend_tokens.py`。
它只读取旧记录，输出含SHA的JSON与逐任务token累计CSV；表格公式由同目录
`frontend_tokens_20261023/build_workbook.mjs`生成。原始1600条带token形状的基线记录
与600条同源前端时间戳均不改写。

## 2026-10-23 version reconciliation

The user-selected nine-row performance table has been reconciled separately in
`version_audit_20261023/`. Its workbook is
`九任务版本核对与时间能耗_20261023.xlsx`; the Chinese methods note is
`九任务时间与能耗计算说明_20261023.docx`. These generated, private research assets
are not published in Git. The audit JSON and CSV in that directory preserve
performance sources, checkpoint identities, component means and calculation inputs.

The adopted text-to-image retrieval simulation is .8600, the exact historical LSP
simulation is .7347857143, and the .8880 OpenMoji full-head report exists. The
laboratory image-to-image result .81125 is verified against the sealed rank72
800-query report. The screenshot's .8150 external result remains a user-provided
display value until its exact original physical report is located.

The numerical timing/energy values below remain **historical component-cost
references**, not new measurements of those performance checkpoints. In particular,
the adopted .86 retrieval PT differs from the old .88 timing PT; the rank72 image
retrieval model adds a late RGB branch; and the .8880 semantic-editing PT differs
from the previously cited application checkpoint. Historical Ours measurements
repeat each fixed-shape GPU component 200 times, not 200 distinct test examples
through the full model. No new A100 run was performed in this reconciliation
because foreign compute jobs were active. The raw records below remain unchanged.

The Chinese note follows the stage-wise accounting in ACCEL Supplementary Notes
4 and 9, with editable equations and an explicit separation of parameter-derived
optical time, CUDA-event component measurements, and estimated energy. It retains
the agreed 80.358-W optical-device sum, 41.388-W controller, 62.842-W Ours A100,
338.2-W baseline chassis base and task-specific baseline board power, without
idle-power subtraction. Its cost scope is not an all-devices-on wall-plug integral.

## Historical workbook and measurement protocol

This directory is the single current location for the nine-row comparison table. The current workbook is `A100_nine_task_energy_efficiency_COMPLETE_20260928.xlsx`; the older same-date workbook without `COMPLETE` was open/locked during regeneration and must not be used. Earlier workbooks remain archived under `20260927_demo_energy_efficiency_a100/archive_superseded_20260927/`.

## Uniform timing scope

- Device: physical GPU 6, NVIDIA A100-PCIE-40GB; every formal run was audited with no foreign compute process.
- Samples: 200 measured calls; no explicit warm-up; the first call is retained.
- Ours electronic time: CUDA Event mean of router arithmetic, CCD nonlinearity/learned readout, learned bridge where present, and the actual task head. Parallel residual branches, host/device transfer, file I/O, SLM layout/encoding, CCD crop/stack, and RMS/alpha fusion are not included.
- Baseline time: synchronized wall mean from the first transformer block input through the final task output.
- Physical time: one optical pass is `0.714 + 0.300 + 0.0307 = 1.0447 ms`; multimodal tasks use six passes, while LSP and SALICON use three.
- LGVQ temporal is a matched workload of 16 videos. Its baseline per-video time is multiplied by 16; Ours already outputs 16 videos in parallel.

## Energy equations

Fixed measured powers:

- optical devices = 80.358 W;
- Ours control chassis = 41.388 W;
- Ours A100 board = 62.842 W;
- baseline chassis = 338.2 W;
- baseline A100 board = the task-specific `nvidia-smi` active mean.

Equations:

`E_ours = t_optical × (80.358 + 41.388) + t_electronic × (41.388 + 62.842)`

`E_baseline = t_baseline × (338.2 + P_A100,task)`

Times in the energy equations are in seconds.

## 2026-09-28 corrections

- ABO image-to-image now uses the current R@1 = 0.8225 checkpoint and the same narrow timing scope as the other ABO tasks. Its formal table value is 7.9384 ms, not the earlier 12.7124 ms broad-pipeline candidate. The broad candidate included the merger, 1600-gallery similarity, stable sorting, and other work outside the table scope.
- `raw_remote/formal_03_narrow/report.json` is retained as an independent identity/component audit. It starts several components independently and therefore charges multiple first-use CUDA initialization spikes; its summed 10.8691 ms is not the unified table statistic. The table uses the single unified 7-task process and the same sequential measurement order/protocol for every existing task.
- ABO clean text-to-image uses six optical passes. Its narrow electronic time is 8.5051 ms and total time is 14.7733 ms. The heavier time is from the current 256×256 RGB neural decoder task head, not from using the wrong GPU.
- ABO text-to-image retrieval (row 04) is now complete. The Ours result is bound to the exact 10 cm checkpoint with R@1 = 0.8800 (SHA-256 `8a96132d8ba68c671426d4e0b6064e87b5b3943505c0ad8b2aa694c879c20883`) and uses six optical passes. Its narrow CUDA electronic mean is 1.4004 ms, total latency is 7.6686 ms, and energy is 0.9091 J.
- The row-04 frozen Qwen baseline uses 100 official title queries against the precomputed 2400-image held-out gallery. Across 200 timed calls (the 100-query set repeated twice), R@1 is 0.8200, synchronized-wall mean is 32.4666 ms, active A100 board power is 65.0821 W, and energy is 13.0932 J. The resulting speedup is 4.2337× and energy-efficiency gain is 14.4025×.

## Evidence

- `raw_remote/ours_narrow_200/`: the unified 7-task A100 run, including 8,400 per-call rows.
- `raw_remote/formal_03_narrow/`: current ABO image-to-image identity and independent scope audit.
- `raw_remote/formal_04_ours/`: row-04 checkpoint identity, complete per-component timings, and occurrence contract.
- `raw_remote/formal_04_baseline/`: row-04 per-query wall/CUDA timings, raw 10 ms A100 telemetry, performance provenance, and gallery embedding hash.
- `raw_remote/formal_08_narrow/`: current ABO clean text-to-image narrow component audit.
- `raw_remote/formal_08_fullaudit/`: matched Qwen baseline timing, board power, checkpoint identity, and test manifest.

The workbook uses colors plus a textual `Basis/status` column: blue = measured, green = fixed hardware input, orange = formula-derived, and yellow = performance.
