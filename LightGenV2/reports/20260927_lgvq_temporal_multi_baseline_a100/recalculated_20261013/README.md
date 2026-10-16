# LGVQ temporal quality: frozen A100 costs and independent resident utilization

## 16 October 2026: CLIP saturation extension and Qwen execution diagnosis

The current figures and three workbooks now contain **31 electronic model–batch points plus one LightGen-II reference**. The original30 accuracy/cost records and90 utilization repeats remain unchanged. The new CLIP B=11 row is appended in green in `plot_data.xlsx`; raw repeats are appended in the utilization workbook. This extension used the same frozen PT, original4-frame inputs and measurement loops, without training, Graph capture, clock/power changes or selection of a high-utilization subwindow. Qwen B=1 remains63.133111480865225%,1188.3861464262009ms and462.1134010767563J from its declared historical sources.

| New model–batch point | Resident repeat1 (%) | Repeat2 (%) | Repeat3 (%) | Pooled utilization (%) | Full558 SRCC | Full558 PLCC | T16 (ms) | E16 (J) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CLIP ViT-B/32, B=11 | 99.6806722689 | 99.3819095477 | 99.7231543624 | 99.5950782998 | 0.725202866664 | 0.728218319846 | 34.6894103289 | 13.2737767190 |

This was the first new batch tested above B=10, so the 98% stopping criterion was met without testing B12–16. The three complete windows total 90.108401 s, 1,788 driver samples and 14,680 forward calls. Each call processes 11 videos, not 88 videos; queue8 means eight consecutive calls on the same stream. The prior B10 result of 94.89367655% remains a different-date observation, not a same-round threshold bracket or proof of a monotonic batch response.

Accuracy/cost were measured in a separate fresh process on all558 unique TEST videos. The first200 determine costs:19 calls including a2-video tail, followed by33 further TEST calls including a6-video tail. Their synchronized inference times sum to433.617629110813ms, giving `T16=433.617629110813×16/200`. The same cost run has11 inference-tagged power samples with mean44.4463636364W; `E16=(338.2+44.4463636364)×T16/1000`. This is the user-approved host-proxy and sampled-board-power estimate, not an integrated energy-meter measurement. Short calls and the driver's averaging window limit attribution of instantaneous power. The first call248.082ms is included under the unchanged no-explicit-warmup rule. The99.60% utilization belongs to the independent resident load, not to this interleaved cost run; neither its utilization nor its approximately247W board power is substituted into E16. Higher resident utilization does not guarantee lower cost: the retained B10 costs are34.107ms and12.895J, slightly lower than this new B11 observation.

### Qwen: what the native execution trace establishes

Four diagnostic points used the original first B TEST inputs prepared and copied to GPU once, then eight native forwards. Input preparation and MOS transfer to CPU are outside the profiler window. Native outputs before/after profiling match exactly at the recorded FP32 MOS output. No weights or model computation were altered. Correlation IDs link scalar-sized device-to-host copies to `aten::_local_scalar_dense` and `aten::nonzero` inside forward.

| Video batch B | GPU scalar reads (`aten::item`) per forward | Device-to-host events per forward | CUDA stream synchronizations per forward |
| --- | ---: | ---: | ---: |
| 2 | 58 | 143 | 148 |
| 4 | 108 | 199 | 206 |
| 8 | 208 | 311 | 322 |
| 10 | 258 | 367 | 380 |

Thus, removing external decoding/H2D does not remove Qwen's internal host–device dependencies. These operations increase with video batch rather than being fully amortized. Source inspection also identifies dynamic video-position/geometry preparation in the official native model. This is evidence for a synchronization-overhead contribution to the observed utilization plateau, **not a causal ablation proving its unique cause or predicting98% after optimization**. The GPU kernel-union fractions of the instrumented traces (55.72%,73.06%,79.86%,81.08%) are not NVML utilization; profiler overhead prevents using them as formal performance or cost values. The one-repeat unprofiled diagnostic controls are also separate audit records, not replacements for the original three-repeat table. No accuracy correction was reopened.

For manuscript use: “An additional CLIP batch of 11 videos reached a pooled resident GPU utilization of 99.60% across three complete 30-s measurement windows, following at least 5 s warmup per repeat. Frozen evaluation on 558 LGVQ test videos yielded an SRCC of 0.7252. Under the separately measured original inference-cost protocol, processing time and estimated energy normalized to 16 videos were 34.689 ms and 13.274 J, respectively. Native Qwen execution traces revealed batch-dependent scalar reads, dynamic indexing and stream synchronizations despite GPU-resident input, indicating internal host–device dependencies that are not removed by eliminating external preprocessing. These diagnostic traces were not used to replace the finalized comparison measurements.”

Evidence is in T06 `runs/simulation/lgvq_clip_saturation_qwen_plateau_a100_20261016/continuation01`: `verified_costs.json`, `qwen_trace_audit.json`, original reports/telemetry, four native traces and SHA receipts. All GPU phases completed; the measurement context exited. The first parent run's earlier CPU-only idle timeout is preserved. To reproduce the updated figures, use `plot_current_comparison.py --clip-extension` (add `--cost-only` to leave the utilization figure unchanged). The following30-point descriptions and tables describe the preserved original cohort unless stated otherwise.

## Original30-point cohort

Updated 16 October 2026: all six models and all thirty previously measured batch configurations are complete. Each configuration has 558 frozen TEST predictions, a first-200-video cost record, matched power telemetry and a SHA-backed raw-data receipt. Only the interrupted GLM B=2 was retried; its failed evidence and the completed prefix were preserved. No new training, precision change, CUDA Graph or batch expansion was introduced.

Current [figure data](plot_data.csv), [full costs and telemetry](timing_energy_recalculated.csv), [performance audit](batch_metric_audit.csv), [workbook](LGVQ_temporal_multi_baseline_recalculated.xlsx) and [two-panel figure](srcc_cost_comparison_reference.png) are in this same directory. The methods, complete numerical table and manuscript-ready descriptions are under “两图含义与实验方法” below. Qwen B=1 retains the finalized historical cost, explicitly distinguished from its new cost audit. LightGen-II retains the approved physical-performance reference and component-cost estimate.

GPU-utilization labels use independent native resident queue8 experiments, not the utilization of the dataset-once cost run. Following the user's explicit finalization on 16 October, Qwen B=1 displays its historical **63.133111480865225%** single-run result; all other points retain the uniform three-repeat pooled means. The newer Qwen B=1 pooled result, 71.17536071032187%, remains an audit record, not the selected display value. The earlier execution/Graph controls and precision diagnoses below are preserved historical evidence. No precision diagnosis was reopened in this update. The original three-sheet cost workbook structure is retained.

## 完整比较图与绘图入口

完整图包含LightGen-II及全部六个电子模型：Qwen3-VL-2B、GLM-4.6V-Flash、
DeepSeek-VL2-Small、CLIP ViT-B/32、YOLO11s、DeepSeek-VL2-Tiny。
原版每个面板均含30个电子model/batch点和1个光学参考；本次追加CLIP B11后为31个电子点和1个光学参考。不只展示Tiny版DeepSeek，
也没有遗漏GLM的B1/B2或Small的B1/B2/B3/B4。

- 双面板：[PNG](srcc_cost_comparison_reference.png)、[可编辑SVG](srcc_cost_comparison_reference.svg)。
- 时延单图：[PNG](latency_comparison_reference.png)、[可编辑SVG](latency_comparison_reference.svg)。
- 能耗单图：[PNG](energy_comparison_reference.png)、[可编辑SVG](energy_comparison_reference.svg)。

PNG为300 dpi，横轴分别为每16视频时间(ms)和能耗(J)，采用对数轴完整覆盖所有点，
纵轴为558 TEST上的Temporal SRCC(0–1)。标签给出B和独立resident利用率，
不把利用率代入成本；Qwen B1定稿锚点及LightGen-II参考性质不变。
可复现绘图入口为本报告上级目录的`plot_current_comparison.py --cost-only`，
从受保护测量JSON及批准展示映射读取，并逐项核对`plot_data.csv`后绘制。
原旧`build_analysis.py`不作为本轮绘图入口，不能导入执行以免覆盖历史。

`plot_data.xlsx`已按服务器SHA核验的上一版完整恢复，保留原颜色、公式、batch列及来源；
误改版先行备份在canonical成本run的`presentation_prior_20261016/accidental_edit_48763b6ec126_plot_data.xlsx`。
当时恢复文件SHA为`342e1814ba7bc185b16dfb1578bdd5b0c58a10a41d2d3cc1bc8a03c44351046b`，CSV当时未改变。
此后追加B11导致当前展示文件SHA更新，旧展示版已在本轮接续run的`prior_report_archive`逐SHA保留。
恢复收据与原图输出SHA收据仍在原canonical run，不改原测量。

## 2026-10-16：教师展示版、定稿锚点与倍数定义

本轮只整理已经测完的数据，没有启动新的 GPU 测试或训练。优先展示文件为同目录的
[彩色完整数据表](plot_data.xlsx)，共六模型30种batch及LightGen-II参考；
[绘图输入CSV](plot_data.csv)含相同完整精度数据与来源。CSV不支持颜色，故保留机器可读CSV，
同时提供可筛查、按模型分色的Excel展示版，不另放outputs目录。

Qwen B=1恢复用户定稿的63.133111480865225%，对应原记录
`lgvq_utilization_execution_20261015/unified_qwen_b1_5s30s01/report.json`：
一次原生resident queue8测量，实际预热5.060464秒、完整测量30.241928秒、601个利用率样本、576次前向。
它与后来三次重复测量使用相同的基本驻留输入/原生queue8方式，但重复次数和测量时刻不同，
不能称为全部同轮或三次均值。本次是用户指定历史测量为展示锚点，不是把71.17536071032187%的
原始复测改写成63.13%。原始`verified_costs.json`、`verified_summary.json`、90份重复记录、
逐样本精度、时延及能耗均未改变。Excel、展示CSV和图中已明确该单行来源例外；
`timing_energy_recalculated.csv`与`recalculation.json`仍保留测量审计值，不冒充展示选择。

每个batch都列出以下两组倍数，避免“加速比”指代不清：

- **LightGen-II时间加速比** = 当前基线T16 / LightGen-II T16；
  **LightGen-II能效优势** = 当前基线E16 / LightGen-II E16。
  两者比较相同16视频工作量；后者是每任务能耗比，不是TOPS/W。
- **同模型相对B=1的时间/能效改善** = 该模型B1的T16或E16 / 当前B的对应值。
  Qwen的B1分子仍采用定稿历史成本，其他模型用配套新成本。

例如Qwen B=1的定稿T16=1188.3861464262009 ms、E16=462.1134010767563 J，
对应LightGen-II时间优势**115.9902468894624×**、能效优势**392.3898423712127×**。
Qwen B=8则分别约为49.83×、196.18×，完整未舍入计算在表中，不因恢复利用率改变成本。

### 参数量口径

| 模型 | 总参数规模 | MoE激活参数规模 | 依据与限制 |
| --- | ---: | ---: | --- |
| Qwen3-VL-2B | 约2B | 不适用 | [官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)，标称完整模型规格 |
| GLM-4.6V-Flash | 约9B | 不适用 | [官方模型卡](https://huggingface.co/zai-org/GLM-4.6V-Flash)，Flash标称规格 |
| DeepSeek-VL2-Small | 约16B | 约2.8B | [官方模型卡](https://huggingface.co/deepseek-ai/deepseek-vl2-small)，总量与激活量分列 |
| CLIP ViT-B/32 | 151,277,313 | 不适用 | 冻结特征收据实载总量，包含本次视觉推理不执行的文本分支 |
| YOLO11s | 9,458,752 | 不适用 | 冻结特征收据实载完整检测模型；前向运行原模型并读取C2PSA特征 |
| DeepSeek-VL2-Tiny | 3,370,501,440 | 约1.0B | 冻结特征收据实载总量；[官方激活量](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny) |

参数以模型本体口径列示，不把小质量头、显存、每次执行参数量或光学等效连接混在一起。
CLIP/YOLO/Tiny的精确实载计数位于原协议重训run的`features/*report.json`，逐项路径列在展示表Measurements。
LightGen-II光学自由度/等效连接与GPU存储权重不是同一种数量，故本次不填一个貌似可比的总参数。

### 为什么Qwen不随batch继续趋近100%

当前曲线：Qwen B1（历史展示）63.13%、B2 87.27%、B3 89.27%、B4 91.07%、B6 91.38%、
B8 91.63%、B10 90.63%。因此它是在本次已测范围进入约91%的平台，而非硬件上限只有90%，
也不保证batch增大必然单调。其他模型中GLM B2为98.07%、Small B4为98.50%、YOLO B3为98.43%；
CLIP B10实际仍为94.89%，Tiny B8为97.78%，不能一概说所有其他模型都达到近100%。

这里是[NVIDIA定义的驱动采样窗口内至少一个kernel运行的时间占比](https://docs.nvidia.com/deploy/archive/R550/nvml-api/structnvmlUtilization__t.html)，
不是SM occupancy，也不是峰值FLOPS利用率。驻留输入只排除了反复解码/预处理/H2D的供数等待，
没有消除原生模型内部的主机调度、动态准备和同步间隙。模型的算子组合、形状、单次计算持续时间及
执行栈不同，参数多寡不能单独预测这个百分数。较小batch可能让调用间隙占比更高，增大后间隙
相对变小；继续增大也可能进入平台，而不必到100%。这是与现有数据一致的解释，不是已完成的
Qwen全路径瓶颈归因。

旧Qwen B8对照中，固定5–30秒原生负载约91.89%，去CPU输出约92.06%，queue8约92.30%，
说明仅去掉末端CPU拷回或多排几个前向并未使该原生路径升至98%。不能用CLIP/YOLO的Graph
对照值替换当前原生协议，也不能只挑Qwen最高窗口；本次未修改输入、精度、模型或采样窗口来凑高值。

## 最新因果隔离：相同输入、不同批形状的运算差异（15 October 2026）

以下补充不是正式A100复测。用户允许利用空闲4090/3090排查、避开另一窗口A100
测速，因此使用物理RTX4090 `GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd`，
Torch2.6cu124/CUDA12.4、原部署BF16权重。`operator_mechanisms04/rtx4090_qwen`
与`rtx4090_deepseek`执行main `f405eaf22`，随后针对性固定输入隔离
`rtx4090_deepseek_isolated`执行`34e61ca2f`；各有600秒上限、单卡串行并已实际
完成/退卡。原A100派发因共享锁busy在CUDA初始化前退出，没有产生新的A100结果。
正式六份数据文件仍逐SHA一致，不将跨卡探针、固定输入重复行或数值控制填入图表。

### Qwen：矩阵形状确实改变实际计算内核

取此前A100保存的真实视频0首deepstack merger fc1操作数，先按部署语义量化BF16，
固定相同输入/权重/偏置，把同一[392,4096]输入重复成392/784/1568行。
RTX4090 profiler实际记录的原生GEMM核分别含
`128x64_sliced1x2`、`128x64`、`64x128_sliced1x2`，完整名称见
`qwen_operator_gpu.json`，证明本探针的矩阵形状会切换执行核，而非仅凭源代码推测。
这些核名不能反推原A100执行了同一核，也不能仅凭`sliced`断言全部split-K舍入机制。

| Qwen孤立fc1，B4相对自身B1 | 不同元素/总数 | RMS绝对差 | 最大绝对差 |
| --- | ---: | ---: | ---: |
| 原生BF16 linear | 422346/1605632 | .009572562 | .125 |
| BF16 matmul与bias显式分开 | 558/1605632 | .000330096 | .125 |
| 操作数转FP32计算、再回BF16 | 118/1605632 | .000078509 | .0625 |
| 仅关闭BF16低精度归约开关 | 422346/1605632 | .009572562 | .125 |

FP32控制没有恢复已量化丢失的位，也没有改变权重数值；差异明显减少但仍非零，不能
称逐位稳定或完整模型修复。该孤立算子的控制结果不是8视频或558视频SRCC结果。
CPU先前检验的简单两次bias舍入及2/4/8/16段BF16部分和均未精确复现原A100 B1，
保留否定证据，不把未经验证的具体累加方式写成结论。

### DeepSeek：上游继承误差与注意力自身误差已分离

先在RTX4090整模型同8视频/B1/2/4记录首语言注意力内部算子；原生无观测/有观测
MOS逐位一致，排除本次观测改变评分。单独把语言attention GEMM转FP32并未稳定
整模型：B4/B1最大MOS差由.7458458变为1.4039307，不能作为修复采用。

进一步在原生首注意力pre-hook保存真实hidden/mask/position并受控停止forward，
同时检查视频0的上游边界。4090上B2最早已观测差为首vision block注意力输出
（max_abs=.015625），B4为首vision block MLP输出（.0625）；到语言注意力时输入
SHA已不同。因此前一探针的Q投影差不能全归因于该投影本身。此定位不同于旧A100
上此前64边界一致、首语言注意力输出才变的结果，须保留硬件区别，不能互相替代。

随后取同一份实际B1注意力输入，精确重复为B2/B4；实际mask/position同样比较，
使用原首LlamaAttention参数，隔离该注意力本身。即使固定了上游输入，B2/B4第一个
已观测数值差仍出现在Q投影`aten.linear.default/0`，不是MoE gate、BatchNorm或
不同视频内容。B4/B1结果如下：

| DeepSeek固定输入首注意力 | 原生BF16 | FP32 GEMM后回原dtype |
| --- | ---: | ---: |
| Q投影不同元素/2231040 | 435414 | 265 |
| Q投影RMS绝对差 | .001978964 | .000023548 |
| 注意力输出不同元素/2231040 | 1535499 | 17787 |
| 注意力输出RMS绝对差 | .000330167 | .000012077 |
| 注意力输出最大绝对差 | .015625 | .00390625 |

这给出了“相同输入也会因有限精度运算路径随batch形状改变而产生差异”的直接干预
证据，同时说明只稳定注意力、忽略上游视觉误差并不足以稳定整模型。原A100上的
后续MoE专家集合变化证据仍有效，但不能称MoE是最早来源或唯一因果贡献。

所有实际注意力输入、原始首注意力参数/config/source、逐算子输出/大矩阵固定query
切片、8视频features/logits、原始kernel名、命令/状态保存在上述run；诊断PT不是
新训练权重。本轮45份证据共885092364字节已全部按SHA备份本地同名run，见
`backup_receipt_operator_all04.json`。另用CPU独立从原始PT重算Qwen各控制、DeepSeek
固定输入/Q投影/注意力输出差，逐项与JSON精确一致；hidden/mask/position跨B SHA
一致，结果在`cpu_final_verification.json`，三GPU子PID均已退出。无需占卡即可复核
这些结果，但原A100特定kernel及全558稳定化仍未完成，不在本轮自动追加GPU任务。

### 可供论文使用的新增英文说明（须与原A100结果分开）

To test whether batch-shape effects can arise independently of changes in model inputs,
we conducted additional operator-level interventions on an idle RTX 4090, using the
deployed quantized operands and frozen weights. Repeating the same input rows at different
matrix sizes in the first Qwen visual-merger projection selected different GEMM kernels
and produced different outputs. FP32 computation followed by conversion back to BF16
reduced, but did not eliminate, these differences. In DeepSeek, an additional fixed-input
experiment separated inherited visual-stream discrepancies from numerical variation
generated within the first language attention module. With identical attention inputs,
the first observed discrepancy occurred at the query projection; higher-precision GEMM
substantially reduced the attention-output discrepancy. However, applying this control
only to language attention did not stabilize complete-model predictions. These experiments
provide causal evidence for finite-precision, shape-dependent arithmetic effects, but do
not identify every original A100 kernel, establish complete-model numerical invariance,
or replace the independently measured A100 SRCC and matched inference costs.

## 最新结论：实际差异来源、路由和论文可用解释（15 October 2026）

本次同canonical诊断run的`focused_merger_routes03`，源main `e34b2a586`；先完成
同源CPU输入/位置收据，再在空闲250W A100上串行DeepSeek→Qwen，各原生与一个控制
模式、各B1/2/4、同8 TEST。40项CPU合同测试两端通过，两GPU子任务均complete。
自有进程均已退出、显卡释放；44份JSON/日志/PT（257439718字节）保存同名run，含
12组features/logits和12组focused边界/专家ID张量，后续复核只用CPU。

**与“诊断修复”严格区分：**此前的计时钩子、命名参数钩子、指纹比较顺序错误都属于
新诊断程序，已保留失败证据并修复；不是已发现正式评分有同样bug。新原生探针重现
旧8视频结果，最大MOS差Qwen7.6293945e-6、DeepSeek1.1444092e-5，支持诊断未改变
正式forward含义。正式六份表/测量文件SHA仍保持原值，不以探针或控制分数替换正式值。

| 模型 | 首个已观测分歧（B2/B4 vs B1） | 此前相同边界 | 该输出最大绝对差 |
| --- | --- | ---: | ---: |
| Qwen3-VL | 首deepstack merger的linear_fc1输出 | 22 | .125 |
| DeepSeek-VL2 | 首语言层self_attn输出 | 64 | .001953125 |

Qwen全名`model.visual.deepstack_merger_list.0.linear_fc1:output`。归一化输出和
linear_fc1输入相同，故已将范围从“整个合并模块”细分到该线性算子；不是MoE或
BatchNorm解释。保存的视频0输入B1/B4逐位相同，形状[392,4096]。钩子观测输入为FP32，
正式autocast执行linear时量化BF16，输出BF16；固定权重而改变batch行数时，输出出现
数值差异，符合有限精度矩阵计算路径随形状变化的表现。尚未证明具体cuBLAS kernel/
归约指令的唯一责任；禁低精度GEMM归约也未消除。CPU参考使用**先量化到部署BF16的
输入/权重，再FP64点积**，不称恢复量化位或新性能结果。该视频B1/B4相对于此参考的
最大绝对误差分别.149562/.062613；这不是“B4任务精度更高”的结论。

DeepSeek全名`language.model.layers.0.self_attn:output`。此前已观测视觉/投影/语言
输入均一致；17848320元素中4930项改变，relativeL2=3.2440191e-5。**最早已观测
差异早于MoE gate**，不能再只说“MoE分组导致差异”。实际11层专家集合随后都有变化。
下表按同一token在同一层的6专家集合是否改变计一次，忽略Top-k返回顺序；分母
153384=8视频×1743有效token×11层，不是视频数、专家调用数或GPU利用率。

| DeepSeek模式 | B2/B1集合变化事件 | B4/B1集合变化事件 | 最大MOS差B2/B1 | 最大MOS差B4/B1 |
| --- | ---: | ---: | ---: | ---: |
| 原生部署 | 22247/153384 (14.5041%) | 22251/153384 (14.5067%) | 1.472759 | 1.418308 |
| 仅gate关闭autocast | 21401/153384 (13.9526%) | 21504/153384 (14.0197%) | .697479 | .628998 |

Qwen原生MOS最大批差B2/B1=2.350662、B4/B1=3.395130；禁低精度归约对照为
2.585449和2.585453。两模型控制均未归零，而且各自B1也会变化；这些只是机制诊断，
不是重新选模、SRCC改善或稳定化修复。DeepSeekattention内部的Q/K/V、RoPE、注意力
matmul/softmax/o_proj未逐primitive区分，亦未通过固定专家干预证明路由的独立因果
贡献。因而能支持“低精度数值差异先出现、后续专家选择敏感”，不能声称路由是唯一
根因或已经排除一切实现问题。结合原重复推理和FP32探针，现证据不支持换权重、抽帧
变化或BatchNorm导致这些8视频的批差；不推广为所有558视频内部路径已完全审计。

最新派生汇总为`focused_cpu_analysis_corrected.json`。首份`focused_cpu_analysis.json`
的CPU fc1参考使用了量化前FP32输入，原位保留但不引用其参考误差；更正版显式先复现
部署BF16操作数量化。原始GPU边界数据及专家ID并未改动。

### 可供论文使用的英文说明

Batch-dependent predictions were investigated using the same frozen checkpoints and eight
fixed test videos at batch sizes of 1, 2 and 4. In Qwen3-VL, the first instrumented
discrepancy occurred at the output of the first linear projection in the first deep-stack
visual merger; its input and the preceding 22 recorded boundaries were bitwise identical.
This directly localizes the first observed discrepancy to a low-precision linear computation,
rather than a change in video identity or model parameters. In DeepSeek-VL2, the first
observed discrepancy arose at the first language self-attention output, before MoE routing,
with a maximum absolute difference of 0.001953125. The subsequent top-six expert sets differed
for 22,251 of 153,384 token–layer events (14.51%) between B4 and B1. Disabling autocast only
within the gates reduced, but did not eliminate, batch-dependent score differences. These
observations, together with repeated-inference and higher-precision probes, support
finite-precision batch-shape effects and downstream routing sensitivity. They do not isolate
every attention primitive, establish routing as the sole cause, or constitute a full-test-set
stabilization experiment. Consequently, the reported SRCC and matched costs remain those
measured independently for each deployed batch configuration, rather than a shared B1 score.

## 新增逐层证据与退卡交接（15 October 2026）

用户要求短时采集并保存中间数据，GPU前向结束立即释放、后续CPU核对。
同canonical诊断run的`recovery_hooks02`（执行源码`881393f78`）中，Qwen已经完成
native和仅关闭低精度GEMM归约两模式、各B1/2/4；DeepSeek未完成。自有GPU进程已退出，
没有占位或自动补跑。24份CPU/GPU证据SHA备份本地，包括6份8视频末端features/logits、
全部已观测边界的差统计、命令/日志与状态；完整逐层激活只在诊断时CPU内存中使用，
未存成可离线重放的全模型激活包，不声称后续无需任何GPU即可定位每个primitive。

| Qwen对照 | MOS最大B2−B1绝对差 | MOS最大B4−B1绝对差 | 首个观测分歧（B2/B4） |
| --- | ---: | ---: | --- |
| 原生部署BF16 | 2.350662 | 3.395130 | deepstack_merger_list.0输出 |
| 仅禁低精度GEMM归约 | 2.585449 | 2.585453 | 同一模块输出 |

原生模式每组8视频MOS与旧探针最大偏差<.001守卫通过；不是更换权重后得到的新结果。
首个观测分歧全名为`model.visual.deepstack_merger_list.0:output`，之前19个记录边界
跨batch逐位一致。B4该合并模块输入max_abs=0；输出max_abs=.0625、relativeL2=
.0029048164295297028（3299780/6422528元素变化）。原生源码含LayerNorm、Linear、GELU、
Linear。此证据把首个已观测差异界定在视觉合并模块内、语言模型之前；不能据此指定
哪个内部primitive最早产生误差。只禁归约未消除差异，不能宣称该开关足以修复，亦不能
用DeepSeek的MoE机制解释非MoE的Qwen。

DeepSeek在前向前被诊断指纹守卫拦下，无实际专家ID或有效边界差输出。纯CPU同视频
审计`cpu_transfer_fingerprint_audit.json`确认：预处理指纹匹配旧探针；原生
DictOutput.to(cpu,BF16)会原地修改同一容器，images FP32→BF16，仅pixels SHA改变。
诊断错误地在transfer后比较transfer前指纹，故误报。已修正比较顺序、补回归测试，
未为此再占A100；此解释是CPU转换复现和代码证据，不冒称保存到了失败时真实GPU输入。
DeepSeek的路由及首分歧层仍待短窗口验证，先不重跑已完成的Qwen或修改正式图表。

For the same eight videos, Qwen3-VL's first observed batch-dependent discrepancy arose
at the output of the first deep-stack visual merger, while the preceding 19 recorded
boundaries were bitwise identical. The merger input was unchanged, whereas its B4 output
differed by up to 0.0625 from the B1 reference. This localizes the first instrumented
discrepancy to the visual merger, before the language model, but does not isolate the
earliest internal primitive. Disabling reduced-precision GEMM reductions did not eliminate
the batch dependence. DeepSeek's new layer/route experiment stopped before inference due
to an instrumentation fingerprint-order error; its routing mechanism therefore remains
unverified by this experiment. Neither diagnostic replaces the frozen 558-video metrics
or their matched inference-cost measurements.

## 2026-10-15：BatchNorm、模式切换和跨样本操作核查

**新增实际CPU输入/位置检查（同一8视频/B1/2/4）：**新run
`lgvq_batch_layers_a100_20261015`的两CPU子任务均complete，准备的图像/有效文本及抽帧
指纹逐项匹配旧诊断。Qwen每条446有效token、0 padding、最后index445；调用原生
`get_rope_index`得到的有效position SHA跨B完全一致，rope_delta均−364。
DeepSeek每条1743有效token、0 padding、最后index1742；按实际DeepseekV2Model的fresh
无cache源码推导的arange位置跨B一致，这项不是GPU位置tensor观测。
这组探针的评分差不能解释为padding数量变化或选错最后token；未推广至全部558视频。
同run CPU证据9文件已本地SHA备份，源码`5e13ea536`、36项CPU检查两端通过。
**GPU诊断尝试失败，尚无逐层结论（15 October 2026）：**第11次核查时A100空闲，
连续两次确认无compute PID、250W、源码SHA匹配CPU后派发一次。Qwen第一次前向
因诊断集成错误`KeyError('vision_calls')`退出，DeepSeek未启动；没有生成有效边界差
或专家路由结果。复用正式loader时遗留`BoundaryTimer`钩子，直接诊断forward未reset
计数状态；这是诊断程序的错误，不是解释batch精度差异的证据。正式推理路径每次
forward调用timer.reset，本次失败不回写正式测量。自有进程退出；14份状态/日志/
命令按SHA备份同名run，收据`backup_receipt_gpu_failed_check11.json`。保留失败原位，
未重新派发；恢复需先修正诊断钩子生命周期、补测试并明确新来源与批准的运行方式。
后续仍限固定A100空闲、每模型600秒、同8视频，不改正式数据。首个观测边界也不
等于最早primitive证明。六份正式表格/测量文件的SHA均保持原值。

用户提出BatchNorm假设后，在**原服务器、原依赖环境、CUDA_VISIBLE_DEVICES=-1**下核查。
Qwen/DeepSeek以原config构建meta结构，不加载骨干权重、不做前向；CLIP/YOLO原骨干只加载
到CPU，不做前向。归一化/Dropout模块清单、实际原生类源码与SHA保存在同一canonical run。
结构审计按正式loader的`.eval().requires_grad_(False)`设置执行；这不是历史运行逐层旗标的
补造收据。正式loader本身已逐处核查eval调用，无新增train调用或模式修复。

| 骨干 | BatchNorm层数 | 非BatchNorm归一化 | 审计eval后的training模块 | 特别核查 |
| --- | ---: | --- | ---: | --- |
| Qwen3-VL-2B | 0 | 52 LayerNorm、113 TextRMSNorm | 0 | LayerNorm维度1024/4096；RMS按末维 |
| DeepSeek-VL2-Tiny | 0 | 56 LayerNorm、25 RMSNorm | 0 | 86个Dropout模块，配置p均为0且eval |
| CLIP ViT-B/32 | 0 | 51 LayerNorm | 0 | 含完整视觉/文本结构，维度512/768 |
| YOLO11s | 81 | BatchNorm2d | 0 | 全部track_running_stats=True，均有running_mean/var |

**结论：BatchNorm不是当前Qwen/DeepSeek较大评分差的解释，它们根本没有该模块。**
YOLO虽然有BatchNorm，原CPU模型的81层均使用已保存的运行统计条件，而不是当前batch统计。
不能只看“调用了eval”就忽略`track_running_stats=False`这个例外；本轮已实际检查该条件和
统计buffer存在性。LayerNorm/RMSNorm的统计轴是token/位置内部的特征末维，不是batch维，
不因为同批增加另一条视频而在数学上重新计算该视频的batch均值/方差。
正式五行头为独立线性读出，softmax沿五个质量级别的末维，不沿视频batch维。

**DeepSeek确实有batch相关的执行分组，但不是BatchNorm式跨样本统计。**
实际12个语言层中11个为MoE，每token在64专家中选Top-6，greedy/softmax。
原`MoEGate.forward`的softmax和Top-k沿每token的专家维；均衡辅助损失只在
`self.training and alpha>0`执行，eval不计算/注入该项。`moe_infer`将整批token按专家排序，
`tokens_per_expert=cnts.sum(dim=0)`确实统计了整批数量，但只用于决定每个专家的输入切片，
不是归一化分母，审查的单卡分支没有capacity-factor限额或token-dropping。
末端按排序逆映射回原token，再对该token自己的专家输出加权，不对不同视频取平均。
这保证了代码意图上的逐token独立性；改变batch会改变专家GEMM输入行数/分组形状，
有限精度结果仍可能变化，不能据源码就证明实际路由完全不变或首个误差层。

**新发现的待验证点：路由里的`.float()`不等于已关闭autocast。**
DeepSeek原gate将特征和权重显式转FP32，再调用`F.linear`；但正式外层处在BF16 autocast。
对符合autocast规则的linear，仅将输入转float32并不强制该操作全程FP32。
本轮小型CPU dtype示例确认：float32输入+float32权重在CPU BF16 autocast下，linear输出
仍是BF16。该示例不用训练权重，只证明dtype规则，不冒称测到了正式CUDA gate dtype或
专家切换。PyTorch 2.6的[AMP规则及关闭autocast子区域说明](https://docs.pytorch.org/docs/2.6/amp.html)
支持这一执行语义。Top-k是离散选择，若临界专家分数次序变化可切换专家，这是具体的候选
机制；它不是已被该批视频的gate日志证明的事实，且Qwen不是MoE，不能用此解释全部模型。

**其他检查：**DeepSeek图像tile先批量提取，但再按各样本tile数/`images_seq_mask[idx]`
写回`input_embeds[idx]`，不是跨视频融合。Qwen视觉attention保留`cu_seqlens`分段；语言
attention保留样本batch轴和attention_mask。正式读出按每行mask找最后有效token，未直接
把补齐位置当评分位置，且两模型`use_cache=False`。已有8视频指纹只证明有效输入/抽帧
一致；未用源码检查替代全部内部mask/position_ids数值运行验证。
DeepSeek实际语言attention类为**LlamaAttention**，不是包中另一个未用的DeepseekV2Attention；
本轮补核绑定到实例里的实际类。首份较粗的DeepSeek源码清单原位保留，最新引用下方
`deepseek_actual_attention`审计，不将未用类当正式路径。

**给老师的当前解释：**

BatchNorm does not account for the larger batch-dependent differences in Qwen3-VL or
DeepSeek-VL2: neither instantiated architecture contains BatchNorm. Their LayerNorm/RMSNorm
operations normalize feature dimensions rather than pooling statistics across videos.
The deployed loaders explicitly use evaluation mode. YOLO11s contains BatchNorm, but all
81 audited layers retain running statistics and are in evaluation mode. DeepSeek's
evaluation-time MoE dispatch groups tokens across the batch for expert computation without
batch-dependent normalization or capacity-based token dropping in the inspected single-GPU
path. This grouping changes computation shapes; together with autocast and discrete top-k
routing, it provides a concrete numerical mechanism to investigate. It has not yet been
verified as the first divergent operation on the actual videos. The current evidence rules
out the proposed BatchNorm explanation, not every possible implementation-level defect.

32项CPU检查两端通过；四份审计JSON已SHA备份至本地。最新证据在同run的
`diagnostic_batch_semantics_20261015_qwen`、`diagnostic_batch_semantics_20261015_deepseek_actual_attention`、
`diagnostic_batch_semantics_20261015_clip`、`diagnostic_batch_semantics_20261015_yolo`内`audit.json`。
本轮无GPU测量，无新视频推理，不更改正式表格、分数、时延、能量、利用率或原PT。

## 关键更正：利用率标注与尚未修复的数值差异

前次交付把GPU利用率放在表格和单独曲线中，却遗漏了主成本图逐batch标注，且最终汇报未展开诊断结论。现在主图每个电子点直接标`B`和`U`，`U`明确为与该图成本同负载的平均GPU kernel-busy利用率，不是峰值。`srcc_cost_comparison_reference`为主交错负载；`srcc_cost_comparison_resident`为另存的15秒resident负载及其自身成本，不能互相替换。LightGen-II未测此轮GPU利用率，标为n.a.。这次只更正图和解释，不改变任何实测分数、时延、能量或利用率。

| B=8模型 | 主交错平均U (%) | Resident平均U (%) | Resident峰值U (%) |
| --- | ---: | ---: | ---: |
| Qwen3-VL-2B | 13.555556 | 89.220000 | 95 |
| CLIP ViT-B/32 | 3.272727 | 71.295302 | 89 |
| YOLO11s | 3.935484 | 97.097643 | 99 |
| DeepSeek-VL2-Tiny | 62.196891 | 95.258278 | 100 |

因此并非所有B8利用率都低于90%。Qwen当前89.22%是整段约15秒的平均，95%是该段峰值。已定位的一份历史Qwen B8报告记录平均70.341463%、峰95%，不能把历史95%峰值与本轮89.22%平均解释成平均利用率降低。历史出处：`raw_remote/qwen3_vl_2b_independent/batch_8/formal/batch_08/report.json`；测量负载/范围不同，仍不将两轮拼接。若另有曾展示的B8平均>90%记录，须绑定其原报告和负载才可判断变化原因。

Resident只表示输入驻GPU、不重复解码与H2D；每次仍调用原`forward`，CUDA同步、执行hook/计时，最后把MOS拷回CPU列表，之后再发下一次调用。它不是CUDA Graph或无CPU间隙的纯GPU kernel饱和实验，不能保证100%利用率。已发现这些间隙，但尚未做逐kernel剖析，不能将89.22%的每一项损失归因到某个算子。主交错统计还受到前序CPU准备和驱动平均窗口影响；按active标签筛选不能把驱动已平均的历史空闲“消掉”。NVIDIA的[利用率定义](https://docs.nvidia.com/deploy/nvidia-smi/index.html)是采样窗口内有kernel运行的时间占比，不是SM occupancy、显存占用率或峰值算力使用率。

精度方面，**复测和诊断完成，不等于正式数值稳定化修复完成**。以下8视频输入SHA/重复前向/FP32探针证据支持低精度批计算是主要贡献，但未逐层定位全部算子，也未做558视频FP32正式评估。正式曲线仍按原BF16/FP16合同运行，因此保留真实batch差异。统一FP32或调整积累/attention执行方式需要新正式对照，并重新配套测SRCC、时延、能量和利用率；不能拿8视频探针分数替换558分数，更不能将B1分数复制到其他B来“修复”图。[PyTorch数值文档](https://docs.pytorch.org/docs/2.6/notes/numerical_accuracy.html)也明确说明，批计算与逐样本计算不保证逐位一致。

## 2026-10-14进一步排查：差异已存在于质量头之前

本轮新增的是**保存证据的CPU误差分解**，没有重跑骨干、重训、选模、修改正式精度或成本。
使用同一8视频探针的原特征/logits、精确原head PT，将线性质量头在CPU FP64重算。
设特征为F、五行权重为W，则batch引起的logits变化应为`(F_B−F_1)W^T`。
比较此预测变化与保存的GPU logits变化，可以区分上游特征变化和质量头数值残差。
CPU FP64仅用于误差分析，不是新的正式评估协议或替换分数。

| 同一8视频，B4相对B1 | Qwen3-VL-2B | DeepSeek-VL2-Tiny |
| --- | ---: | ---: |
| 最终读出特征最大绝对差 | 2.750000 | 0.062500 |
| 最终特征相对L2差 | 1.884409% | 1.270936% |
| 保存的五路logits最大差 | 0.897963 | 0.162239 |
| 由特征变化预测的logits最大差 | 0.897973 | 0.162234 |
| 未被特征变化解释的logits变化最大残差 | 0.000040097 | 0.000021795 |
| 保存logits经CPU FP64 softmax后的最大MOS差 | 3.395131 | 1.418310 |
| 改变特征、固定head经CPU FP64重算的最大MOS差 | 3.395238 | 1.418287 |
| CPU重投影与保存logits的MOS最大偏差（所有8视频） | 0.000448 | 0.000235 |
| 固定特征仅改变CPU质量头batch分组的logits最大差 | 0 | 0 |

**能支持的定位：**在这8个样本中，batch差异主要已经存在于末端质量头的输入特征；
同一头权重即可近乎重现保存的评分差。质量头投影/softmax自身的读出舍入不是该数分MOS
差异的主要来源，不能只把末端head改成更高精度就称解决。结合已有输入SHA、原精度重复
前向、math-only及全骨干FP32探针，可支持低精度上游计算路径是主要贡献。
DeepSeek的读出特征仍带原FP16量化，当前分析尚未分离其量化与此前BF16骨干各自贡献。
**不能支持的定位：**尚不能指出误差最先出现于某个视觉block、某个语言block、attention、
GEMM或MoE路由；FP32+math探针联合改变执行路径，不能直接指定某个算子为唯一原因。

另独立复核保存的完整558视频预测：各batch样本顺序、标签和抽帧位置一致，
SRCC由各自预测重新计算。下表是相对B1的严格视频对排序反转；共有155403对视频，
预测相等的状态变化另计，本轮这些设置为0。不是训练随机性或不同batch用了不同标签。

| 模型 | B | 最大MOS差（558视频） | 严格排序反转对数 | 最大预测名次变化 | SRCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-2B | 1 | 0 | 0 | 0 | 0.7663428036522781 |
| Qwen3-VL-2B | 2 | 3.311459 | 3315 | 52 | 0.7664148425389119 |
| Qwen3-VL-2B | 4 | 3.971672 | 3454 | 61 | 0.7694958345192725 |
| Qwen3-VL-2B | 8 | 3.971672 | 3461 | 61 | 0.7694089803486666 |
| DeepSeek-VL2-Tiny | 1 | 0 | 0 | 0 | 0.5684669009566514 |
| DeepSeek-VL2-Tiny | 2 | 2.147045 | 3739 | 75 | 0.5703507558300323 |
| DeepSeek-VL2-Tiny | 4 | 2.559540 | 3638 | 73 | 0.5680463057503653 |
| DeepSeek-VL2-Tiny | 8 | 1.639896 | 3524 | 61 | 0.5698244506371757 |

Qwen B8最后一位与原汇总相差约1e−16，来自当前SciPy数值实现，保留原正式表值不替换。
排序扰动会改变与真实MOS排序的相关性，既可能增加也可能减少SRCC；同一权重并不代表
有限精度下逐位batch不变，更不存在“batch越大性能越好”的理论结论。

**给论文作者/老师的解释：**

The batch-dependent SRCC differences are reproducible numerical effects rather than changes
in model weights or test-set identity. A CPU double-precision decomposition of the preserved
eight-video probes shows that the differences are already present in the features supplied
to the five-row quality head. Reprojecting those features with the identical trained head
nearly reproduces the observed MOS differences, while the unexplained cross-batch logit
residual is below 4.1×10⁻⁵ for Qwen and 2.2×10⁻⁵ for DeepSeek in the B4-versus-B1 probe.
Across the full 558-video evaluation, 3,454 and 3,638 video pairs, respectively, reverse their
predicted ordering at B4 relative to B1, explaining why SRCC is not exactly invariant.
The existing FP32 probes substantially reduce these differences, supporting upstream
low-precision computation as a major contributor. The first divergent internal operation
has not yet been isolated; these diagnostic recomputations do not replace the formal
BF16 results or their matched timing, energy and utilization measurements.

分析入口`analyze_saved_batch_numerics`；本地与服务器30项CPU合同测试通过。
原位证据：[Qwen分析](../../../tasks/t06_video_quality_assessment/runs/simulation/lgvq_temporal_telemetry_a100_20261014/full_remeasurement02/diagnostic_saved_analysis_20261014_qwen/analysis.json)、
[DeepSeek分析](../../../tasks/t06_video_quality_assessment/runs/simulation/lgvq_temporal_telemetry_a100_20261014/full_remeasurement02/diagnostic_saved_analysis_20261014_deepseek/analysis.json)。
每份记录保存原PT、原诊断JSON/12份特征PT/完整预测CSV及分析源码SHA；新增分析已逐SHA备份。
前期逐层GPU诊断因A100外部占用等待；第11次空闲后的第一次尝试在Qwen首前向因
上述诊断计时钩子错误退出，DeepSeek未启动，未得到有效逐层证据。后续须先修复和
明确恢复方式，再在同一8视频/B1/2/4合同下比较视觉及语言边界；不能宣称已定位首个分歧层。

## 两图含义与实验方法

图a给出能量与时间质量性能的关系，图b给出推理时间与相同性能指标的关系。横轴分别为 Energy per 16 videos (J) 和 Inference time per 16 videos (ms)，均使用对数坐标，范围覆盖全部测量点；纵轴 Temporal quality SRCC 为无量纲量，显示0–1。B表示每次原生GPU前向调用中的视频数，N表示同幅光场中的视频数。电子曲线连接同一模型按B递增的独立测量，并非不同batch共用一个性能值。LightGen-II为N=16物理SRCC参考及既定部件成本估算，不是本轮新测光路。

LGVQ固定划分包含2250个TRAIN和558个TEST视频，共2808个不同视频。每个视频按10%、37%、63%、90%的相对位置取原4帧。Qwen与GLM原解码尺寸为448，Small/Tiny为384，CLIP为224，YOLO为640像素；保持各模型原中心裁剪、原生processor及图像/视频token路径，不减帧或额外改分辨率。所有原PT冻结，eval/inference模式，每模型、每B使用全新进程。Qwen、GLM、Small和Tiny维持原BF16骨干，CLIP与YOLO维持FP16骨干，质量读出为FP32，原FP16特征量化路径不变。本轮不训练、不选模；此前根据该TEST划分选出的head仍属于开发指标，冻结复测不使其自动成为独立泛化测试。

30种设置为Qwen B=1/2/3/4/6/8/10、GLM B=1/2、Small B=1/2/3/4、CLIP B=1/2/3/4/6/8/9/10、YOLO B=1/2/3、Tiny B=1/2/3/4/6/8。每种设置对相同顺序的完整558个TEST视频独立推理，按该B自己的预测计算SRCC与PLCC；逐视频ID、目标MOS、抽帧位置及PT/划分SHA均核对。近水平曲线不是复制B=1分数，也不意味着逐视频输出逐位相同。

每批先在CPU上准备该批输入，再H2D，再执行一次原生批前向并输出MOS，之后处理下一批。同步墙钟从首原生backbone block输入开始，到GPU MOS完成并同步为止；YOLO首block为卷积，其他为视觉Transformer。计时包含此边界内的视觉/语言骨干、相关GPU处理和质量读出，不是只统计单个Transformer层。模型加载、CPU解码/裁剪/processor、H2D、随后CPU复制预测及审计开销不计最终成本，但原始逐批表保留审计字段。无显式预热，首调用保留；预钩子不额外同步，因此这是既定软件边界的同步墙钟，不是校准的纯kernel时间积分。

成本固定使用前200个不同TEST视频，不是只测16个视频。分别在200和558处拆批，保留全部不足满批的尾样本。设这200个视频各实际调用的边界墙钟为tᵢ（ms），则
`T16 = Σtᵢ × 16/200`，
`E16 = (338.2 W主机功率代理 + 同轮前200视频inference-active平均A100板卡功率) × T16/1000`。
16是固定工作量的吞吐归一化；B并不意味着GPU同时处理16个视频。只有前200恰好整除B时，才可等价写成mean batch ms×16/B；B=3/6/9的尾批必须按实际总时间统计。表中的每视频摊销成本为T16/16与E16/16，不应将光学16槽并行的摊销时间称为单视频独立调用延迟。

| B | Full558 calls | Final TEST call videos | First200 cost calls | Final cost call videos |
| --- | --- | --- | --- | --- |
| 1 | 558 | 1 | 200 | 1 |
| 2 | 279 | 2 | 100 | 2 |
| 3 | 187 | 1 | 67 | 2 |
| 4 | 140 | 2 | 50 | 4 |
| 6 | 94 | 4 | 34 | 2 |
| 8 | 70 | 6 | 25 | 8 |
| 9 | 63 | 7 | 23 | 2 |
| 10 | 56 | 8 | 20 | 10 |

本轮均使用同一NVIDIA A100-PCIE-40GB、固定250W设置和同一manifest。板卡功率、驱动利用率和显存以名义50ms轮询并记录阶段。成本功率/active均值仅使用同轮前200视频的推理标签窗口，不把后358视频的遥测混入，也不使用独立驻留负载的功率。利用率是驱动平均窗口内kernel忙碌时间比例，不是SM occupancy或峰值FLOPS比例；50ms轮询并不使驱动窗口变成50ms瞬时值，前序CPU等待可影响短窗口的读数。均值板卡功率加338.2W主机代理再乘边界时间是沿用的能量估算口径，不冒称短调用精密积分或实测整机墙插能量。250W是功率设置，原始峰值不因此被裁去。

图中每点的百分数为独立resident利用率测量：预先准备前B个不同TEST视频的原4帧输入，一次H2D后驻留GPU；原生前向同流连续提交8次后同步，再继续同样循环。每次前向仍为B个视频，不是8B视频的一个大batch。除用户指定的Qwen B1历史单次63.13%展示锚点外，其余各B三次完全相同重复，每次至少5秒预热、至少30秒完整测量，合并全部测量窗口样本计算均值，不选三次最高值，无CUDA Graph。它用于描述减少CPU供数间隙后的原生忙碌率，独立于本轮逐批准备输入的成本测量；其利用率不代入E16，也不冒称与Qwen历史成本同轮。原Qwen B1新三次均值71.18%与全部重复原始证据仍保留，不被历史展示选择覆写。完整方法见`UTILIZATION_METHOD_20261015.md`。

## 图a：能量与时间质量性能

图a比较完成等效16视频工作量所需的能量估算及相应的全558视频SRCC。LightGen-II物理SRCC为0.7977，E16为1.1777J。以已定稿Qwen B=1为对照，其E16为462.1134J，等效任务能效比为392.39倍，能量降低99.7452%；这一定稿点使用历史成本身份，而非新测功率推算的值。Qwen B=10的本轮E16为222.9824J，对应189.34倍能效比。这里能效指同工作量能量的反比，不等于TOPS/W。随B增大，调用次数减少且开销被摊销，但板卡功率、尾批与启动开销同时变化，能量不必单调下降；例如CLIP B=9的14.025980J高于B=8的13.160310J。图中不人为平滑这些实测差异，也不将独立高利用率视为该成本运行的遥测。

## 图b：时间与时间质量性能

图b比较相同16视频归一化工作量的推理边界时间及全558视频SRCC。LightGen-II沿既定部件估算为10.2456ms。定稿Qwen B=1为1188.3861ms，时间比为115.99倍；Qwen B=10本轮为479.1690ms，时间比为46.77倍。增大B通常增加单调用计算量，却减少固定工作量的调用数，因此摊销时间可下降，之后趋于饱和或非单调。该图不是只比较一次GPU调用，更不是把B=8的八视频调用时间与光路16视频时间直接相除。不同batch的SRCC来自各自的真实预测；本轮保持原精度，不为获得相同SRCC而改变数值路径。

## 全部绘图数据与配套成本遥测

成本测量日期为2026-10-15、2026-10-16（北京时间）；以下31行包含30种电子设置和1个不变光学参考。独立驻留利用率来自2026-10-15统一测量。保留多位小数用于复核，不表示相同位数的测量有效精度。未提前舍入值及逐点来源、PT SHA、PLCC、成本调用数/尾批、active样本数、峰值功率和独立利用率日期均保存在`timing_energy_recalculated.csv`及`recalculation.json`；直接绘图输入为`plot_data.csv`，逐B精度复核为`batch_metric_audit.csv`。

| Model | B / N | SRCC (558 TEST) | Time/16 (ms) | Energy/16 (J) | New-run board mean (W) | New-run active util (%) | Independent resident util (%) | Peak VRAM (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGen-II | N=16 | 0.7977138739 | 10.245570 | 1.177690 | n.a. | n.a. | n.a. | n.a. |
| Qwen3-VL-2B † | 1 | 0.7663428037 | 1188.386146 | 462.113401 | 49.057 | 13.897 | 63.133 | 4705 |
| Qwen3-VL-2B | 2 | 0.7664148425 | 775.464571 | 299.266888 | 47.719 | 12.519 | 87.273 | 4883 |
| Qwen3-VL-2B | 3 | 0.7688994244 | 594.155985 | 241.654724 | 68.519 | 5.927 | 89.268 | 4995 |
| Qwen3-VL-2B | 4 | 0.7694958345 | 529.877741 | 220.545789 | 78.020 | 3.886 | 91.068 | 5107 |
| Qwen3-VL-2B | 6 | 0.7690275127 | 530.216597 | 228.256550 | 92.297 | 8.347 | 91.384 | 5363 |
| Qwen3-VL-2B | 8 | 0.7694089803 | 510.585322 | 231.044964 | 114.310 | 14.261 | 91.634 | 5603 |
| Qwen3-VL-2B | 10 | 0.7689643492 | 479.169046 | 222.982372 | 127.152 | 21.757 | 90.629 | 5847 |
| GLM-4.6V-Flash | 1 | 0.8092619561 | 1823.807501 | 804.723114 | 103.032 | 24.582 | 97.684 | 20495 |
| GLM-4.6V-Flash | 2 | 0.8103099734 | 1637.954050 | 793.101065 | 146.002 | 14.726 | 98.069 | 20743 |
| DeepSeek-VL2-Small | 1 | 0.7000797712 | 6394.885049 | 3249.991677 | 170.017 | 40.074 | 78.485 | 32093 |
| DeepSeek-VL2-Small | 2 | 0.7011629394 | 4962.690410 | 2713.023316 | 208.484 | 56.002 | 91.331 | 32719 |
| DeepSeek-VL2-Small | 3 | 0.7006516273 | 4438.846306 | 2469.259451 | 218.084 | 65.897 | 96.723 | 33393 |
| DeepSeek-VL2-Small | 4 | 0.6975365498 | 4275.846012 | 2391.574259 | 221.122 | 72.734 | 98.495 | 34053 |
| CLIP ViT-B/32 | 1 | 0.7252558080 | 182.672089 | 69.161005 | 40.407 | 2.160 | 38.934 | 903 |
| CLIP ViT-B/32 | 2 | 0.7252452404 | 98.634829 | 37.260293 | 39.560 | 1.844 | 42.402 | 923 |
| CLIP ViT-B/32 | 3 | 0.7252388170 | 82.313271 | 31.075554 | 39.328 | 1.727 | 48.610 | 925 |
| CLIP ViT-B/32 | 4 | 0.7252388170 | 58.950832 | 22.270524 | 39.581 | 3.600 | 53.061 | 931 |
| CLIP ViT-B/32 | 6 | 0.7252201339 | 45.331876 | 17.114467 | 39.337 | 2.000 | 67.841 | 953 |
| CLIP ViT-B/32 | 8 | 0.7252388170 | 34.831300 | 13.160310 | 39.630 | 1.250 | 75.929 | 975 |
| CLIP ViT-B/32 | 9 | 0.7252247615 | 37.140156 | 14.025980 | 39.450 | 4.182 | 88.806 | 995 |
| CLIP ViT-B/32 | 10 | 0.7252388170 | 34.106913 | 12.895483 | 39.890 | 4.222 | 94.894 | 999 |
| YOLO11s | 1 | 0.7015970082 | 374.377717 | 141.922212 | 40.888 | 4.000 | 44.456 | 841 |
| YOLO11s | 2 | 0.7015940037 | 209.886779 | 79.243728 | 39.355 | 3.071 | 73.517 | 1037 |
| YOLO11s | 3 | 0.7015940037 | 167.186430 | 63.184433 | 39.728 | 2.725 | 98.430 | 1259 |
| DeepSeek-VL2-Tiny | 1 | 0.5684669010 | 2962.976354 | 1358.412224 | 120.262 | 21.773 | 74.497 | 7771 |
| DeepSeek-VL2-Tiny | 2 | 0.5703507558 | 2106.624516 | 1038.187488 | 154.620 | 19.910 | 84.115 | 8479 |
| DeepSeek-VL2-Tiny | 3 | 0.5718993847 | 1802.669920 | 910.704630 | 166.998 | 24.852 | 88.851 | 9035 |
| DeepSeek-VL2-Tiny | 4 | 0.5680463058 | 1678.749006 | 868.184257 | 178.961 | 35.333 | 92.303 | 9609 |
| DeepSeek-VL2-Tiny | 6 | 0.5705645932 | 1618.090537 | 840.004509 | 180.933 | 49.801 | 96.084 | 10883 |
| DeepSeek-VL2-Tiny | 8 | 0.5698244506 | 1537.324385 | 817.289328 | 193.431 | 60.676 | 97.782 | 12065 |

† Qwen B=1的Time/16与Energy/16为历史定稿成本；本行New-run功率、active util、显存及当前精度属于本轮审计，不构成“历史成本与新功率配套”的声明。本轮实际审计T16=1230.453189611ms、E16=476.501019216J，另存工作表审计列，未覆盖定稿值。当前B=1全558 SRCC=0.7663428036522781，与历史定稿一致。独立驻留利用率按2026-10-16用户最新指定展示历史63.13311148%；新三次均值71.17536071%仍作为审计保留，不声称它们是同一轮或同一重复次数。

## 历史连续负载存档：2026-10-14（不用于当前两图）

本节仅保留旧15秒连续负载的原始对照；不是当前统一三次驻留利用率，也不替代当前30种设置的成本。下列19行均属历史测量，未混入上方完整新表。

另在同一进程、同一PT/B下，用前B个不同TEST视频已驻GPU的输入重复推理约15秒。此处没有每次CPU准备，成本和util来自连续负载自己的记录，严禁将本表高util配给上一表交错成本。时间折算mean boundary ms×16/B，B=3/9/10为吞吐等效值，不宣称整数次调用刚好处理16条。连续负载未独立计算新的全TEST性能，性能仍来自同PT/B的558视频独立前向。CLIP B9平均80.7617%触发补B10；B10平均89.1913%未达98%，不扫描更大B。YOLO B3平均85.6284%，B4为96.5436%，B8为97.0976%；不能把峰值98%当作平均98%。

| Model | B | Time/16 (ms) | Energy/16 (J) | Mean util (%) | Peak util (%) | Mean board power (W) | Peak VRAM (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B | 1 | 778.934426 | 407.094159 | 63.117 | 71 | 184.430 | 4705 |
| Qwen3-VL-2B | 2 | 493.315589 | 284.183387 | 83.431 | 88 | 237.868 | 4883 |
| Qwen3-VL-2B | 4 | 455.337138 | 263.522301 | 85.367 | 90 | 240.541 | 5107 |
| Qwen3-VL-2B | 8 | 422.711758 | 243.963920 | 89.220 | 95 | 238.940 | 5603 |
| CLIP ViT-B/32 | 1 | 89.181167 | 36.400936 | 39.010 | 60 | 69.968 | 903 |
| CLIP ViT-B/32 | 2 | 45.296298 | 20.212696 | 40.805 | 63 | 108.033 | 923 |
| CLIP ViT-B/32 | 4 | 22.608331 | 10.856470 | 50.671 | 64 | 141.998 | 935 |
| CLIP ViT-B/32 | 8 | 11.922274 | 6.511954 | 71.295 | 89 | 208.001 | 975 |
| CLIP ViT-B/32 | 9 | 10.267646 | 5.880074 | 80.762 | 98 | 234.480 | 999 |
| CLIP ViT-B/32 | 10 | 8.932255 | 5.185225 | 89.191 | 95 | 242.306 | 999 |
| DeepSeek-VL2-Tiny | 1 | 2471.141476 | 1302.795259 | 71.463 | 87 | 189.004 | 7771 |
| DeepSeek-VL2-Tiny | 2 | 2064.253894 | 1120.997979 | 74.415 | 83 | 204.852 | 8481 |
| DeepSeek-VL2-Tiny | 4 | 1579.469002 | 909.900767 | 89.662 | 98 | 237.880 | 9651 |
| DeepSeek-VL2-Tiny | 8 | 1438.482441 | 833.185891 | 95.258 | 100 | 241.012 | 12065 |
| YOLO11s | 1 | 206.996660 | 90.978331 | 39.503 | 69 | 101.316 | 841 |
| YOLO11s | 2 | 88.403786 | 45.909937 | 67.772 | 87 | 181.121 | 1037 |
| YOLO11s | 3 | 62.638440 | 35.967365 | 85.628 | 98 | 236.006 | 1259 |
| YOLO11s | 4 | 53.780790 | 31.315951 | 96.544 | 99 | 244.089 | 1469 |
| YOLO11s | 8 | 49.939259 | 29.122482 | 97.098 | 99 | 244.958 | 2329 |

## 历史精度变化的诊断证据（本轮未重测）

同权重精确数学推理不应因batch而改变输出，也无性能随B单调提升或下降的预期。第一步核所有558个样本ID、MOS及每模型抽帧一致；同模型PT与manifest SHA一致。Qwen和DeepSeek另对前8个固定TEST做B1/2/4输入tensor SHA及重复前向。逐视频有效文本、图像、抽帧相同，DeepSeek还核图像crop及图像token mask；原精度重复结果相同，原精度诊断与正式前向最大差0。

| Model | Probe | B2–B1 max absolute MOS difference | B4–B1 max absolute MOS difference |
| --- | --- | --- | --- |
| qwen | native | 2.350662231 | 3.395130157 |
| qwen | native_repeat | 2.350662231 | 3.395130157 |
| qwen | native_math_sdpa | 1.700469971 | 1.902404785 |
| qwen | cast_deployed_weights_fp32_math_sdpa | 0.000484467 | 0.000289917 |
| deepseek | native | 1.472774506 | 1.418304443 |
| deepseek | native_repeat | 1.472774506 | 1.418304443 |
| deepseek | native_math_sdpa | 1.472774506 | 1.418304443 |
| deepseek | cast_deployed_weights_fp32_math_sdpa | 0.001331329 | 0.001548767 |

FP32探针同时将部署BF16权重转FP32并使用math SDPA，不恢复权重被量化丢弃的位，也不替代正式结果。DeepSeek保留部署FP16特征取值；math SDPA不控制未调用torch SDPA的算子。Qwen math-only未消除差异，DeepSeek math-only无改变；FP32探针大幅减小差异，支持低精度计算路径是主要贡献，而不是输入改变或随机重训。范围仅8视频，未逐层分解每一算子，不能声称完整唯一因果证明。SRCC基于MOS排序，数值差异可改变秩；近水平曲线不意味着每条MOS完全相同。2026-10-14的19组正式测量未改精度、权重或选模；当前30组也沿用原数值路径。该八视频诊断不冒称已逐层解释六模型的全部差异。

## 历史总表锚点与LightGen-II口径

根据用户确认，当前两图及正式表中的Qwen B=1继续使用SRCC0.7663428036522781、1188.3861464262009ms/16、462.1134010767563J/16。当前新SRCC与其完全一致；新成本1230.453189611435ms/16、476.5010192155209J/16仅作审计，不替换历史定稿。历史成本来源为`LightGenV2/reports/20260928_nine_task_a100_table/latest_nine_task_summary.csv`中的01a及批准配置保存的未舍入值。图注、Sources与逐点provenance均标明此唯一成本例外；没有把旧成本、新功率和独立驻留util冒称同一轮。九任务总表未被本轮修改。

LightGen-II沿此前表格参考：N=16同光场，六周期6×(.714+.300+.0307)=6.2682ms；电子部件3.9773695913702247ms是整幅16槽场计时，不是测一条视频再乘16。router .8302489486336708、CCD后处理2.0079718434810636、bridge .2413414404541254、head .8978073588013649ms。总10.245569591370225ms；E16=.0062682×(80.358+41.388)+.0039773695913702247×(41.388+62.842)=1.1776895097085185J。每视频摊销.0736055943567824J，13.5859238518309videos/J；.6403480994606391ms/video为摊销服务时间而非单视频调用时延。光周期与电子FP32 CUDA-event部件估算不冒称本轮实测精确正式PT光路端到端，不能与baseline同步墙钟说成相同仪器/精度。


## English Methods

We evaluated temporal video-quality prediction on the fixed LGVQ split of 2,250 training and 558 test videos. Four frames were sampled at relative positions of 10%, 37%, 63% and 90%, retaining each model's original crop, native processor and resolution. All 30 previously established configurations of six frozen models were independently evaluated on the same ordered 558 test videos in fresh processes on an NVIDIA A100-PCIE-40GB at a 250-W power-limit setting. Original BF16/FP16 backbone execution, feature quantization and FP32 quality readout were retained. No training, checkpoint selection, precision modification or CUDA Graph was introduced. Checkpoints previously selected using this test split retain development-set status; frozen reevaluation does not create an independent generalization test.

For each batch, CPU input preparation and host-to-device transfer preceded native inference. These stages, model loading and post-inference CPU audit were excluded from the reported cost boundary. Synchronized host-wall timing began at the first native backbone block input (the first convolution for YOLO) and ended after the GPU MOS output was complete and synchronized. The boundary includes the subsequent native visual/language computation and quality readout, not only one Transformer layer. No explicit warm-up was performed, and the first call was retained. Costs used the first 200 distinct test videos, with all partial batches retained and batches split at the 200- and 558-video boundaries. Inference time was normalized as T16=(sum of the first-200-video boundary times)×16/200. Energy was estimated as E16=(338.2-W host-power proxy + same-run inference-active mean A100 board power)×T16/1000. This is the agreed power-times-duration estimate, not a calibrated per-call or whole-system wall-plug energy integral. SRCC and PLCC were calculated separately for each configuration using all 558 predictions.

Board power, driver utilization and memory were polled together at nominal 50-ms intervals with phase tags. Cost telemetry statistics used only inference-active windows for the first 200 videos; driver utilization represents kernel-busy time over the driver's averaging window, not SM occupancy or a 50-ms instantaneous quantity. Figure annotations instead report independently measured utilization under one uniform GPU-resident native protocol: the first B distinct inputs were prepared and transferred once; eight same-stream native forward calls were queued before synchronization. Each call still processed B videos, not a single batch of 8B videos. Each B used three identical repeats, each with at least 5 s of warm-up and 30 s of measurement, and all measured telemetry samples were pooled without selecting the highest repeat. No CUDA Graph was used. These utilization values are descriptive independent measurements; their power or timing is not substituted into the dataset-once costs.

Qwen B=1 retains the finalized historical T16=1188.3861464262009 ms and E16=462.1134010767563 J as the sole electronic-cost exception. Its new full-test SRCC matches the historical value exactly; new cost and telemetry are retained separately as audit data. The LightGen-II point uses the unchanged physical SRCC reference and optical-cycle/electronic-component estimate for a 16-video optical field, not a new exact-checkpoint end-to-end optical benchmark.

## English figure description

Temporal video-quality performance versus normalized inference cost on LGVQ. (a) Energy per 16 videos and (b) inference time per 16 videos, shown on logarithmic horizontal axes. The vertical axis is dimensionless temporal-quality SRCC. Each electronic point represents an independent full-558-video evaluation at the indicated maximum batch size B, with costs normalized from the first 200 videos and all partial batches retained. Lines connect increasing B within each model, without imposing monotonic cost or performance. Percentages beside the points denote independently measured native GPU-resident utilization under the uniform three-repeat queue8 protocol; they are not utilization measurements paired with the plotted dataset-once costs. Qwen B=1 retains the finalized historical cost, explicitly distinguished from its new audit measurements. The LightGen-II reference represents N=16 videos per optical field, with physical SRCC 0.7977 and unchanged component-estimated costs of 1.1777 J and 10.2456 ms. Numerical data, separate provenance, methods and raw-data receipts are retained in the accompanying tables.

## Reproducibility and evidence

Canonical run: `LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_frozen_batch_costs_a100_20261015`. Its original eight completed configurations are retained in place. The original GLM B=2 attempt stopped before producing any prediction when a foreign GPU process entered; it remains failed evidence and is excluded. Under the subsequent explicit user authorization, `continuation01` validated and reused the completed prefix and measured only the interrupted/unstarted configurations. Its suite and child reports are complete; the old failed parent status was deliberately not overwritten. All 30 valid settings passed offline recomputation from per-sample predictions, actual batch plans, tagged power samples and process audits.

Original measurement source was main344c0e3a9 (entry SHA c34444be26cf0cd62bc14be5002061a8388a98d8172b9b23566b63842297101e). Continuation source main d5f0d2603 uses entry SHA6222b7edbb7654fbfa11cad4c1579e673ef81070ea4fa095dfedd1c45d264ab4. CPU AST tests verify unchanged worker arithmetic, batching, cost formula, native timing, model hashes and idle guards; only orchestration for preserving completed settings changed. Later unrelated main commits do not change this bound source identity. Manifest SHA607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc is unchanged. Exact six checkpoint identities, per-point sources and source-file hashes are in `recalculation.json` and the SHA-backed raw receipts.

The final full-run backup contains 251 original files, with the initial adjacent launch/admission controls additionally archived by their own receipt. The offline verifier checks 252 distinct immutable/snapshot records across the receipts, including previously preserved failed/control evidence; this is an evidence-file count, not a number of measurements. Failed attempts never contribute to the 30 valid results. Earlier report bytes are recoverably archived under the same canonical run's `prior_report_archive`, with a SHA manifest. Old 15-s resident controls, precision diagnoses and the historical Qwen reference remain separate. The previous side-effecting `build_analysis.py` was not executed. Figure/table refresh is CPU-only and launches no GPU processes.
