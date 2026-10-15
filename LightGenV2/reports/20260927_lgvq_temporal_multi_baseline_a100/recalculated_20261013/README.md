# LGVQ temporal quality: matched A100 inference costs and batch diagnostics

Updated 15 October 2026 (both bounded layer diagnoses complete; formal measurements unchanged). All 19 frozen settings, earlier numerical probes, and the new eight-video Qwen/DeepSeek boundary and route experiments are complete. This does not mean every primitive or CUDA kernel has been isolated or a stable full-558 replacement evaluation has been performed. The displayed curves still use only uninterrupted `full_remeasurement02` measurements. Interrupted rounds and failed diagnostics remain preserved, excluded from final data. The original report location and three-workbook-sheet structure are retained.

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

图a展示每16视频推理能量与时间质量SRCC的关系，图b展示相同工作量的推理时间与SRCC的关系。横轴分别为Energy per 16 videos (J)和Inference time per 16 videos (ms)，使用对数坐标；纵轴为无量纲Temporal quality SRCC，范围0–1。B为每次GPU调用的视频数，N为每幅光场的视频数。每一个电子点都有其batch自己的558个TEST预测和同轮成本，不复制统一分数。LightGen-II仅为N=16物理SRCC参考和部件成本估算点，不是本轮重测光路。

LGVQ固定划分含2250 TRAIN、558 TEST，共2808个不同视频。每视频在10%、37%、63%、90%相对位置取4帧。Qwen/DeepSeek/CLIP/YOLO输入分别448/384/224/640像素正方形；后三者沿原短边65%中心裁剪。权重冻结，eval模式；各模型各B独立新进程。Qwen采用原历史head，其他三者采用2026-10-13原协议重训head，均保持PT/manifest SHA不变。重训head为50epoch/seed42/五级质量行/冻结backbone/CE，最高TEST选模为开发指标，无VAL。本轮无训练或选模。正式骨干精度Qwen与DeepSeek为BF16，CLIP与YOLO为FP16，质量头FP32；后三者保留FP16特征量化再FP32读出。

每批先CPU解码、裁剪和processor，再H2D，再推理，随后下一批。计时从首原生backbone block入口到GPU标量MOS就绪并完成同步；YOLO首block为卷积，其余为视觉Transformer。模型加载、CPU准备、H2D、之后CPU预测复制和审计开销不计最终成本。起点预钩子内无额外同步，因此为操作定义的同步墙钟边界，不是纯GPU kernel积分。无显式预热，第一调用保留。完整558视频统计SRCC，成本仅固定前200视频，并在200/558边界拆批，不丢尾样本。

T16=sum(first200 synchronized inference wall ms)×16/200；E16=(338.2 W主机代理+同轮tagged-active A100平均板卡W)×T16/1000。B=1/2/4/8时等价于mean batch ms×16/B；B=3/9/10含实际不足满批的尾批，不能用均值×16/B偷换。16仅为吞吐工作量归一化，不是仅测试16视频。

| B | Full TEST calls | Final TEST call videos | First200 cost calls | Final cost call videos |
| --- | --- | --- | --- | --- |
| 1 | 558 | 1 | 200 | 1 |
| 2 | 279 | 2 | 100 | 2 |
| 4 | 140 | 2 | 50 | 4 |
| 8 | 70 | 6 | 25 | 8 |
| 9 | 63 | 7 | 23 | 2 |
| 10 | 56 | 8 | 20 | 10 |
| 3 | 187 | 1 | 67 | 2 |

功率、GPU利用率与显存同轮按50ms轮询并标记阶段。主表统计只取前200视频各调用的active标签窗口，与成本子集配套；后358视频仍计算性能，但不混入这些active均值。驱动自己的平均窗口并非50ms瞬时值，util是kernel-busy而非SM occupancy或峰值FLOPS使用率。短推理窗口可包含前序CPU间隔影响，CLIP/YOLO主交错util较低不代表算子没有GPU运行。均值功率×边界时间是沿用的能量估算，不是校准短调用积分。A100设250W；原始短功率读数中存在超过250W的峰值（DeepSeek约321–342W），保留不裁掉，不把设置值描述为瞬时硬限。

## 图a：能量与时间质量性能

LightGen-II物理SRCC为0.7977，每16视频能量估算1.1777J。新同轮Qwen B1为458.3508J，B8为228.3324J，分别对应389.19和193.88倍任务能效比（相同工作量的能量反比），能量降低99.7431%和99.4842%。多数基线增大batch后每视频成本下降，但这不是必须单调的理论：CLIP B9高于B8，包含尾批及运行噪声，不能人为平滑。性能曲线近水平源于SRCC变化很小，不是固定分数复制。

## 图b：时间与时间质量性能

LightGen-II每16视频时间估算10.2456ms。新同轮Qwen B1与B8为1180.8067与512.4304ms，时间比分别115.25与50.01。增大B通常使单调用更长，但完成固定工作量的调用数下降并摊销开销，吞吐改善随后可能饱和或因显存/计算限制退化；不要求所有成本点单调，也不存在同权重准确率随batch上升的理论趋势。

## 全部绘图数据及主交错遥测

数据来源同一轮14 October 2026。以下20点含19电子设置和1光学参考。成本保留6位用于核对，不表示6位有效精度。对应未提前舍入数据为`plot_data.csv`与`timing_energy_recalculated.csv`；后者还保留mean/peak功率、利用率、显存、调用数及SHA。

| Model | B / N | SRCC (558 TEST) | Time/16 (ms) | Energy/16 (J) | Active util mean (%) | Peak VRAM (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| LightGen-II | N=16 | 0.7977138739 | 10.245570 | 1.177690 | not measured | not measured |
| Qwen3-VL-2B | 1 | 0.7663428037 | 1180.806745 | 458.350757 | 14.972 | 4705 |
| Qwen3-VL-2B | 2 | 0.7664148425 | 765.604070 | 296.237303 | 11.183 | 4883 |
| Qwen3-VL-2B | 4 | 0.7694958345 | 553.714911 | 233.495159 | 3.121 | 5107 |
| Qwen3-VL-2B | 8 | 0.7694089803 | 512.430360 | 228.332385 | 13.556 | 5603 |
| CLIP ViT-B/32 | 1 | 0.7252558080 | 166.339300 | 63.111235 | 1.898 | 903 |
| CLIP ViT-B/32 | 2 | 0.7252452404 | 94.548801 | 35.723075 | 2.474 | 923 |
| CLIP ViT-B/32 | 4 | 0.7252388170 | 52.953970 | 20.029098 | 0.600 | 931 |
| CLIP ViT-B/32 | 8 | 0.7252388170 | 34.412313 | 13.029472 | 3.273 | 975 |
| CLIP ViT-B/32 | 9 | 0.7252247615 | 36.311530 | 13.795622 | 2.000 | 995 |
| CLIP ViT-B/32 | 10 | 0.7252388170 | 31.707193 | 12.048240 | 4.444 | 999 |
| DeepSeek-VL2-Tiny | 1 | 0.5684669010 | 2917.680463 | 1332.941805 | 20.556 | 7771 |
| DeepSeek-VL2-Tiny | 2 | 0.5703507558 | 2039.096788 | 1008.243099 | 18.043 | 8479 |
| DeepSeek-VL2-Tiny | 4 | 0.5680463058 | 1714.183199 | 883.138613 | 38.224 | 9609 |
| DeepSeek-VL2-Tiny | 8 | 0.5698244506 | 1543.665272 | 819.439993 | 62.197 | 12065 |
| YOLO11s | 1 | 0.7015970082 | 336.490797 | 127.530261 | 4.358 | 841 |
| YOLO11s | 2 | 0.7015940037 | 189.279941 | 71.566832 | 3.409 | 1037 |
| YOLO11s | 3 | 0.7015940037 | 154.108012 | 58.414706 | 3.188 | 1259 |
| YOLO11s | 4 | 0.7015916554 | 132.892382 | 50.314170 | 2.032 | 1467 |
| YOLO11s | 8 | 0.7015904121 | 121.058940 | 45.901801 | 3.935 | 2327 |

## 连续负载：配套时延、能量与GPU利用率

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

## 精度变化的诊断证据

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

FP32探针同时将部署BF16权重转FP32并使用math SDPA，不恢复权重被量化丢弃的位，也不替代正式结果。DeepSeek保留部署FP16特征取值；math SDPA不控制未调用torch SDPA的算子。Qwen math-only未消除差异，DeepSeek math-only无改变；FP32探针大幅减小差异，支持低精度计算路径是主要贡献，而不是输入改变或随机重训。范围仅8视频，未逐层分解每一算子，不能声称完整唯一因果证明。SRCC基于MOS排序，数值差异可改变秩；近水平曲线不意味着每条MOS完全相同。正式19组未改精度、权重或选模。

## 历史总表锚点与LightGen-II口径

历史Qwen B1仍保留SRCC0.7663428036522781、1188.3861464262009ms/16、462.1134010767563J/16，不写入新同轮曲线，也没有GPUutil记录。新B1为1180.8067452907562ms/16、458.3507566057941J/16，性能相同但成本为独立测量；不能新util与旧成本拼接冒称配套。九任务总表未被本轮修改。

LightGen-II沿此前表格参考：N=16同光场，六周期6×(.714+.300+.0307)=6.2682ms；电子部件3.9773695913702247ms是整幅16槽场计时，不是测一条视频再乘16。router .8302489486336708、CCD后处理2.0079718434810636、bridge .2413414404541254、head .8978073588013649ms。总10.245569591370225ms；E16=.0062682×(80.358+41.388)+.0039773695913702247×(41.388+62.842)=1.1776895097085185J。每视频摊销.0736055943567824J，13.5859238518309videos/J；.6403480994606391ms/video为摊销服务时间而非单视频调用时延。光周期与电子FP32 CUDA-event部件估算不冒称本轮实测精确正式PT光路端到端，不能与baseline同步墙钟说成相同仪器/精度。

## English Methods

We evaluated temporal video-quality prediction on the fixed LGVQ split of 2,250 training and 558 test videos. Each video contributed four frames at relative positions of 10%, 37%, 63% and 90%. All 19 frozen model–batch configurations were evaluated independently on the same 558 test videos. Inference cost was measured on the first 200 videos and normalized to a workload of 16 videos. Partial batches were retained at the 200- and 558-video boundaries. Each configuration ran in a fresh process on the same NVIDIA A100-PCIE-40GB. For each batch, CPU preprocessing and host-to-device transfer preceded inference; these stages were recorded for audit but excluded from the reported costs. Synchronized wall-clock timing began at the first native backbone block and ended after the scalar quality score was available on the GPU. No explicit warm-up was performed, and the first measured call was retained. Energy was estimated as inference duration multiplied by the sum of the 338.2-W host-power proxy and the same-run inference-active mean board power.

GPU utilization, board power and memory use were sampled together at a nominal 50-ms polling interval. Main telemetry statistics used only inference-active windows for the same first 200 videos as the cost subset. GPU utilization denotes the driver-window fraction of time during which kernels were active, not streaming-multiprocessor occupancy. A separate approximately 15-s continuous-load experiment repeatedly processed the first B distinct test inputs already resident on the GPU. Continuous-load latency, energy and utilization are reported together and are not assigned to the interleaved-inference costs. Qwen and DeepSeek retained BF16 backbone execution, CLIP and YOLO retained FP16 execution, and quality heads used FP32. The LightGen-II point retains the physical SRCC reference and the previously approved optical-cycle plus electronic-component cost estimate; it is not a new end-to-end optical measurement.

## English figure description

Temporal quality versus inference cost on LGVQ. (a) Energy and (b) inference time normalized to 16 videos. Electronic points use the complete matched A100 remeasurement of 14 October 2026; B indicates videos per GPU call. SRCC is calculated separately from all 558 test predictions at each batch size. The LightGen-II reference represents 16 videos per optical field (N=16), with physical SRCC 0.7977 and component-estimated costs of 1.1777 J and 10.2456 ms. Larger batches generally reduce amortized electronic cost, but this trend need not be monotonic. The nearly horizontal performance curves do not imply batch-invariant predictions: bounded same-input probes support low-precision computation as a major contributor to the observed batch-dependent differences. Historical measurements, preprocessing costs and continuous-load telemetry are not mixed into these points.

## Reproducibility and evidence

Formal run: `LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_temporal_telemetry_a100_20261014/full_remeasurement02`. Measurement entry SHA28fc58f1; telemetry28ad0666. Manifest SHA607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc. Original Qwen PT e102adc6; reproduced CLIP28729f41/YOLO47606708/DeepSeek3748b08f. Exact per-file SHA and sources are in `recalculation.json` and the canonical run backup receipts. Old refreshed report files are recoverably archived in the same run's `report_before_refresh` with a manifest; all original measurements and failed attempts remain unchanged. The historical side-effecting `build_analysis.py` was not executed. Report refresh is CPU-only and launches no GPU processes.
