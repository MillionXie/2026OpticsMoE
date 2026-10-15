# T06 视频质量评价

## 2026-10-15：利用率执行对照完成（不改定稿成本与精度）

本轮新授权只查利用率/统计窗口/执行开销，不再处理精度诊断。原目录
[方法与完整结果](../../reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013/UTILIZATION_METHOD_20261015.md)
记录`lgvq_utilization_execution_20261015`：7组21项对照完整完成，原生PT/精度/输入不变。
Qwen B8旧整段89.22%包含启动期；原记录固定1–15秒91.05%，新固定5–30秒原循环91.89%、
去CPU取数92.06%、八调用队列92.30%。新B1原200视频同边界审计1245.912659ms，比历史
1188.386146ms高4.8407%，只审计不强凑一致；历史锚/六份正式表数据SHA均未变。
另独立CUDA Graph一次捕获对照：CLIP B10整段98.9431%、YOLO B3整段99.2642%，
两者固定5–30秒100%，两批不同输入MOS均与原生一致；时间/能量单列，不混原B1成本。
Qwen B8在原生位置嵌入插值处不支持整图捕获，失败原位保护、未patch模型或重试。
运行源码分别main`9a751b9c8`和`fd36098a8`，后续准入/文档修正不改已运行源码身份。
98份原证据已逐SHA备份本地同名run；全部自有A100进程退出、不占位。CPU审核见
`comparison_verified.json`，后续任何正式协议替换须明确区分旧交错、resident和Graph负载。

## 2026-10-15：继续逐算子因果排查（未替换正式数据）

**最新实际结果：4090两模型及固定输入隔离均完成，已退卡，后续先CPU分析。**
`operator_mechanisms04/rtx4090_qwen`实际记录相同操作数随矩阵行数改变而切换GEMM核；
B4/B1不同元素422346/1605632，FP32计算回BF16后118，仍非零。
`rtx4090_deepseek`同GPU原生无观测/有观测MOS差0；只转attention GEMM为FP32未稳定
整模型，B4批差.7458458→1.4039307，未作为修复采用。针对性
`rtx4090_deepseek_isolated`（main34e61ca2f，两端48CPU测试）保存真实上游输入后，
用相同B1 hidden重复B2/B4隔离attention；首差在Q投影，B4不同元素435414→265，
attention输出RMS.000330167→.000012077，但仍非零。4090上原生视觉上游已有差异，
所以不能将跨卡定位直接套用旧A100。完整数据/方法/英文解释在原比较报告README新
小节，不替换正式6表。本轮45份证据885092364字节已全部SHA备份本地，CPU独立按
原始PT重算关键差值与JSON一致、固定hidden/mask/position SHA一致；收据为
`backup_receipt_operator_all04.json`，审核为`cpu_final_verification.json`。自有4090
进程均已退出，无占位、无新增训练、无A100使用；不自动追加GPU任务。

**最新人工授权：用空闲4090/3090排查，不等待或干扰另一窗口A100测速。**
第一次A100派发父1477076/timeout1477077被共享锁busy拦截，在创建GPU输出和CUDA
初始化前退出，未执行Qwen，DeepSeek未启动。该记录保留，不是模型计算失败。
新入口支持显式诊断UUID白名单（不含GPU0），各物理卡单独锁/foreign进程守卫。
迁移目标空闲4090 UUID`GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd`，仍两模型串行
各600秒，不测正式成本。输出使用本轮子run内`rtx4090_qwen`和`rtx4090_deepseek`。
DeepSeek新增同GPU无观测/有观测原生MOS对照，跨卡相对旧A100的差另列，不用跨卡
差判定观测改变输出；卡名/UUID/功率上限和来源写进status。非A100只能支持机制
复现与数值对照，不能替换原A100精度/速度，A100特定kernel仍需原卡短复核。

用户要求不止定位边界，还要验证误差机制及最小数值修正。新增诊断入口
`diagnose_batch_primitives`，仍在同一canonical诊断run内保存、不得覆盖旧证据。
Qwen先用已存真实BF16操作数CPU检验固定split-K舍入假说；空闲A100上仅重放该
linear的B1/2/4矩阵形状并保存实际kernel名，以及FP32/GEMM归约/显式bias对照。
重复视频0的operator重放不是全模型吞吐或精度结果。

DeepSeek下一步观测实际首LlamaAttention内mm/bmm、RoPE算术、softmax及实际mask/
position参数，保全视频0差统计和大attention矩阵固定query切片；仍用同8 TEST、
同原PT/B1/2/4验证原生MOS。数值对照仅在语言attention内将GEMM以FP32计算并
回转原输出dtype，不改变权重值、不训练、不将对照填入正式558表格。
每模型GPU命令必须外层600秒上限；A100仅250W且无foreign compute PID才运行。
本轮GPU尚未启动，不占位或抢卡；入口先取得既有`.temporal_telemetry_a100.lock`
并核无foreign，再创建输出/初始化CUDA，尊重另一T06 Smol测量队列。CPU阶段已完成：
`operator_mechanisms04/cpu_qwen/qwen_cpu_mechanisms.json`，同BF16操作数的单次舍入
参考与实际B1/B4分别有374788/2111项不同（1605632项）；2/4/8/16段的部分和BF16
舍入、顺序BF16累加等固定假说均未精确复现实际B1，因此不称split-K已获证明。
较小的该层数值误差不等于任务SRCC更优。简单“dot先舍入再加bias”同样未复现。

实际部署Torch2.6cu124的LlamaAttention小型CPU接口检查覆盖两模式×B1/2/4共6组，
原生观测逐位保持输出、18个目标事件含linear/matmul/softmax；该版本inference
dispatch保留这些复合算子名，不能只盯mm/bmm而漏记。首份接口证据保留，更正完整
证据为`cpu_attention_interface_fixed.json`。两端45项合同测试通过后执行GPU；
这些是诊断接口及CPU机制结果，不是实际全模型GPU修复已验证。新证据已逐SHA备份本地。

## 2026-10-15：两模型实际定位完成并退卡（focused03最新证据）

`lgvq_batch_layers_a100_20261015/focused_merger_routes03`，执行main `e34b2a586`，
同8 TEST/B1/2/4、同原PT与manifest；DeepSeek→Qwen串行，两个模式各3组全部complete。
40项CPU合同测试两端通过。自有suite1293705、DeepSeek1296180、Qwen1302609均退出；
A100已无自有compute进程，不保留占位。44份JSON/日志/PT共257439718字节原位保存，
含12组末端features/logits、12组focused边界张量/专家ID；SHA备份仍在同名本地run。
此为诊断证据，不改正式558分数、成本、利用率、表格或PT。

**现在查到了什么：**Qwen首个已观测分歧细分到
`model.visual.deepstack_merger_list.0.linear_fc1:output`，B2/B4相同，此前22个记录
边界（含归一化输出及fc1输入）逐位一致；fc1输出max_abs=.125。保存的视频0 fc1
输入B1/B4逐位相同，形状[392,4096]；观测输入FP32，正式autocast将linear操作数转
BF16，输出BF16。用同一BF16权重及显式量化的输入在CPU FP64重算，B1/B4输出均为
同一数学输入的近似，不能把差异解释为换输入/权重/BatchNorm/MoE。未追踪具体cuBLAS
kernel，不声称禁低精度归约即可修复（该对照仍有差）。

DeepSeek的首个已观测分歧为`language.model.layers.0.self_attn:output`，此前64个
边界（含已观测视觉/投影输出和该attention输入）相同；max_abs=.001953125，
17848320个元素中4930项改变。**差异早于第一个MoE gate**；随后11层实际专家集合
均有变化：B4/B1为22251/153384个token–layer事件（14.5067%，不是视频比例）。
只在gate关闭autocast仍为21504/153384（14.0197%），未消除；不能把MoE作为最早来源
或唯一原因。未细分该attention内部Q/K/V、RoPE、matmul、softmax等primitive。

原生新旧8视频MOS最大差：Qwen7.6293945e-6、DeepSeek1.1444092e-5，原生守卫通过。
对照中最大MOS批差：Qwen B4/B1 3.395130→2.585453；DeepSeek 1.418308→.628998，
均未归零，且对照自身B1也变化；不是正式SRCC提升。证据支持低精度批形状相关数值
路径及后续传播/路由敏感性，没有发现正式输入/权重身份错误；不宣称排除所有实现
问题，也未做完整558稳定化重测。此前修复的是**诊断脚本**，不是修改正式成绩。

最新派生汇总只用`focused_cpu_analysis_corrected.json`：首份CPU fc1参考曾用量化前
FP32输入，保留在`focused_cpu_analysis.json`，更正版本明确先做部署BF16操作数量化。
两版均不改变原始GPU证据。完整边界/专家ID及原生一致性证据见对应PT/JSON与SHA收据。

## 2026-10-15：用户再次确认空闲后的短窗口细分诊断

用户明确授权现在继续使用空闲A100，优先DeepSeek路由、随后Qwen合并内部；
仅T06诊断/说明，不改正式测量。当前main的指纹顺序修复已两端39项CPU测试通过。
本轮同run子目录`focused_merger_routes03`，仅原8 TEST/B1/2/4，CPU/GPU都按
`--baselines deepseek qwen`串行、各600秒上限；CPU收据绑定选择顺序与源码SHA。
Qwen新增首deepstack merger内部norm/linear/GELU观测，不改模型forward；每组保存
视频0这些边界完整张量及所有已观测专家ID，连同末端特征/logits和差统计，避免只留
聚合数据无法离线复核。保留旧失败/成功原位，结束立刻退卡、不占位或重训。

## 2026-10-15：短诊断已退卡，Qwen完成、DeepSeek待补实际路由

用户后续要求尽可能保存中间数据、A100还需用于其他工作。本轮不再追加GPU任务。
自有suite1251252、Qwen1251256和DeepSeek1254035已退出，未保留占位/等待进程。
`lgvq_batch_layers_a100_20261015/recovery_hooks02`执行源码`881393f78`；CPU同源收据
完成，Qwen原生及关闭低精度GEMM归约对照各B1/2/4全部完成，原生8视频MOS与旧探针
最大偏差守卫<.001通过。6份8视频features/logits PT、逐边界统计、CPU身份、命令/日志
共24份证据按SHA备份本地，随后另存DeepSeek纯CPU转换审计；不保存/冒称完整逐层激活。

Qwen首个观测分歧为`model.visual.deepstack_merger_list.0:output`：B2/B4相同，之前
19个记录边界逐位一致；B4该模块输入差0、输出max_abs .0625、relativeL2 .0029048164。
原生MOS最大B2/B1差2.350662、B4/B1差3.395130；关闭低精度归约后分别2.585449／
2.585453，未消除差异。该模块含LayerNorm和两层Linear/GELU；不能仅凭模块边界指定
最早primitive，也不能把Qwen差异解释为MoE。8视频诊断不替代正式558性能和成本。

DeepSeek在首次前向前因指纹检查顺序被拦截，没有有效逐层/路由输出。
`cpu_transfer_fingerprint_audit.json`使用同一视频和原processor，仅CPU复现：转换前
指纹与旧探针一致，原生DictOutput.to(cpu,BF16)返回同一容器，images从FP32转BF16，
仅pixels SHA改变。诊断曾在transfer之后比较转换前指纹；不是模型输入身份改变的
证据，更不是已查清DeepSeek路由。已将指纹守卫移到transfer前，补原地变异回归，
本地39项CPU检查通过；修复后未再启动GPU、不重跑已完成Qwen。下次只需在新CPU来源
和明确短窗口下补DeepSeek。正式六份表/测量SHA仍保持原值；旧失败证据全部保留。

## 2026-10-15：用户确认A100空闲后授权恢复逐层诊断

本轮只负责T06诊断源码/测试/说明，不改正式表格、558预测、成本或PT。
修复直接诊断forward前遗留的正式计时钩子：先close/remove成本计时观测，不改模型
forward、权重或精度。补计时状态回归和调用顺序测试，本地37项CPU检查通过。
上次失败原位保护；恢复使用同一run内`recovery_timerfix01`，按修复后Git源码先新做
CPU输入/位置收据，再同源GPU串行Qwen/DeepSeek各600秒上限、同8 TEST/B1/2/4。
`recovery_timerfix01`越过计时错误，但在首视频Qwen语言注意力的命名参数输入钩子处
失败；14份证据已SHA备份，无有效批比较，DeepSeek未启动，GPU进程已退出。
已据实际栈修正pre-hook的`with_kwargs=True`及hidden_states显式取值，补回归测试，
两端核接口后在`recovery_hooks02`重新生成修复来源CPU收据；不是原位覆盖或盲重启。
用户要求尽可能保存中间数据并尽快释放卡：每个模式/批量完成即落盘边界差统计及
8视频features/logits；不为分析占位、不跑全558、不中途改变正式成本。GPU阶段结束
立即退出并备份，后续复核用已保存数据。旧监督已删除，不复活其旧派发命令。

## 2026-10-15：Qwen/DeepSeek batch根因定位（新授权）

用户要求继续查清两模型差异：先不占GPU，A100空闲后再验证逐层误差和DeepSeek路由，
及时报告，正式数据不改。新入口`run_batch_layer_diagnostics`/`diagnose_batch_layers`，
唯一新run `runs/simulation/lgvq_batch_layers_a100_20261015`，源码发布`5e13ea536`。
本地与服务器36项CPU合同检查通过；两模型CPU检查已实际complete，9份原位JSON/日志/
命令按SHA备份本地，不是GPU逐层运行完成。

同一原8 TEST视频、B1/2/4，输入图像/有效文本/抽帧SHA与旧诊断一致。
Qwen各组全部446有效token、0 padding、读出index445；原生CPU `get_rope_index`的
逐视频有效position SHA跨B一致，rope_delta均−364。DeepSeek全部1743有效token、
0 padding、读出index1742；fresh无cache的位置arange按实际DeepseekV2Model源码推导
一致，不冒称已观测实际GPU position tensor。此结论仅覆盖这8视频，不推广到全558。

2026-10-15第11次监督核查时，固定A100已空闲；连续两次核无compute PID、250W、
0%/17MiB且源码SHA与CPU收据一致后，实际派发一次GPU suite PID889806／Qwen889816。
Qwen第一次前向触发`KeyError('vision_calls')`，suite失败，DeepSeek未启动，未生成
`gpu_layers.json`或有效逐层/路由结论。原因是诊断复用正式loader后保留了
`BoundaryTimer`的原钩子，但直接调用诊断forward未先reset计数状态；该错误属于
诊断集成，不是模型batch差异的根因，也不意味着正式测速存在同一问题（正式路径
每次forward先调用timer.reset）。全部14份CPU/GPU状态、日志、命令已逐SHA备份本地，
收据`backup_receipt_gpu_failed_check11.json`。失败记录原位保留，不盲重启覆盖；
下一步须隔离非计时诊断中的遗留计时钩子、补CPU合同测试并以新Git/CPU来源收据
明确批准恢复方式后再测。自有suite/子进程已退出，A100无自有compute PID。
不占位、抢卡或并发测试。后续仍只允许两个新进程串行，各最多600秒，原PT/receipt/
manifest保持；仅同8视频/B1/2/4做观测边界全张量差和专家ID统计，CPU参考上限8GiB，
磁盘只保存统计及小读出PT。原生预测必须与旧探针一致，否则停止、不作因果结论。
Qwen受控对照只禁低精度GEMM归约；DeepSeek只在gate范围关闭autocast，通过hook管理
上下文，不替换forward源码，不训练、不新增层，不把诊断写入正式指标或时延能耗图。
首个观测分歧边界不等于已定位最早primitive，尚未宣称任何一个模型根因完全查清。

监督`lgvq-qwen-deepseek-batch`每20分钟最多12次，等待/正常增量静默，重大发现、故障、
完成或需要人工协调通知；到限如实交接并删除，不无限等待或复活Smol旧任务。
GPU显卡空闲后，按同run CPU保存的源码SHA核验一致再派发一次：
`CUDA_VISIBLE_DEVICES=GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55 /DATA/DATA1/guest3/.venvs/lgvq-visual-baselines/bin/python -u -m LightGenV2.tasks.t06_video_quality_assessment.run_batch_layer_diagnostics --phase gpu --output LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_batch_layers_a100_20261015`。
若已有GPU输出或失败，不盲重试覆盖；只Git同步源码、不接实验室。

## 2026-10-14：SmolVLM2 等规模末端读出对照（独立授权）

**2026-10-15 Ovis单B1最终核验（本轮有限筛选结束，未达到超过ours的目标）：**
`lgvq_ovis25_temporal_20261015/suite01`已完成原2250TRAIN/558TEST、2808条冻结特征、
44分片与完整FP16缓存逐值及样本顺序/标签核对、50轮原预算五行训练；训练收据53项
artifact SHA与全部运行源码SHA通过。best e40 strictCPU缓存开发SRCC=.7933612885340906，
last e50=.7929585476443863，不能用缓存开发分数代替独立原生B1。
freshprocess A100 B1完整558唯一视频的SRCC=.7918416340914095、PLCC=.8061727368258899，
均从逐样本CSV独立重算吻合；相对参考ours约.7977低.0058583659，未达标。
best PT SHA `ee4b4e823391d018b39871cf53c18ab30eccfd5f62405b8cb3e3eb3527837ecc`，
training receipt SHA `aed3a8d259a4cf3347c5fcd9cd252318a3fba5d3998f4e0ac80c949b877fb4b1`。
全部77文件24,428,620字节已按SHA备份本地同名canonical suite，见
`backup_receipt_complete01.json`；suite及子进程已退出，A100无本聊天compute进程。
仅accuracy-only，无本轮时延/能耗结论；按TEST最高SRCC选模，全部为开发指标、无VAL。
不改分、不另扫PT或超参，不重启旧模型；本轮监督按已批准有限规则结束。后续若需新
候选/更高模型规模或预算，应先明确有限方案，不自动无限续跑。

| 本聊天候选 | A100原生B1 SRCC（558视频） | 相对ours约.7977 |
|---|---:|---:|
| SmolVLM2-2.2B | .7699090393508032 | -.0277909606 |
| InternVL3.5-2B | .7707267117951120 | -.0269732882 |
| Ovis2.5-2B | .7918416340914095 | -.0058583659 |

**2026-10-15 Ovis准备恢复与原生输入合同：**Git待发布的`ccb677caf`已确认在远端，
服务器通过main同步后14项CPU合同测试通过，`lgvq_ovis25_temporal_20261015/prepare`
已实际启动单CPU固定SHA下载（wrapper1708546）；不是训练。官方`configuration_ovis2_5.py`
及`modeling_ovis2_5.py`均已完整只读审查：原生video/NaViT、视觉soft-token→VTE、原Qwen3
未改图；元数据CPU meta模型已构造通过，质量词均原生单token17082/84103/60795/15216/49655，
尚无GPU性能结论，不需要升级现有环境。专属入口`ovis25_temporal_baseline`固定每视频
四帧/原65%crop/448，原生preprocess显式min=max448²、grid[4,28,28]、784视觉token，
Siglip原生0.5归一化，无额外帧/crop/thinking/生成；B2仅批处理独立视频，B1评估。
原视觉merge后取原语言骨干最后有效token，只更新独立五行原生LMhead副本；GPU smoke须
核原生forward末token logits与feature tap一致、原骨干及原生行无梯度/不改。数据配置及
固定官方依赖SHA均入身份；suite一次有界CPU预检→两TRAIN smoke→2808特征/原50e→
A100 B1完整558，3h总界/训练2h，错误保部分证据无自动重试，不更改已封存Smol/InternVL。

**2026-10-15 InternVL单B1实际交付与下一候选准备：**
`lgvq_internvl35_temporal_20261015/suite01`已完成：官方资产SHA、真实CPU/GPU smoke、
2808冻结特征（44分片逐值等于完整FP16缓存）、原2250/558划分、50轮五行头训练，
best e43 strictCPU缓存开发SRCC=.7726929658935687，last e50=.7724298823660634。
freshprocess A100 B1完整558结果SRCC=.770726711795112、PLCC=.7814365039173818，
低于参考.7977约.0269733；accuracy-only，无正式时延/能耗结论。best PT SHA
`ad20dd527da4ea7c6ea5fb175a9e8744b93f6247bd1a92d814ccd386182d3712`，
训练55文件24,182,768字节及B1四文件83,551字节已逐SHA备份到本地同名canonical run；
suite complete，自有A100推理进程已退出，不重训/不扫PT强凑超过ours。
按用户“不成功即换下一个”的授权，有限下一候选采用官方`ATH-MaaS/Ovis2.5-2B`，
固定revision`393c932b2a03e28eb9aaa503e3c4ab3ad384d958`；官方配置hidden2048，
仍只五行10240、原4帧/划分/50epoch/B1，不thinking/CoT/LoRA/全微调。
profile为`configs/baselines/ovis25_temporal_4f.json`，资产blob/LFS来自公开镜像API，
revision及维度已独立对照官方HF配置页；官方自定义依赖源码须先审查/CPU兼容性核验，
不得通过修改第三方计算图或升级既有环境绕过失败。准备/下载不称训练，结果未保证。

**2026-10-15最新简化与清理授权（覆盖下文三batch候选计划）：**寻找后续候选只固定
测B=1完整558TEST判断性能，不再为筛选候选跑B4/B8或正式成本。InternVL入口的有界
CPU suite自动等待已启动的官方权重下载，依次CPU输入核验、真实TRAIN两视频GPU smoke、
原预算50epoch五行头训练和freshprocess A100 B1；允许本用户另一诊断并发，标accuracy-only，
不抢卡、显存挤占或改功率，初次至少18GB空闲显存才准入。总界3小时/下载等待1小时/
训练2小时，失败保存证据不自动复活。训练入口独立，不修改原Smol/Qwen源码。
Smol旧三组已完成，B1=.7699090393508032、B4=.775330017371687、B8=.7751693975913737，
每组558预测与配套成本原证据各8文件已本地逐SHA备份，未超过参考约.7977。
用户明确允许为释放服务器空间删除Smol官方预训练大权重；须确认无运行使用并记录精确
文件、SHA、大小。小型五行best/last、特征、训练收据、逐样本结果和代码全部保留；
官方大权重可按固定revision/LFS SHA重新下载，不因清理改变历史收据或分数。

**2026-10-15精度优先的新授权：**用户说明A100上的另一诊断聊天可并行，现阶段只优先
考察能否超过本项目Temporal SRCC（参考约.7977）；速度/能耗不作为等待条件。
并发期间的成本不得冒称正式公平测量；已经有效完成的Smol记录保护，不改分或重训。
Smol原生A100 B1=.7699090394、B4=.7753300174均未超过参考，B8按既有队列收尾。
用户已授权不成功即换下一个：先准备官方`OpenGVLab/InternVL3_5-2B-HF`（约2.3B），
固定revision `3f301ffcf3dcbb47893afae6650ea3e78d96fb6d`，官方Git blob/LFS SHA校验。
仅下一模型原协议单次50epoch、五行10240参数、骨干全冻结/无LoRA、同4帧/原划分；
不扫超参或强凑超过ours。权重下载只是CPU准备，不能冒称训练已开始。
官方HF视频processor默认384/CLIP归一化需与vision配置448/256token接口适配，候选
明确使用模型原生448及官方image processor的ImageNet归一化、每帧单patch，不增帧/
多crop；须先真实CPU/GPU smoke通过。精度优先推理若允许并发，显式单列accuracy-only，
不把重叠时段的时延/功耗纳入正式结果。源码仍仅Git，模型/证据不进Git。

**2026-10-15新授权覆盖训练设备限制：**用户要求先在空闲3090上验证、提取冻结特征并
完成末端训练，正式B1/4/8准确率与时延/能耗/利用率仍在A100上配套复测。只允许训练和
两视频`gpu-smoke`显式传入`--training-gpu-uuid GPU-d53ce4c8-272d-c2fb-dc09-f182d586c4eb`；
3090为物理GPU5，不碰GPU0或其他进程，不改变功率。`evaluate`和旧`suite`拒绝3090覆盖，
保持A100/250W测量守卫。BF16原生前向、B2特征提取、FP16缓存→FP32五行头及其余科学
协议不变；不因换卡改精度/LoRA/帧数/训练预算，不保证提高SRCC。新训练特征完整重提，
不混用上次A100失败的128条部分特征；原失败证据保护。
两条真实TRAIN视频的smoke只验证显存容量、有限特征及五行头一次更新/骨干冻结，不是
正式精度或成本结果。通过后同canonical run下`rtx3090_train_20261015`完成2808特征、
50epoch与CPU best/last核查，不在3090测正式速度/功耗，不在训练期间占A100。
旧监督已删除；本条是用户新的有界运行授权，非延长旧等待。源码发布后运行，既有共享
Qwen/CLIP/YOLO/DeepSeek测量与图表不改。后续A100必须读训练收据与同SHA best PT、
同划分/模型/源码，逐B原生重算558预测；3090缓存CPU开发分数不冒称A100 batch性能。

用户随后明确要求训练完立即衔接A100，不等提醒。独立CPU调度入口
`smolvlm2_evaluation_queue`每15秒检查训练状态，确认50轮、best/last与全部收据SHA、
558条CPU预测通过后，再等待A100/250W空闲和共享测量锁；等待不建立CUDA上下文。
有界等待A100最多1小时，本轮训练启动后总界4小时，每个B上限30分钟且不超总界。
一次取得锁后B1→B4→B8串行，各为既有`evaluate`的新进程与新输出目录，保持同best PT。
训练失败、源变化、外部进入导致评估失败或到限均保护证据并退出，不自动重训/反复测。
每个B的558独立预测与前200成本、同轮功率/利用率/显存/时延配套；resident另计不混。
该入口只编排既有训练与评估，不改变正在运行的Smol模型、训练源码或测量算法。

**本轮训练完成核验：**`rtx3090_train_20261015`已完成2808条冻结特征、50轮五行头训练，
44分片与完整缓存逐值相同，样本顺序/标签/2250 TRAIN+558 TEST与原manifest一致、FP16有限。
best e41 strictCPU缓存开发SRCC=0.773713217488881，last e50=0.7736313362807273；
这不是A100原生B1/4/8性能。55文件共24,176,071字节已逐SHA备份到本地同名run，
两个3090自有进程均退出。best PT SHA `8e8ca80d00adae93ad19a787c6e8f22fd2ac2590490e6c3095f22fe1c4dc3ec9`。
队列`a100_queue_20261015`（PID1407887/timeout1407885，发布源码`6c2dc11b8`）两端CPU
合同测试通过（服务器48项），已实际验证训练收据并进入空卡等待；当次foreign PID1365456。
每15秒主动衔接，不等心跳或用户提醒；不另起重复队列。单次等待上限仍1小时，总界不延长。

用户授权在固定A100上训练SmolVLM2-2.2B，再分别以B=1/4/8新进程冻结推理。
入口`smolvlm2_temporal_baseline`及配置
`configs/baselines/smolvlm2_temporal_4f_a100.json`；不改原Qwen/CLIP/YOLO/DeepSeek结果。
只训练五个无偏置、与输入embedding解绑的原生质量输出行，5×2048=10240参数，
与Qwen改造程度相同。所有视觉/语言/connector/原LM head冻结；无LoRA、新adapter、
backbone更新或超参扫描。Smol使用前导空格质量词，各自单token；不带空格的Excellent
是两token，故不能直接沿用Qwen的token ID。token ID和初始行SHA进入收据。

保持原2250 TRAIN/558 TEST/无VAL、四帧位置[.10,.37,.63,.90]、65%中心裁剪和
同一Temporal问题。解码先448×448；Smol原生视频processor再处理为384×384，
每帧81个视觉token，四帧324个。该原生几何差异公开记录，不声称与Qwen视觉token数相同。
BF16冻结骨干、B2提取FP16特征后转FP32训练五行头，seed42/50epoch/b512/
AdamW .001 wd0/cosine/CE。best按TEST SRCC选取，是开发指标，不冒称独立泛化。
保存best/last、50轮history、完整特征/分片、558预测、strictCPU best/last及SHA。

模型固定官方revision`482adb537c021c86670beed01cd58990d01e72e4`；服务器官方域名
DNS暂失败，允许公开镜像下载，但配置/tokenizer用官方Git blob SHA1、权重两分片
用官方LFS SHA256逐项核验。已有不同文件和失败partial保留且拒绝覆盖；不读凭据。
服务器现有Torch2.6/cu124、Transformers4.57.3支持模型；仅在专属system-site-packages
环境补num2words，不升级或改写现有baseline环境。源码只Git main发布同步。

运行目录仅`runs/simulation/lgvq_smolvlm2_temporal_a100_20261014`。唯一物理卡
`GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55`，要求A100/250W及无foreign compute PID；
不抢卡、占空计算、杀人或改功率。suite自动下载→冻结特征→末端训练→B1/4/8各完整558。
2026-10-14启动前出现他人A100进程，守卫拒绝GPU启动，没有训练或正式结果。
CPU下载不占GPU；suite对foreign PID仅CPU等待，最多从suite开始3小时，超过即失败退出，
不扩大预算或建立显存占位。GPU阶段中出现foreign PID继续失败保护，不盲重启。
成本首200视频并保留200/558处尾批，无warmup，起点原生视觉Transformer block0输入，
终点GPU MOS同步。保留末token完整native词表投影，与Qwen计时边界一致。
T16=前200视频推理墙钟之和×16/200；E16=(338.2W主机代理+同轮active A100 W)×T16/1000。
解码/预处理/H2D只审计，不计最终成本。50ms遥测同时记录power/util/显存，util是
驱动窗口kernel-busy而非SM occupancy。另15秒resident重复负载与主交错结果分开，
不混配利用率/功率/时延。各B精度独立原生前向，不能用缓存B1替代。

实际运行更新：原CPU预检在Smol视频嵌套接口处失败，未进行GPU训练；修复经Git发布后
真实视频CPU B1/4/8均通过，42项服务器CPU合同测试通过。唯一当前suite PID291753在
同run的`attempt02`（原失败记录原位保留），启动源码main`5dbb17456`。
21:52核验第一片4,959,390,584字节已校验落盘，第二片实际下载到532,676,608字节；
A100仍由foreign PID260192占用，尚未加载GPU/提取特征/训练，不把下载称训练开始。
本聊天监督`smolvlm2-a100`每20分钟最多12次，记录实际增量；空卡等待本轮suite最多3小时。
本段不是训练完成或更高性能结论；后续实际阶段以run status/log/receipt为准。
原图表的共享报告暂不修改，本轮新证据审核完成后再确定纳入方式。

**第5次监督实际核验（2026-10-14 23:33 +0800）：**两片官方权重已完整下载并SHA通过；
原占用结束后，`attempt02`确实加载模型并提取128/2808条冻结视频特征（两个64条分片）。
随后新foreign A100 PID429114进入，运行守卫抛出并发占卡错误，suite PID291753与
提取子PID425587均已退出，自有GPU进程已释放。尚未进入五行头训练（0 epoch），
没有best/last或B1/4/8正式推理结果，不能把特征提取称为末端训练完成。
两个FP16分片均为64×2048、有限数值、128个不同样本ID且身份一致；但守卫每64条
检查一次，不能证明此前所有条目均无并发。仅保存为部分证据，不纳入正式成本结果。
特征、日志、身份及失败status共8文件已在本地同名`attempt02`逐SHA备份，见
`backup_receipt_check05.json`和`check05_audit.json`。原失败记录与下载预检记录不覆盖。
当前不自动重启、不扩预算、不干预他人进程；需要用户协调稳定A100使用窗口后再继续。
监督仍有限12次，原3小时边界不延长；本条是失败交接而非性能结论。

**2026-10-15 00:53 +0800到限交接（第9次核查）：**原3小时边界00:47已过，
A100仍有foreign PID429114（约12.9GB/100%），自有suite与提取子进程均不存在。
失败status/log及128条部分特征SHA未变，本地8份备份再次全部通过校验。
已通过应用删除`smolvlm2-a100`监督，仅删除调度，不删除源码、官方模型、缓存或证据。
未完成项：完整2808条冻结特征、50轮五行头训练及strictCPU审计、B1/4/8各558视频
配套推理/遥测和独立Smol结果汇总；目前无可用于论文的Smol性能或成本结果。
最终核查与剩余项留在同run的`attempt02/check09_final_handoff.json`。
继续需用户确认稳定A100使用窗口与新的有界运行授权；不延长旧等待，不自动复活任务。

## 2026-10-14：配套 GPU 遥测复测与 batch 数值诊断

**用户质询后的图注更正：**主成本图已在每个batch点标与其负载配套的平均GPU利用率。
主交错与resident成本图分开，另有resident配套图；B8 resident平均Qwen89.22%、
CLIP71.30%、YOLO97.10%、DeepSeek95.26%，不混峰值与平均。输入resident但仍有逐调用
同步、CPU MOS拷贝和记录，不能将此循环称为无CPU间隙的纯GPU饱和实验。
复测/诊断完成不代表数值稳定化已应用：正式BF16/FP16分数仍有batch差异，FP32仅是
8视频诊断，未运行新558 FP32正式对照，不修改原曲线分数。详细证据与限制在原比较报告。

**完整交付更新：**`full_remeasurement02`19组冻结测量及两项8视频诊断均完成，
226份原始记录已SHA备份，558 TEST/前200成本/尾批/固定PT/manifest/PID审计通过。
DeepSeek接口修复后的诊断在`diagnostic_deepseek_interfacefix`，原失败证据不删除。
其B2/B4原精度MOS差1.472775/1.418304，math-only不变，FP32探针降到.001331/.001549；
Qwen/DeepSeek输入与抽帧SHA一致，重复原精度前向相同，支持低精度计算路径为主要贡献，
8视频探针不作为全部算子的唯一因果证明，也不改正式精度/权重。
原[比较报告](../../reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013/README.md)
已更新为20绘图点（19电子+1光场参考），原三页Excel、CSV、PNG/SVG及中英文论文说明同路径。
主交错与15秒resident连续负载各自保留配套时延/能量/利用率/显存，绝不交换使用。
新同轮Qwen B1为1180.806745ms/458.350757J；历史总表锚1188.386146ms/462.113401J
单独保留，不将新util配旧成本。LightGen-II原N16成本参考10.245570ms/1.177690J未改，
其电子3.977370ms为整16槽场计时。原报告可恢复备份在同run的`report_before_refresh`。
全部自有A100 suite/诊断进程已退出；不再启动推理/占位/训练。

**20:14 核验：**`full_remeasurement02`的19组正式测量均完成，各558个TEST视频。
Qwen附加8视频数值诊断完成：输入/抽帧SHA逐视频跨B相同，原精度重复前向相同；
BF16原生B2/B4对B1最大MOS差2.350662/3.395130，math SDPA仍为1.700470/1.902405；
部署权重转FP32的math SDPA诊断降至0.000484/0.000290。仅为诊断，不更改正式精度或数据。
DeepSeek附加诊断在输入指纹阶段因官方DictOutput无`__contains__`触发KeyError(0)，
不影响已经完成的四组正式测量。原失败记录与210份本轮证据已逐SHA备份。
仅修复诊断keys接口，并补同一8视频有界诊断，不重复19组测量、不重训或增加batch。
当前自有suite/诊断进程已退出，核A100无compute进程、250W；表图尚待完整审计更新。

**19:20 用户再次明确要求继续：**重新只读确认A100空闲、250W、无compute进程后，
启动全新 `full_remeasurement02`，suite PID4091416／首YOLO PID4091420。
实际预测文件45行并继续到日志51/558，PID对应固定A100且无外部A100并发。
启动main `cf3f6c50e`，四份测量/诊断源码SHA与原授权一致，不重训、不改计算精度。
之前两轮受干扰记录仍保护且排除，当前只监控此新suite，不复活占位或重复启动。
这是新的实际启动记录，不是已完成结果，也不宣称已获得调度层独占保证。

**19:05 第二次并发中断：**占位PID4060490在外部进程退出后销毁context并进入
`full_remeasurement01`；首YOLO B3测量PID4068142到75/558时又检测到外部
PID4068699，守卫19:05:52退出。该轮完整报告/有效完成组为零，不纳入最终数据。
占位与测量自有进程已退出，39份原始证据逐SHA备份本地同名run；A100后续虽
出现短暂空闲，不能当作稳定独占窗口。不静默重测第三轮，待用户确认对方暂停
在该卡自动启动任务／安排稳定使用窗口。既有表格图和历史权重不替换。

**18:52 后用户追加授权：**先前空闲窗口已完成YOLO B3与CLIP B9，各558 TEST；
CLIP B10到250/558时外部进程4035998进入A100，守卫退出，套件停止。
前两组原报告、预测、时延与遥测及失败B10证据共28文件已逐SHA备份本地，
不纳入用户随后要求的“完整重测”新轮次。
用户要求可见占位、待外部任务退出再完整重测。
`reserve_temporal_a100` 只建立一个CUDA context与64MiB空tensor，不运行假负载、
不填满VRAM、不驱逐/杀进程或改功率；创建前要求至少2GiB余量。
这是合作式可见占位，不是独占GPU的调度锁，也不保证其他人退出。
等待活动完全不计入性能、时延、能量或利用率结果。
外部A100进程退出后销毁占位context，fresh suite全新测量输出到原run下
`full_remeasurement01`；旧轮次和受干扰数据仍保留，不能混入新轮次。
仍使用原等待截止2026-10-14 21:24 +0800，不无限占位或自动反复重测；
若后续再出现外部进程，测量守卫继续退出并通知，需要稳定使用窗口。

用户授权复用原固定 PT 在唯一 A100 UUID
`GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55` 上重新冻结推理，不重训。
入口 `run_temporal_telemetry_suite`，输出限定同任务
`runs/simulation/lgvq_temporal_telemetry_a100_20261014`。每模型每 batch 新进程；
先补 YOLO B3、CLIP B9，B9 连续负载平均 GPU 利用率小于98%时仅再补B10；
随后四模型各B1/2/4/8。不为达到利用率继续扫描，不保证B10达到98%。

主测量保持逐批CPU准备→H2D→冻结推理，完整558 TEST计算各B的SRCC，
首200视频成本包括不足满批的尾批。对B3/9/10在第200视频处另设批边界，
保留该实际处理方式，不丢弃尾视频或假装200可整除。
T16=首200视频推理边界墙钟总和×16/200；整除200的旧B与原mean×16/B等价。
能量仍为(338.2W主机代理+同轮active A100平均W)×T16/1000。
起点仍是原生首个骨干block输入，终点MOS在GPU就绪并同步；预处理、H2D等只审计。
YOLO起点为首卷积block，不写为Transformer。

同一50ms轮询同时保存功率、GPU kernel-busy利用率及显存，均值/峰值与阶段标签
可追溯。nvidia-smi利用率有驱动自身平均窗口，不能把50ms轮询称为50ms瞬时利用率，
更不能称为SM occupancy或峰值FLOPS使用率。短active窗口会受前序活动影响。
每组另做15秒GPU-resident首B个不同视频连续推理，独立记录该负载的时延、功率、
利用率与显存，不将连续负载利用率配给交错执行的主成本。98%条件只作用于此明确
连续负载；16/B在此为吞吐折算，不冒称实际处理16视频的整数次调用。

Qwen/DeepSeek另对前8个固定TEST视频做B1/2/4有界诊断：逐视频有效文本及图像
tensor SHA、抽帧位置、重复原精度前向、math SDPA、部署权重转FP32 math SDPA。
保存小特征/logits证据，验证诊断原精度前向与正式前向一致；不改PT、标签、损失
或正式精度，不以诊断性能选模。FP32转换不能恢复BF16权重已舍弃的位；DeepSeek
仍保留原FP16特征读出量化，math SDPA不控制不调用PyTorch SDPA的算子。
不把“降低差异”直接等同于定位全部原因。诊断失败保留证据，不覆盖正式复测或盲重试。

历史图及Qwen B1总表锚点仍保护。新同轮配套结果单列；本轮尚未完成时不得用旧
GPU利用率补造新点或将pending写成实测。完成并核验后更新原比较报告和表格。

## 2026-10-13：LGVQ frozen baseline 批量复核

用户授权在服务器 A100 上补做冻结权重推理，不训练、不采光路。唯一入口
`evaluate_temporal_baseline_batches`，每个模型/B=1,2,4,8使用独立进程，完整558条
TEST计算SRCC，前200条按原顺序测成本，无warmup且首批计入。输出限定在本任务
`runs/simulation/lgvq_temporal_frozen_batch_a100_20261013/<model>_batch<nn>`。

沿用总表的交错执行顺序：每批解码/processor、H2D、冻结前向，再下一批。
最终时延只取原生首个backbone block输入到MOS就绪的同步墙钟；解码、预处理、
H2D及审计开销只留记录，不纳入最终时延或能耗。T16=(16/B)*mean_batch_ms；
E16=(338.2W主机代理+推理活动期A100实测平均板卡W)*T16/1000，不加CPU间隔。
CUDA-event为并列诊断，不替代墙钟。GPU固定UUID
`GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55`，校验型号/实际功率上限/其他compute PID；
其他任务显卡不动。已有输出、非正式PT/manifest SHA在GPU初始化前拒绝。

Qwen batch=1的最终展示锚保留总表原记录：SRCC .7663428036522781，
16视频1188.3861464262009ms、462.1134010767563J。新B1复测另存审计，不强迫新测量
等于历史值；其余批量须标明测量日期/执行协议，不能拼接旧连续GPU sweep冒称同轮测量。
原始预测、总表与旧sweep均保护。CLIP/YOLO/DeepSeek须精确原训练读出头，
随机timing head不用于性能结果，缓存B1不能冒充其他batch的原生前向。

Qwen四组已完成，每组558个不同TEST视频、前200成本；原PT/manifest/model文件SHA、
实际A100 UUID/250W上限、逐批功率/时延/预测及中途/终点PID审计保存。运行源码
main `4cd6fe86d2314fdf3bae1264808abc242f99466a`。无warmup，四组SRCC分别为
.7663428036522781/.7664148425389119/.7694958345192725/.7694089803486667，
与历史对应批量结果精确一致。8项CPU成本/防覆盖/无训练检查在两端通过。

最终Qwen B1仍使用上方历史总表锚；新B2/4/8的16视频时间分别为
751.1531734466553/554.0055650472641/504.5564776659012ms，能量分别为
289.25610600062987/229.76200380177073/227.74037765183186J。新B1复测
1150.4916036128998ms/442.9714733293941J仅审计，未进入最终B1数据。四份原报告与
预测/时延/功率CSV已备份至原比较报告的`raw_remote/frozen_batch_20261013`，
更新表和图仍在同目录`recalculated_20261013`。不是四点同日测量：B1保留历史，
B2/4/8为本轮同协议复测。原文件/旧sweep不动。

我方四个GPU进程及队列已退出；结束时各自审计只有自身PID。随后A100出现他人
进程，不称整卡当前空闲，不干扰。CLIP/YOLO/DeepSeek原生backbone已核身份，
但精确trained质量读出PT在限定本地/服务器原run、归档/保留目录中尚未定位，
这三组未启动，不能据此断言原PT已删除或用随机head/新训练替代。新入口依Git发布，
不传输源码补丁或建工程副本。

**后续人工授权更新（2026-10-13 19:57 +0800）：**再次查找原任务头／缓存未定位后，
用户授权按原协议在A100重训。CLIP、YOLO、DeepSeek三组head均已完成50轮及strict CPU
558样本复核；它们是新PT，不是找回原PT。
**2026-10-13 20:27 +0800：**CLIP/YOLO/DeepSeek共12组B1/2/4/8原生冻结复测已全部
完成，每组558个不同TEST视频、首200成本。逐样本SRCC、时延/功率原始CSV、PT/receipt/
源SHA、PID至UUID均核验；所有自身suite/评估PID已退出，A100已释放。60份batch文件、
12份log、suite_status共73份本地与服务器SHA一致，head best/last、history/CPU复核和
合并feature cache也已本地备份，分片parts仍保护在服务器。新B1 SRCC分别为
.7252558079932262/.7015970081958657/.5684669009566514，与旧参考差+.0000990/
+.0000148/-.0003500。全部标原协议重训、TEST选模开发/noVAL，不冒称旧PT找回。
比较表/图沿原`reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013`
更新；Qwen B1总表锚点不变。细节、SHA与备份范围见本页末尾“原协议重训”和比较报告。

**2026-10-04整理覆盖：**下文保留历史仿真、baseline和测速协议，不代表所有默认命令已适配。
正式Spatial/Temporal实拍版本、各自权重及已发布核心代码见
[当前版本与待收敛边界](CURRENT_VERSION_20261004.md)。主线核心通过57项CPU合同测试、
两份正式PT严格加载；旧Temporal-36默认profile和后续实拍微调入口仍须单独核验。

## 先选版本：当前实拍模型与历史默认不是同一套

| 当前登记版本 | 原固定权重仿真 SRCC | 未适配实拍 SRCC | 权重身份 |
| --- | ---: | ---: | --- |
| Spatial，单视频4帧，六层原1M读出 | .6710968960 | .5786364901 | 95e12397…592828b |
| Temporal，16视频×4帧，同场六层 | .8043868643 | .7977138739 | 5303b574…a243a77c |

精确完整SHA、源数据、原报告、离线适配与只读权重验收命令均从上方
`CURRENT_VERSION_20261004.md`进入。两行不是同一模型；旧测速不套到这两份PT。
以下Temporal-36、旧Spatial-4、9×4等段落保留历史成绩和对照。
**未指定profile的旧默认仍是Temporal-36，不是当前Temporal实拍模型，且预检未通过。**
本轮只澄清入口，不静默改默认profile、原SHA、资产路径或光学合同。

2026-10-06补齐主线测试依赖：`lab_stage_coordinator.desktop_code` 仅返回历史阶段脚本
字符串，不连接机器、不打开SDK、不启动采集；T03测试不再依赖未跟踪的
`lab_manual_stage.py`。同时恢复历史 `stage_config` / `effective_stage_identity` 纯配置
函数及原测试，既有 `lab_bench` 函数未改。来源见
[函数投影身份](stage_coordinator_source_identity_20261006.json)。这不是迁入完整旧控制器，
也不是宣称现有采集入口已经应用逐层曝光；未改变硬件合同或正式模型。
进一步检查主线483份任务Python的2294个内部导入名，发现T03采集入口还缺
`capture_staged`；现从同一历史提交原函数恢复，T03导入及4项采集前置拒绝测试通过。
这些测试不接设备、不写实际会话，原T06的 `capture` 函数未改；不是现场采集验收。


## 历史仿真结论（不替代上方当前实拍版本）

2026-10-09新增[未调制光場消融入口](reports/paper_results/temporal_dc_ablation_20261009/README.md)：
核实两份固定权重的DC比例，先同权重干预，再匹配续训。rho=0只移除未调制场，
不会移除光分支；不修改原正式PT，保存best/last，最多使用两张GPU。

用户随后要求改为[同一随机初始化从头训练的光学FFN残差对照](reports/paper_results/temporal_dc_ablation_scratch_20261009/README.md)：
只继承原Temporal架构JSON，不读取旧学生PT；一组训练/测试固定20%，另一组固定0。
两组均100epoch，记录实际初始参数SHA，只保存best/last。上一轮续训结果不冒称从头对照。

以下四份服务器原报告现已原样收录，便于查询必要历史对照，不将报告中当时的
“正式／当前”称谓作为今日最终版本：

- [连续相位、DC20旧对照](reports/paper_results/spatial_continuous_no_k_20260908/README.md)：s463，SRCC .624532，含原低层质量输入边界。
- [五档加权读出](reports/paper_results/spatial_weighted_readout_20260909/README.md)：s495，SRCC .631329。
- [感受野／校正尺度对照](reports/paper_results/spatial_rf_fusion_20260909/README.md)：s523，SRCC .632483；不是新网络旁路。
- [9月12日电子参数／压缩核对](reports/architecture/ELECTRONIC_PARAMETER_AUDIT_20260912.md)：历史Temporal rank256与Spatial压缩链，不替换本页顶部封存PT和复评指标。

四份与实际服务器保留原件逐LF SHA一致，来源记于`source_import_20261004.json`；
原数据、PT、旧测速和报告里的开发期TEST选模口径保留，未重训、重评或套用新测速。

### 旧 Spatial 候选说明入口

以下原说明按历史版本保存，其中“当前／正式”仅指当时，不替代本页顶部的
Spatial `.6710968960` 和对应封存权重：

- [严格双支路旧候选](reports/paper_results/spatial_single_video4_two_branch/README.md)：SRCC `.6251`。
- [EMA／8-bit 相位旧候选](reports/paper_results/spatial_single_video4_two_branch_ema_q8/README.md)：SRCC `.62561`。
- [s586／s581 历史优化对照](reports/paper_results/spatial_srcc_optimization_20260909/README.md)：SRCC `.633614`／`.633359`。

三份说明与实际服务器原字节一致；保留原配置、PT、结果和测速，不重评或套用旧测速。
其中旧说明的“失败 run 可以删除”不是本轮清理授权，仍须遵守根目录保护规则。
另一个 `.666503` 候选含冻结 ResNet18，属于被否决的上限对照；其旧profile、
结果说明及指定配置已在两端分别完整归档至`archive/rejected_candidates_20261008/t06_resnet_upper_bound`。
本地的拒绝标注与服务器旧“正式”措辞差异均保留，未删除PT、数据或测速；
恢复身份见仓库[旧目录归档收据](../../../maintenance/storage/LOCAL_LEGACY_LAB_PROJECTS_20261008.json)。
不按旧标题采用该候选，也不将旧profile继续当日常最终入口。
这些历史 TEST 选模成绩是开发指标，说明归主线不代表资产闭包或新机复现通过。

原1M读出候选的[仿真说明](reports/paper_results/spatial_readout_1m_srcc067/README.md)
及[同权重去光／最后相位随机对照](reports/paper_results/spatial_readout_1m_srcc067/LAST_PHASE_ABLATION.md)
现已原样收录。它们记录历史SRCC `.671008`，不是本页顶部固定实拍包复评的
`.6710968960`，不能相互替换。来源是本地保留报告及既有Git恢复blob，
本次限定检查未在服务器主目录或compress_all目录找到原件，不改称服务器运行新结果。
两份原始SHA依次为`10b0be9a96fbab912a6c66b2d5aaaa0398e557cc9f99c12b22d80f3fae673c77`
及`40de254f400efd0900b6be7282a0fceb42f0188a27a62b5bdec785cf0ab144be`；
原文件不改，未复算消融、训练或运行其中的旧输出命令。

旧独立审阅包在 `LightGenPublic/tasks/t06_lgvq_temporal_consistency`，不是另一个
日常开发工程。原 `teacher_release_final/lgvq_temporal_08044` 使用35个固定field、
558条视频及PT SHA `5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c`，
原仿真SRCC .8043868643。其 `teacher_release_final_v2` 改了8µm传播网格与探测器扰动，
记录 .8022806420，不能按同名PT/继承commit冒称与原版等价。二者仍保留完整原目录、
固定输入与权重，未做新性能评估；训练缓存缺项和重训练边界继续按原说明保留。
版本身份见 [审阅包审计](../../../maintenance/storage/T06_REVIEW_PACKAGE_IDENTITY_20261006.json)，
80份具名包内源码的Git恢复身份见
[历史源码副本](../../../maintenance/storage/T06_REVIEW_SOURCE_VISIBILITY_20261007.json)。
忽略副本不删除文件，不改变封存包的SHA清单；后续开发仍从本任务入口进行。

三个旧审阅包的同一份原依赖说明已集中到
[历史审阅包依赖](reference/temporal_review_requirements_20260922.txt)，原字节SHA为
`10300b8bf6c39ceeebb6345e46075a9090a588cb6305523eae426bd7ae4b9f9c`。
这只记录当时numpy/PyYAML/torch版本，不安装软件、不替换当前环境或正式模型合同；
三份原文件保留，退出Git待提交列表，不再维护重复依赖说明。

误命名的`tasks/06_video_quality_assessment`仅含14份MobileNet历史报告，已原字节归入
[MobileNet旧对照](reports/paper_results/spatial_mobilenetv2_b11_20260910/RESULT.md)。
其33级配置继承链缺失的五份旧配置已按原字段纳入兼容后端；
[配置来源和本地／服务器注释差异](reports/paper_results/spatial_mobilenetv2_b11_20260910/CONFIG_SOURCE.md)
单独说明。原报告、PT和测速不变，没有重训或复测，也不将此历史候选改为默认。
原报告与配置快照纳入main，其余原结果仍保留在该规范任务目录；14份原字节
均可从Git恢复提交`a96c1fc4d7bd6062002cb73955cbc0433229326c`的旧路径恢复。
旧目录已无报告文件，空目录未删除。
这是2026-09-10的SRCC .665159、去光 .561838、PT SHA 5e5020e3…55c4459历史对照，
不是当前Spatial .6710968960/PT 95e12397，也没有补测实拍或套用旧测速。
原s745配置继承及原字段加载已复核；数据/缓存/PT等资产闭包未完整复核，
原配置快照保留出处，不当作已通过新机训练或实拍验收的profile。

历史 Spatial-4 对照归档是 `spatial_single_video4_balanced`：一条视频均匀取 4 帧并排成
2×2，**没有多视频复用**。它使用两套物理光 Router Top-2、六次光传播和 20% 名义
未调制分量，在 558 条 test 视频上达到 SRCC 0.6393、KRCC 0.4642、PLCC 0.6743、
RMSE 8.452、MAE 6.646。入口、checkpoint SHA 和光关闭对照见
[`reports/paper_results/spatial_single_video4_balanced`](reports/paper_results/spatial_single_video4_balanced/README.md)。

历史 Temporal-36 对照版本为 `temporal36_balanced`：一个视频均匀取 36 帧，以 6×6 lane 放进同一个
478×478 有效光场。四专家光学 Top-2 router、六次光传播、20% 名义未调制直流分量、
鲁棒位移/相位/CCD 扰动和目标专属电子读出头保持不变。

正式测试集 558 个视频：SRCC 0.8454、KRCC 0.6394、PLCC 0.8650、RMSE 7.183、
MAE 5.451。平衡候选四专家占比正常，结果详见
[`reports/paper_results/temporal36_balanced`](reports/paper_results/temporal36_balanced/README.md)。

历史正式仿真候选 `temporal_multivideo9x4` 把 9 条互不相关的视频各取 4 帧，以 3×3 视频
tile、每 tile 内 2×2 帧的方式放入同一 478×478 光场。它不是把 Temporal-36 checkpoint
改名，而是新的六次全场相干传播模型，输出形状为 `[B,9]`；其中每个数仍只评价一条视频。
正式单 seed 结果为 SRCC 0.8082、KRCC 0.5999、PLCC 0.8131、RMSE 8.226、MAE 6.109；
视频路由最大专家占比 28.0%，跨样本选择变化率 55.7%，九槽位循环审计平均 SRCC 0.8019。
完整证据见
[`reports/paper_results/temporal_multivideo9x4_contentroute`](reports/paper_results/temporal_multivideo9x4_contentroute/README.md)。
Temporal-36 保留为“单视频 36 帧”的独立基线，二者不能混报。

16×4 吞吐优先模型 `temporal_multivideo16x4` 在同一光场同时处理 16 条视频、每条 4 帧，
输出 `[B,16]`。正式单 seed 结果为 SRCC 0.8044、KRCC 0.5968、PLCC 0.8180、
RMSE 7.991、MAE 5.992；没有达到预设 SRCC≥0.81，但一次光场输出数相对 9×4 增加
77.8%，且没有发生全局专家坍缩。正式配置和诚实的限制说明见
[`reports/paper_results/temporal_multivideo16x4`](reports/paper_results/temporal_multivideo16x4/README.md)。

论文“大模型 baseline”的 Spatial 行不是 Temporal 结果复用。它固定每视频 4 帧、
448×448 输入、冻结 `Qwen3-VL-2B-Instruct`，只训练 5×2048 个质量词输出行。正式配置为
`configs/baselines/qwen3vl_spatial_quality_tokens_4f_r448.yaml`，执行命令见仓库根目录
`RTX5090D_QWEN_BASELINE_PROTOCOL.md`；性能、单视频速度和板卡功率必须在同一 RTX 5090 D
dataset-once run 中产生。

## 不可静默改变的任务合同

- Spatial 与 Temporal 是两个独立单指标模型；当前 profile 只输出一个 Temporal MOS。
- Temporal prompt 必须存在；不能把文本输入从模型合同中删除。
- 冻结 Qwen 图像/文本前端负责生成缓存；学生推理图不得加入 Attention 或 Transformer。
- 当前 router 是四专家光学 Top-2，不是电子 router。
- 当前物理合同是 532 nm、17 µm 仿真像素、10 cm、518 canvas、中心 478 有效孔径。
- 当前正式鲁棒仿真包含至少 20% 名义未调制光功率；四个融合 alpha 受同尺度归一化约束。
- 当前 Temporal-36 表示一个视频的 36 帧。改成多视频必须新建 profile、checkpoint 和报告。
- MultiVideo-9×4 的 9 个标签和 9 个输出必须一一对应；不允许九视频聚合成一个分数。
- MultiVideo-9×4 必须整幅联合传播；逐视频传播后软件拼接只能作为容量上界，不能报告为硬件结果。
- MultiVideo-16×4 同理输出 16 个独立 MOS；64 帧只共享光传播，不共享标签或读出结果。
- 以上任一项变化都不能覆盖 `temporal36_balanced` 的名称或结果。

## 当前源码关系

为了不复制并分叉已经验证的核心模型，当前唯一后端仍是：

```text
experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54
```

本地和源服务器核心源码 SHA256 已核对一致。`LightGenV2` 现在接管唯一入口、新 runs、
正式报告和 releases。这个兼容关系集中写在一个 profile 中：

```text
configs/lightgen/temporal36_balanced.yaml
```

## 仿真操作顺序

以下命令都从 `2026OpticsMoE` 根目录执行。

2026-10-06预检修正：`check_environment --task t06` 仍检查历史Temporal-36，
但不再仅因Torch可用就返回成功。后端/config/PT身份及已列输入路径有问题时返回2，
错误以JSON报告；`--profile <旧兼容profile名称>` 可显式选择，不自动换成另一架构。
本机实查默认PT和缓存缺失，五份后端SHA不匹配，因此当前默认预检失败；
这不是宣布文件被删除，也不是允许更新SHA绕过检查。检查范围不含缓存语义或SDK。
正式Spatial/Temporal实拍版本仍从上方 `CURRENT_VERSION_20261004.md` 查精确入口，
不据旧默认预检或下方历史命令称它们已在本机可运行。

```powershell
# 1. 环境和文件检查
python -m LightGenV2.scripts.check_environment --task t06

# 2. 不加载数据的 CPU 冒烟测试
python -m LightGenV2.tasks.t06_video_quality_assessment --phase smoke

# 3. 正式训练前检查缓存、manifest 和初始化 checkpoint
python -m LightGenV2.tasks.t06_video_quality_assessment --phase preflight

# 4. 正式训练；新产物自动进入本任务 runs/simulation
python -m LightGenV2.tasks.t06_video_quality_assessment --phase train

# 5. 用正式平衡 checkpoint 评估
python -m LightGenV2.tasks.t06_video_quality_assessment --phase evaluate
```

上面未显式指定 `--profile` 时默认运行 Temporal-36。运行单视频 Spatial-4 时必须显式写：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment `
  --profile spatial_single_video4_balanced `
  --phase evaluate
```

指定 checkpoint 或输出位置时使用：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment `
  --phase evaluate `
  --checkpoint D:\path\model.pt `
  --run-dir D:\path\evaluation_run
```

每次入口都会保存 `resolved_launch.yaml`、`launch_identity.json`、`command.txt` 和
`run_manifest.json`。继承配置中的数据路径会先冻结为绝对路径，因此把输出迁移到
LightGenV2 后不会错误地相对到新 config 目录。

如果本机缓存位置与源服务器不同，复制根目录 `paths.example.yaml` 为
`paths.local.yaml`，仅填写 `t06` 下有差异的项；入口会在生成最终配置时应用这些
覆盖，不需要修改正式 profile。

## 硬件与交付

2026-10-08共享采集入口改为显式导入主线 `hardware_common/shs/slm_camera`，
不再从旧 `--bench-root` 注入并加载另一份控制源码。`--bench-root` 仍作为原机器
SDK相对路径基准及 `results/dual_jobs/ACTIVE.lock` 的互斥根，未移动锁、SDK或设备配置；
Controller显式传入该基准，逐层配置和原采集守卫不变。SHS打包源码清单同步包含
共享控制器及其驱动依赖。仅CPU结构与前置拒绝检查，不构建包、不接设备或重评模型。
现有旧交付ZIP保持原样，不能据此宣称旧SHS整目录已不再需要。
相位保持器同样改为主线显式导入，不再加载旧目录的guarded_workflow或phase脚本；
原UTF-8-sig配置读取、相位锁位置、SDK参数、重复显示和退出流程不变。
相位源码加入新打包清单，原包及历史源码身份仍保留；没有启动相位设备。

原服务器compress_all交付材料`hardware/reproduce_simulation.py`和
`hardware/AI_TEMPORAL08044_HANDOFF.md`已原样纳入main，逐LF SHA分别为
`a41d1cba72c1644931e8c8fea96852758a39966dc5d15b2dfb55fbeb0e7fd0b4`及
`6ba683249ed7aa0c1a81a026f91107cf4a40ee80313cd63a5b1921b5739de5b4`。
脚本是独立包的simulate入口模板，按自身目录找runtime、SHA256、release和固定输入；
main里的模板位置没有这些包内资产，不能直接运行完整评估。仅帮助检查通过，未读取
PT或558视频、不改变原数值容差；使用时必须对应完整原包及其清单，不混用原版/v2。
本轮保留原脚本的输出行为，不把它当新增训练／打包协议，原合同亦不证明另一光路已复现。

2026-10-07补齐既有SHS打包器引用的`lab_phase.py`和`hardware/run_lab.py`，
不再依赖本地未跟踪源码。两份服务器原Spatial/Temporal交付ZIP中的对应成员已只读核验；
portable入口原样保留，相位保持器保留后续已有的可选`--release-file`退出控制，
未指定时与原包行为相同。不会自动切层或采CCD，本次未打开设备或重建旧包。
来源和差异见[打包依赖身份](package_dependency_source_import_20261007.json)。
这仅关闭两份源码缺项，不代表旧Temporal默认资产或机器SDK已完整。

同日恢复正式SHS打包分派：显式使用 `--bench shs --target spatial|temporal
--source-root <原资产根目录> --output <新的输出目录>`，调用既有 `lab_bundle`；
不传 `--bench` 仍是旧legacy流程。SHS不接受 `--checkpoint` 静默替换封存权重，
已有输出目录或相邻ZIP在加载模型前拒绝。参数分派及输出保护测试不加载PT/SDK，
本次没有构建新包或覆盖旧包，不代表完整现场复现通过。

- 六阶段顺序：[`hardware/README.md`](hardware/README.md)
- 构建实验室完整包：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment.build_lab_package
```

该命令默认使用 SHA256 为
`159b1d8cd31aa5f817d274f2930129601d4f0a365f01c430a8fefcc5989c8730`
的 Temporal-36 平衡 checkpoint，并把 ZIP 写进 `releases/`。如果当前机器没有权重，
命令会明确报出缺失路径；可传 `--checkpoint`，或直接在源训练服务器构建。

## MultiVideo-9×4 训练

新图的任务内入口为：

```powershell
# 结构、梯度与禁止 Attention/Transformer 检查
python -m LightGenV2.tasks.t06_video_quality_assessment.multivideo `
  --config LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/temporal_multivideo9x4_formal.yaml `
  --phase smoke

# 缓存/manifest 检查
python -m LightGenV2.tasks.t06_video_quality_assessment.multivideo `
  --config LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/temporal_multivideo9x4_formal.yaml `
  --phase preflight

# 评估已经选定的正式权重
python -m LightGenV2.tasks.t06_video_quality_assessment.multivideo `
  --config LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/temporal_multivideo9x4_formal.yaml `
  --phase evaluate `
  --checkpoint LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/multivideo9x4_contentroute_d30_s114/best_checkpoint.pt
```

训练集每个 epoch 都会重新把 2,250 条视频随机分成 250 组，并随机交换组内九个物理
slot；测试集固定为 62 个物理场、558 条视频，指标仍对 558 个单视频预测计算。训练损失
同时包含 MOS 回归、排序/相关性、教师软标签、光电对齐、Top-2 专家均衡、保护带能量和
周期性 slot 置换一致性。相位调制保留 20%–35% 相干直流分量、k 空间限制和输入/相位/
CCD 位移扰动。配置目录只保留 `base` 与 `formal`；历史 sweep 参数已经写进正式报告，
不会再以一批失效 YAML 干扰后续操作。

六次传播的含义依次为：36 帧光 router、144 帧专家、9 个视频内帧融合、9 个视频
router、36 个视频专家、9 个视频 global。视频 router 只读取每视频 4 个已受 prompt
条件化的帧摘要；完整的 4 个图像 token 与 38 个文本 token 仍进入后两级专家/global，
避免公共 prompt 能量把不同视频的路由差异淹没。后端共享同一个无 Attention/Transformer 的
电子读出头，对九个视频分别调用，最终输出九个连续 Temporal MOS。

当前六张 mask 的实际覆盖率和 9×4 多视频设计见
[`reports/handoff/MASK_LAYOUT_AND_MULTIVIDEO9X4_PLAN.md`](reports/handoff/MASK_LAYOUT_AND_MULTIVIDEO9X4_PLAN.md)。

正式候选训练完成后，必须做 9 次循环换位审计。该审计把同一批视频依次放入九个物理槽位，
同时检查路由是否随样本变化、预测对槽位是否稳定，以及在**不重新训练**的条件下关闭光支路会下降多少：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment.multivideo_audit `
  --config LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/temporal_multivideo9x4_formal.yaml `
  --checkpoint LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/multivideo9x4_contentroute_d30_s114/best_checkpoint.pt
```

`slot_cycle_audit.json` 是正式比较依据；只报告九个槽位合并后的全局专家占比不够，因为不同槽位固定选择
不同专家也可能伪装成“均衡”。视频级 router 的 `selection_variation_fraction` 必须大于零，才能说明
Top-2 选择确实随视频内容改变。

正式损失中的路由 diversity 项最小化 `H(expert|sample)-H(expert)`：一方面让单条视频的
Top-2 选择明确，另一方面让同一物理槽位上的不同视频使用不同专家。固定选同一对专家、
或者对四个专家始终犹豫不决，都不会被误判为有效均衡。

最佳 checkpoint 的六次相位排布可按论文常用的 Arial 7 pt 生成两张 18 cm × 5 cm 图：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment.visualize_multivideo_masks `
  --config LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/<formal_profile>.yaml `
  --checkpoint LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/<run_id>/best_checkpoint.pt
```

黑色只表示该次传播中没有可训练相位的保护区，不代表实际 SLM 必须在该处吸收光。输出同时包含 PNG、
嵌入字体的 PDF 和每次传播的占用率/相位统计 JSON。

## MultiVideo-16×4：16 个视频并行

该 profile 在相同的 518 仿真 canvas、478×478 有效孔径、10 cm 距离和六次全场传播下，
同时评价 16 个互不相关的视频，每个视频固定取 4 帧。输出为 `[B,16]`，每个槽位仍对应
一个视频的连续 Temporal MOS，不会把 16 个标签聚合成一个结果。

精确排布为：4×4 个 115×115 视频 macro tile，起点为 `[3,122,241,360]`，相邻视频间
4 pixel；每个视频内部为 2×2 个 56×56 frame lane，相邻帧间 3 pixel。因此第一部分是
8×8、共 64 帧并行。帧专家为 27×27、内部间隔 2 pixel；视频级专家为 56×56、内部
间隔 3 pixel；global 相位为每视频 111×111。有效孔径尺寸和光路均未改变。

探索候选已经完成并清理。今后复现只使用唯一正式配置：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment.multivideo `
  --config LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/temporal_multivideo16x4_formal.yaml `
  --phase train
```

新正式训练默认只保留 `best_checkpoint.pt` 和 `last_checkpoint.pt`。每 5 epoch 仍测试并
更新同一个 best 文件，但不再生成周期相位 PT。只有独立命名的 mask-evolution 研究才可
把 `phase_snapshot_interval_epochs` 改为正数。

## 冻结 Qwen 方案二电子 baseline

`qwen3vl_quality_tokens_r448` 是纯电子 baseline 的正式 profile，不改变上述光电模型。
Qwen3-VL-2B-Instruct 全部冻结，只训练五个质量词 `Bad / Poor / Fair / Good / Excellent`
对应的 5×2048 输出行，共 10,240 个参数；五路 softmax 概率按训练集 MOS 范围内的五个
等距质量分数加权，得到一个 Temporal MOS。

每个采样帧固定为 **448×448**。当前 Qwen3-VL 视觉前端是 16×16 patch、2×2 空间 merger
和 temporal patch size 2，因此有效空间步长为 32；448 可被 32 整除。4/9/16 帧在视觉
block 0 前分别形成 1,568 / 3,920 / 6,272 个 1024 维 patch token；空间 merger 后分别
成为 392 / 980 / 1,568 个 2048 维视觉 token，再与 prompt 的文本 token 进入语言模型。
正式表中的输入始终是 448×448，不是 700 多像素。

4、9、16 帧的抽帧位置、65% 中心裁剪、Temporal prompt、训练/test 划分和随机定位解码
保持固定。每个帧数必须用对应特征训练自己的五个输出行。

正式计时是一组方案/帧数对应一个独立进程：模型只加载一次，不做显式 warmup，不做
test 前推理，第一条也进入 558 视频总体统计。主时间从 Vision Transformer block 0
输入到标量分数在 GPU 上就绪；模型加载和 MP4 解码/裁剪/resize/tokenizer/H2D 分栏保存。

```bash
CONFIG=LightGenV2/tasks/t06_video_quality_assessment/configs/baselines/qwen3vl_quality_tokens_r448.yaml
MODEL=/absolute/path/Qwen3-VL-2B-Instruct
MANIFEST=/absolute/path/lgvq_split.csv

# 检查分辨率、帧语义、模型、manifest、GPU 和 Git 身份
python -m LightGenV2.tasks.t06_video_quality_assessment.quality_token_resolution \
  --config "$CONFIG" --phase preflight --model "$MODEL" --manifest "$MANIFEST"

# 可断点续跑：依次提取 4/9/16 帧特征、训练 50 epoch、各自完整评估 558 个 test
python -m LightGenV2.tasks.t06_video_quality_assessment.quality_token_resolution \
  --config "$CONFIG" --phase all --model "$MODEL" --manifest "$MANIFEST"
```

如果中途终止，可把 `--phase all` 换为 `extract`、`train` 或 `benchmark`，并用
`--frames 4|9|16` 只续跑一组。正式结果进入 T06 自己的 `runs/simulation/<run_id>`；
论文图和紧凑结论进入 `reports/paper_results/`，大特征、checkpoint 和逐视频 CSV 不提交 Git。

当前 5090D 的 448×448 正式结果、完整指标表、token 几何、计时边界和论文图见
[`reports/paper_results/qwen3vl_quality_token_baseline_r448`](reports/paper_results/qwen3vl_quality_token_baseline_r448/README.md)。

## 历史 A100 批量与 dataset-once baseline 已收敛（2026-10-06）

### 可选历史 Temporal 读出压缩工具（2026-10-07）

旧服务器`t06_readout_compress_20260912` h512试验目录已完整归档退出日常工作目录：
3577份原文件、13处数据链接身份及全部旧测速保留，四份早期冒烟元数据亦在完整归档中。
原数据／PT链接目标不删，正式Spatial/Temporal及另一个compress_all原结果目录未变。
恢复路径、完整SHA及原服务器Git引用见仓库
[目录恢复收据](../../../maintenance/storage/SERVER_WORKTREE_BALANCE_20261007.json)
的`latest_t06_readout_trial_retirement_20261007`；不是新的模型成绩或本地独立数据备份。

`compress_temporal_readout` 与三份原 h256/h512/h640 配置已从本地保护历史
`ac7e3ecc` 纳入主线，保留独有代码，不把它当作服务器正式运行版本。
核验时训练服务器主目录没有这四份文件，也没有该原历史对象；来源明确记录在
[历史工具身份](temporal_compression_import_20261007.json)。只用于16视频×4帧六层
光电之后的原电子读出头缩减，不能把其配置名或训练缓存成绩替代封存 .8044 PT。
本轮未找到并绑定压缩候选正式成绩，未提取缓存、训练、重评视频或使用GPU/设备。

原九项函数的AST保持一致，仅 extract/train 的首行增加已有输出拒绝；缓存文件、
伴随JSON或训练目录存在时，在读取配置/PT、初始化RNG及CUDA之前失败，保护原结果。
三份配置继承既有formal，原256/512/640宽度及种子不改，默认为可选历史profile，
不修改主入口默认Temporal-36或最终16×4模型。原工具按TEST SRCC选模，是开发指标，
只保存压缩best；它不是新的已验收正式训练协议，不补造历史last或独立泛化结论。
入口帮助：`python -m LightGenV2.tasks.t06_video_quality_assessment.compress_temporal_readout --help`。
9项合成CPU/AST/配置/防覆盖检查通过，不等于真实缓存或完整模型复现。
实际训练服务器main `3a130341` 同样通过这9项检查，CUDA设备隐藏，未读取正式PT/数据
或调用设备；来源收据保留原本地历史身份，不因发布后存在而改称此前服务器运行版。

同一保护历史的八份`temporal_multivideo16x4_readout_rank{R}_s{S}.yaml`亦归主线：
R/S依次为512/173、384/174、320/175、256/176、192/177、128/178、64/179、48/180。
它们继承同一formal并仅声明原末端读出矩阵低秩，供既有压缩工具显式选择；
不加电子支路、不改变六层上游，不修改正式默认或 .8044 PT。服务器此前没有这八份
文件，来源是本地独有历史，不冒称已在服务器验证过训练成绩。原始配置SHA见上方
历史工具身份，八项配置加载/几何/原SHA检查只证明配置合同，未跑真实数据或新性能。
rank48原输出名为`multivideo16x4_readout_rank48_kd_s180`，不为了统一文件名改写原run身份；
名字中的`kd`本身不证明某份历史训练已经完成或具有什么指标。

2026-10-07补回旧A100综合测速导出缺失的15份baseline源码／配置快照，逐文件匹配
其原SHA（含混合CRLF行尾），没有替换现用模型、重测或更新旧SHA清单。
该导出仍有9项文档／渲染图缺失或检查文件变化，不称完整；原README也保持不变。
具体恢复身份见 [测速源码恢复](../../../maintenance/storage/DEMO_TIMING_SOURCE_RESTORATION_20261007.json)。

历史 Spatial s586 报告的28级配置继承链亦已核对，14份此前仅在工作目录／Git归档中的
必要配置纳入主线，内容不变。来源见 [配置闭包](spatial_historical_config_import_20261006.json)。
该候选是既有历史对照，TEST选模属开发指标；不是上方当前实拍Spatial权重，不用旧代码
覆盖后续模型，也没有重新训练／评估。配置加载与逐文件SHA测试不等于完整PT/资产复现。

实际 `2026OpticsMoE_t06_a100_batch` 的两份未提交运行源码现已归入本任务，
不再需要到旁边的独立工程找测速脚本：

- `quality_token_batch_benchmark`：每批不同视频、每视频4帧、448×448；`sweep` 与
  `formal` 分开，558条正式TEST的首批计入，没有正式 warmup。主计时为全部输入已在
  GPU 上、完整Qwen前向至质量分数；Vision block0计时仅为并列的次要诊断。
- `quality_token_dataset_once`：4/9/16帧单视频的既有方案1/2入口，恢复原服务器
  schema 2（同步墙钟为主、CUDA-event并列）。原 schema 1 报告保留，不转换或重写，
  两种时钟与不同批量不能混作一个速度数字。

两入口恢复 GPU 型号与实际功率上限校验，显式使用 CUDA-visible 的物理编号／UUID。
批量入口默认要求 A100，dataset-once 默认要求 RTX5090D；跨GPU必须显式指定
`--expected-gpu`，不以绕过校验或套用575W来生成 A100 报告。新增输出保护会在
查询 GPU 前拒绝已有目标目录，旧结果不覆盖。入口参数可用相应模块的 `--help` 查看。

本次没有加载Qwen、PT或视频，没有测速、训练或触碰GPU。12项无Torch/设备的 CPU
依赖、身份、预处理审计及旧输出保护测试通过；13项批量性能函数、5项dataset-once
非入口定义与封存源码AST相同，已有共享函数和计时器默认行为另核验。
来源、投影范围和SHA见 [历史baseline迁入记录](batch_baseline_source_import_20261006.json)。
它不代表另一机器已完成实际功率/吞吐复测，也不能拿这些历史计时套给正式实拍模型。

## RTX 5090 D 光学 MoE 分段计时

T01–T04 与 T06 的 CCD 后串行电子处理、并行残差、跨层 SLM 场重建、bridge 和任务头已经
在同一块 RTX 5090 D 上按计算图边界正式测量。每个分量 50 次预热、1000 次正式同步调用；
T06 每次调用是一幅同时承载 16 个视频×4帧的物理场。任务级临界路径、能量口径、冻结
Qwen 对照和论文图统一见
[`reports/5090d_moe_and_qwen_baselines_20260907`](reports/5090d_moe_and_qwen_baselines_20260907/README.md)。

该报告明确区分物理光场时间、可被光路覆盖的并行残差、必须串行的 CCD 后处理和任务尾部。
其中 80.388 W 只能计算光学设备能量代理；在没有同步采集电子处理 GPU 功率前，不把它写成
完整光电系统能耗。

## 冻结 CLIP／YOLO 视频质量历史 baseline

旧 CLIP ViT-B/32 与 YOLO11s 的四帧 Temporal／Spatial baseline 运行入口已从其原
运行提交 `95904ecf97bcefe4a7ae76a6eab08a1d7cb217f0` 原样恢复到本任务。
命令、冻结参数与数据合同见 [VISUAL_BACKBONE_BASELINES.md](VISUAL_BACKBONE_BASELINES.md)。
历史结果、选模口径、原PT/缓存SHA和抽取耗时见
[CLIP／YOLO历史记录](reports/reproduction/VISUAL_BACKBONE_LGVQ_20260922.md)；
原中转说明仍保留，不把其2026-09-22成绩或耗时套给当前光电模型。
两模型只拟合五个无偏置质量输出行，共10240参数；按原无VAL划分的TEST SRCC选模，
属于开发指标，不称独立泛化。本轮未重训、未读取视频、未重评性能或测速。
原报告和本地中转源码仍原位保留，来源及CPU合同核验见
[恢复记录](../../../maintenance/storage/T06_VISUAL_BASELINE_IMPORT_20261005.md)。

## 9月27日历史 A100 多基线汇总源码

原 [LGVQ temporal multi-baseline A100说明](../../reports/20260927_lgvq_temporal_multi_baseline_a100/README.md)
及同目录 `build_analysis.py` 已按实际服务器原字节纳入main。
脚本SHA256为 `418cb591c3a7703ac0eb12d40d22271c330e3fdef77488e81954b443c4ddfa99`，
恢复来源为 `refs/archive/server-untracked-safe-source-20261006`；原测量、CSV、表格及功率记录仍保留原位。

这是历史200视频、无预热、同步墙钟、A100板卡加338.2W主机代理的对照；
Ours点沿用当时已审计的16视频记录，不是新PT或当前硬件的重新测速。
脚本是原始产物重建器，导入即读取 `raw_remote` 并写汇总/图/校验表，不是可安全导入的库；
本次只做源码身份/语法检查，未执行重建、未改旧报告或测量、未启动GPU。
如需重建，应先保存原目录并在独立输出副本操作，不能覆盖原测量证据。

## 缺失 Temporal baseline 资产的原协议重训（2026-10-13）

用户授权再次寻找 CLIP／YOLO／DeepSeek 原任务头与缓存，确实找不到时在 A100 上按原规则
重训。已只读核查当前注册 worktree 的 T06 run、main/A100 formal/demo/baseline 目录、
三份相关完整 tar 成员及存量移动收据，仍未定位三份原 head PT 或对应特征缓存；这不是
永久丢失证明。原 manifest 和三个预训练骨干仍存在且 SHA 匹配，历史成绩与证据不删。

`reproduce_temporal_baseline` 仅重提取冻结特征并训练 Temporal 五行质量头，不训练整个
骨干或 Spatial，不改划分、架构或搜索种子。三者沿用 seed42、50 epoch、batch512、
AdamW lr0.001/wd0、cosine、交叉熵；2250 TRAIN/558 TEST，四帧与原中心裁剪不变。
CLIP/YOLO 为10240参数，DeepSeek为6400参数；FP16缓存再转FP32训练保持原合同。
最高TEST SRCC选模仍是开发指标，不称独立泛化。DeepSeek源码／配置从保留的
`refs/archive/local-inactive-20261004/codex/deepseek-lgvq-baseline` 恢复，不创建新工程。

入口先拒绝已有输出，再验证原资产/配置/版本及唯一A100 UUID和foreign PID；串行suite有
任务锁，提取中定期核查占用。新产物进入现有 `runs/simulation`，保存best/last、50轮
history、558预测、strict CPU复核、协议与全部SHA。新PT明确标为`retrained_original_protocol`，
不能冒充原始 PT；旧 SRCC 0.7251568/0.7015822/0.5688169 仅作比较参考，不强凑相同数字。

```bash
CUDA_VISIBLE_DEVICES=GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55 \
  /DATA/DATA1/guest3/.venvs/lgvq-visual-baselines/bin/python -u \
  -m LightGenV2.tasks.t06_video_quality_assessment.reproduce_temporal_baseline --suite \
  --manifest /DATA/DATA1/guest3/2026OpticsMoE/experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv \
  --output LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_temporal_reproduction_a100_20261013
```

suite先串行训练三组，再逐模型逐B1/2/4/8启动fresh-process冻结推理。新头由完整
`reproduction_receipt.json`绑定checkpoint/训练报告/history/CPU验证SHA；原历史SHA常量不改。
每批精度对应完整558个TEST，成本仍用首200视频；最终时延仅推理边界同步墙钟，能耗为
`(338.2W主机代理+实测active A100 W)×推理秒`。解码/预处理/H2D记录但不计最终成本，
不混入旧Qwen B1总表锚点，也不以旧12行固定精度冒充新batch实测。

2026-10-13 19:15 +0800 已实际启动本轮suite：源码发布／服务器快进到
`7e2b170d857a92a9dd7d902a6c63d9379a46565f`，17项CPU合同测试两端通过。
suite PID2529940、首组CLIP PID2529959对应上述唯一A100 UUID；实际提取日志已从
75增长到275/2808，四个64行shard（256行）已落盘，尚未到head训练成绩阶段。
后续顺序是CLIP→YOLO→DeepSeek原协议head训练，再三组各B1/2/4/8冻结复评；
`suite_status.json`、各组`status.json`、`*_training.log`保存实际阶段与错误，不能仅凭PID
称任务完成。DeepSeek既有隔离环境2.6.0+cu124／4.38.2／0.0.29.post2、正式config与
单shard权重SHA已预先复核，未加载GPU或升级依赖。该段是启动记录，不是新模型结果。

### 第1次定时核查：三组head已完成（2026-10-13 19:57 +0800）

| Frozen backbone | 新PT best epoch | 新CPU TEST SRCC | 历史参考SRCC | 差值 |
| --- | ---: | ---: | ---: | ---: |
| CLIP ViT-B/32 | 48 | 0.7252558079932262 | 0.7251568 | +0.0000990080 |
| YOLO11s | 49 | 0.7015970081958657 | 0.7015822 | +0.0000148082 |
| DeepSeek-VL2-Tiny | 43 | 0.5684669009566514 | 0.5688169 | −0.0003499990 |

均为2250 TRAIN／558 TEST、50轮原协议的TEST选模开发指标，不称独立泛化。
三组各自best/last及CPU558预测完整；CPU/GPU best预测最大绝对差分别为
1.14441e−5、1.14441e−5、1.18256e−4，骨干文件前后SHA一致，只有原五行head被训练。
全部服务器receipt产物SHA已核查（CLIP54、YOLO54、DeepSeek98项），50轮history及
样本身份通过。三份实际启动Git为7e2b170d、bc34aa70、db48a68f；训练入口SHA一致
`a46abaf907d460260b1d3b04b6d852f7c02a9236c146f5a1e1bbaab8ae919e54`，
任务训练模块分别与各自Git对象及当前源逐字节匹配，不因其他任务／文档提交称算法变化。

新best PT SHA（last完整SHA仍在各自receipt）为：

- CLIP：`28729f41b09c59866ed54e0d520c48edca276ac0237d0b6bd885c8a0db4619d4`。
- YOLO：`47606708c66b313174d3a04edef6ebac1b06d3310d2f1049d7a989dbc386b848`。
- DeepSeek：`3748b08f9b4c7b61cd18425f49f899ea34598e3169846119ae75272bccdb7951`。

本地同名canonical run已备份每组10项核心产物并逐SHA匹配：best/last、完整FP16缓存及
报告、50轮history、558训练评估预测、558 CPU预测、CPU复核、训练报告和run identity；
另保留各receipt/status和完整训练日志。提取分片仍完整留在服务器，未重复下载或删除，
不能把本地核心备份称为服务器全部分片的第二份副本。

12组batch复测当前完成2组（CLIP B1/B2），各full558 TEST、前200成本、相同new PT。
B1 SRCC0.7252558079932262与缓存CPU结果一致，B2为0.7252452404480152；各五份原报告／
预测／时延／功率／状态已本地SHA备份，源字段明确retrained_original_protocol。
此时suite2529940仍运行，A100上B4 PID2741446对应唯一UUID，没有重启／新增训练。
最终汇总表／图待12组实测完成后统一更新，不拿这两组覆盖其余历史点；Qwen B1锚保留。
监督`lgvq-a100`每20分钟、最多24次，计数与增量仅记同run的monitor_state.json。
