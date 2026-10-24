# A100 nine-metric timing and energy table — 2026-09-28

## 新旧版入口与完成清单（2026-10-24 20:35，北京时间）

20:35仅推进离线准备，没有新GPU计时，也未重启待人工恢复的T08批次。
身份常量修正`8e7c5ed7f`已完成服务器干净快进，28项CPU测试通过、旧控制器已退出。
随后`operation_ledger.py`准备独立不计时的实际形状MAC台账；34项纯CPU合同两端通过，
另用真实CPU Linear、因果SDPA及Conv2d核对插桩前后逐位相同，MAC各24/144/1350，
确认fused attention不重复计两次矩阵乘法。仅这些小张量测试，不是任务200样本回放。
该台账同时保留实际执行总MAC、按批准计入的串行MAC、并行残差、非MAC、未知算子，
数字FFT与物理光等效OP分开；不会把不支持的算子当作0或把容量当任务算量。
私有`operation_ledger_cpu_contract_01.json`已逐SHA取回；全任务TOP/TOPS/TOPS/W仍待补。

LGVQ只读资产查找有增量：9月10日原完整ZIP内保留空间prompt缓存，其语言张量
与采用的9月14日`.6710`正式release首场逐位一致（`[1,38,2048]`FP16）。
原缓存记录Instruct身份及前端/embedding张量指纹，不再凭任务名猜模型；现有快照的
原参数指纹和真实视频前端仍需核验，不能把这份离线缓存当在线计时输入。
同包Conv5缓存元数据还指向确切的原Stem源PT，已找到原文件并重新核SHA
`d357fe51b888ecace74c050096febebc09c08abc21d6b42f533ecc3cf1f4de55`，
其中`frame_stem.`参数339,312；尚未执行新的质量前端数值合同。
原prompt数据和结构化收据已按SHA本地备份到T06
`runs/smoke/lgvq_frontend_identity_recovery_20261024_01/`。
Temporal前端绑定、两项Ours正式测量及全部任务算量没有因此标完成，九行仍8/18合格。

20:09 T08文搜图新适配的CPU身份预检拦截：`a100_title_input_inclusive_20261024_01`
控制器1142505在CPU原始前向之前停止，GPU正式调用为0、MPS未启动，三轮均未执行。
原因是新`abo_query_contract.py`抄写权重SHA时多了一个`2`（65位，非合法SHA256），
不是原模型发生改变。独立重新读取实际4,255,140,312字节权重所得64位SHA为
`c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1`，
与原`.82`完整性能报告完全相同。校验常量已按原报告纠正，新增64位hex前检，
28项CPU单元测试通过；真实标题的原/新神经前向逐位合同仍未执行，不能称该合同通过。
没有改模型、候选库、提示词、精度或正式计时范围。
失败run四份日志/状态/原命令已逐SHA本地备份，收据为该run内
`backup_receipt_cpu_rejection01.json`；私有诊断为
`measurement_protocol_20261024/abo_text_qwen_preflight_rejection_20261024_2009.json`。
按“新失败停止下游，不自动重跑”的约定，本轮没有覆盖旧run或重新启动该批；
等待人工恢复指示后才使用新run继续。此前已合格的8/18项和多baseline24/24均不受影响。
20:08复核A100无计算进程；不把入场空闲快照当作未来测量的独占保证。

**本轮唯一协议/结果入口是`measurement_protocol_20261024/`。两份正式新Excel尚未
生成，不要发送下方2026-09-28或`frontend_tokens_20261023/`里的历史工作簿作为新结果。**
`measured_index_partial_02.json`和`temporal_matrix_partial_03.json`是当前可审计中间索引，
名字和内部状态均明确PARTIAL；时间推进不自动将其升级为正式表。
新九行partial02已实际独立重放并纳入空间Qwen和LSP双方，不覆盖旧partial01。
LSP六轮均已完成独立CPU复算：原joints.mat及200张原图SHA、原人体框/224裁剪、
原始NPZ与CSV坐标、PCK分子/分母、源Git、逐段无重叠/merger/三传播次数、
进程/MPS及完整窗口功率支持均核验；未重复执行模型推理。19项CPU合同测试通过。

九指标表中的“完成”只指输入包含200×3的时间/能量已独立审计，尚不表示TOP完整：

| 行 | 指标任务 | LightGen | Qwen |
|---|---|---|---|
| 01a | LGVQ时间质量 | 真实输入前端与NN分段适配/正式测量待补 | 三轮已审计，共用两表唯一锚 |
| 01b | LGVQ空间质量 | 真实输入前端与NN分段适配/正式测量待补 | 三轮完成、本地独立审计通过 |
| 02 | ABO图搜文 | 原模型计时适配/正式测量待补 | 原模型计时适配/正式测量待补 |
| 03 | ABO图搜图 | 三轮已审计 | 三轮已审计 |
| 04 | ABO文搜图 | 原模型计时适配/正式测量待补 | 原模型计时适配/正式测量待补 |
| 05 | LSP | 三轮完成、本地独立审计通过 | 三轮完成、本地独立审计通过 |
| 06 | SALICON | 三轮已审计 | 三轮已审计 |
| 07 | OpenMoji新布局 | 原模型计时适配/正式测量待补 | 原模型计时适配/正式测量待补 |
| 08 | ABO文生图 | 原模型计时适配/正式测量待补 | 原模型计时适配/正式测量待补 |

这相当于18个模型—任务项中8项已有合格三轮时间/能量，其余十项仍待适配/正式测量。
匹配双方完成的任务是图搜图、LSP和SALICON；LGVQ时间/空间目前仅Qwen完成。
全部TOP及其派生TOPS/TOPS/W仍需匹配真实输入形状的独立不计时回放；共同输入
Tokens/s分子已经在已完成Qwen行保留，最终整表与时间倍率一起校验。

LGVQ多baseline表：Qwen/GLM/CLIP/YOLO四模型、MPS100/50各三轮，当前40GB范围
24点全部完成并通过本地独立CPU复算。`temporal_matrix_partial_03.json`固定原路径重放
全部24轮，原partial01/02不覆盖；共享Qwen100仍是原先合格的同一三轮，没有重测。
LightGen时间质量的共同点仍待补；完整DS/Kimi
各两档三轮共12点待人工提供新的80GB机器，现阶段不提前租赁或连接旧实例。

原始数据均留原任务run，服务器统一前缀`/DATA/DATA1/guest3/2026OpticsMoE/`；
已验收原始文件在本地相同仓库相对路径按SHA备份：

- 时间质量baseline：`tasks/t06_video_quality_assessment/runs/simulation/lgvq_mps_e2e_a100_20261024/`
  内固定引用原`qwen100_repaired01`、`qwen100_remaining01`、`lab_remaining01`和续测`lab_remaining02`，
  不通过目录日期或快慢挑轮；原拒绝/中断记录完整保留，不入正式均值。
- 图搜图：`tasks/t07_abo_image_retrieval/runs/simulation/a100_input_inclusive_20261024_01/`
  和`a100_input_inclusive_20261024_qwen01/`。
- SALICON：`tasks/t03_saliency/runs/simulation/a100_input_inclusive_20261024_01/`。
- LSP完成已复算：`tasks/t02_keypoint_detection/runs/simulation/a100_input_inclusive_20261024_02/`；
  `_01`是中断依赖，不能误用。
- 空间Qwen完成：`tasks/t06_video_quality_assessment/runs/simulation/lgvq_spatial_input_a100_20261024_01/`。

本次SSH故障定位到本机默认TUN路径：同端点经默认路径在服务端SSH标识前断开，
绑定物理WLAN后握手及原known_hosts指纹校验通过，并已恢复认证读取。只调整本聊天
私有连接工具的socket来源，不改系统VPN/路由、服务器sshd或防火墙；未重启任何测量。
原三个控制器在断线期间按原串行合同全部完成并退出。T06续测211文件、LSP95文件、
空间Qwen45文件已逐SHA备份；各run的`backup_receipt_recovered_complete01.json`
与私有`ssh_connection_recovery_20261024_1914.json`保留恢复/传输证据。
恢复时A100另有hmh进程1044253，起始19:08:33、1968MiB；不启动新测或终止对方。
已通过审计的原测量窗口占用证据独立核验，不因当前外部进程重新跑已完成轮。

LSP新独立审计为`measurement_protocol_20261024/lsp_audit_01.json`，SHA
`588d6d2cacc01133b1ef6c1a8a61ab26f38ccc10782a771211229e2d7a3ea985`。
新首200子集PCK分别`.7264285714/.7353571429`，不冒称原TEST1000的`.7348/.7226`。
原人体标注框裁剪任务不变，不是全图多人检测。新输入包含计时Ours21.865820ms、
Qwen36.475580ms/图；Ours光3.134100ms、计入电子18.731720ms，分阶段估算
2.333971J；Qwen全窗估算14.124261J、时间加权板卡49.025122W。
这些是三轮全部600调用聚合，与历史窄口径4.770/17.463ms不可互相替代。
每个神经网络段/原输入处理、排除layout/并行残差及Event配对仍保存在原stages和ledger。

T08下一批适配先通过`abo_query_contract.py`固定原CSV/两方向/200调用顺序，23项
纯CPU合同检查通过。已找到指定`.8058`主体的两份归档PT，均为`a2aa9a93…`，
不是main里另一份`.7983`。文搜图`.82`原基线的提示词是
`Retrieve product images that match the following product description.`，不同于图搜文
及标题候选的提示词，不能直接套T07或反向主体模板。原2048维FP32 gallery和报告已
逐SHA备份；图搜文共享目录的标题PT为FP16保存件，不证明等价于原在线FP32候选库。
候选库必须按原精度单独核验/准备，不能为省事改变描述子精度。以上仅适配准备，
没有增加GPU正式测量或将T08行升级完成。

以上`tasks/`路径均以`LightGenV2/`为基准。正式交付时会在本节列出两份已生成Excel
的准确文件名、SHA和全部raw清单；当前不提供尚不存在的正式下载链接。

## 2026-10-24 两表统一输入口径：已批准，重新测量

用户已确认空闲并重新授权续测；当前`lab_remaining02`只计划余16点，合格Qwen100
三轮、Qwen50三轮及GLM100前两轮固定复用。续测控制器先独立审核已完成raw及源
身份，再执行缺失点，新增守卫只覆盖加载空档，不改原计时窗或模型计算。
LSP使用新的任务run排在续测成功后，原中断记录和零样本目录全部保留。

GLM100缺失的第三轮已完成并通过服务器与本地独立CPU复算；15原始文件逐SHA备份，
复用原前两轮，不新增选择性重复。队列继续GLM50，再CLIP/YOLO两档。
空间质量Qwen的新适配单独复用原`quality_token_batch_benchmark_200`的prepare与
完整forward，仅使用原spatial prompt及原9155ec7a…评分头，保留完整词表投影。
原558视频参考是`.6907725734`，旧`.7232404106`是200子集，不可互换；
新的空间baseline三轮必须等LSP成功完成且重新独占后才开始，尚未执行不填旧时间。

空间Qwen控制器已实际排入`T06/runs/simulation/lgvq_spatial_input_a100_20261024_01`，
等待LSP的`a100_input_inclusive_20261024_02`成功完成。只读前检发现服务器缺少原558
参考报告，已从本地保存原件按9858954d…SHA恢复这一份数据，不改源码/权重或覆盖文件；
随后两端14项CPU合同通过，未因此多跑GPU样本。原始恢复收据私有保留。
`audit_temporal_matrix.py`固定24点原始路径，独立复算后只给完整三轮聚合值；新私有
`temporal_matrix_partial_01.json`已核Qwen100/50与GLM100各三轮，不是最终论文表。
两个表的Qwen100锚共用同一三轮，不能用历史窄口径覆盖；其余点继续留待测状态。

16:20测量暂停：T06 `lab_remaining01`新增Qwen50三轮和GLM100前两轮已完成并独立
CPU审计通过；原75文件逐SHA本地备份。GLM100第三轮在loading/零计时样本时出现
外部guest1进程，已只停止自有worker并释放MPS，LSP后续队列未开始推理即停止。
外部进程随后自行退出，仍等待独占窗口确认再补余16点，不重复合格轮，不把零样本
中断当有效测量。原始占用回执和停止审计均在T06该run中；当前整表仍未完成。

两表现在由“测量模型推理时间”**一个窗口**统一负责源码、独占排程、原始数据审计和汇总；
“保住光路性能-图五GPU利用率”已按人工要求停止新测量并交接已有run。
共享的Qwen temporal MPS100三轮固定为T06 canonical下
`qwen100_repaired01/qwen_cap100_repeat01`及
`qwen100_remaining01/qwen_cap100_repeat02,03`，均已双端CPU审计通过。
`qwen100_shared`的采样缺口轮、`qwen100_repaired01`被外部T19占用而失败的第二轮
只作拒绝证据保留，不参与任何正式平均。

机器可读批准协议为 `measurement_protocol_20261024/protocol.json`。用户已批准开始：
连续200调用、三次独立进程、不做专门预热、保留首条，基线文件输入至CPU结果；
九行表共用T06的Qwen temporal MPS100原始run，不各自生成不一致的同名数字。
LightGen保留16视频整场相干物理并行，逐任务保存输入处理与神经网络段的同步墙钟及
CUDA Event配对记录、源码/权重/输入身份；布局、数值FFT与并行残差分别保留但不计入
所选神经网络子总和。物理光时间是参数推算，不能把混合硬件投影冒称新实测装置端到端。
能量仍按用户确认的光/电两个作用阶段计算，不做空载扣减或全程通电积分。
文搜图100个唯一查询每轮完整执行两遍，200次计时、100个唯一查询评价。
当前新增入口/CPU合同不等于完成200×3正式测量；下方历史成本不会自动升级为新结果。
旧的“待确认”Word仅作历史草稿，不作为当前批准协议或测量结果。

首批执行证据：T07指定`.8225`的三次新测量在原任务
`runs/simulation/a100_input_inclusive_20261024_01/`，源码`d1b59e2ac`。
每轮200不同查询、fresh进程、首条保留，六次物理传播参数及32类原函数计时身份保留；
真实图片CPU插桩前后逐位一致。`measurement_protocol_20261024/audit_t07.py`
独立核验源Git、600调用、输入SHA/顺序、所有原文件SHA、无双计区间、merger/传播
出现次数与光电分段能量。审计JSON在本目录私有产物中，原始run已逐SHA本地备份。
这只是已完成的LightGen图搜图输入包含版，不是九任务整表完成；对应Qwen新测及
其他任务仍须逐项验收，TOP和统一Tokens/s未完整匹配时留空，不沿用旧计数。
T07对应Qwen同200查询的三轮也已完成并独立复算通过：输入有效token每轮13784，
统一Tokens/s分子两模型必须共用此数，不能使用Ours原生77×200替换。
SALICON指定`.8625/.8748`原模型已通过真实图片CPU合同并完成各三轮；原始run在T03
`runs/simulation/a100_input_inclusive_20261024_01/`，仍须独立数据/分段审计后才进入新表。
上述进展不表示九行均已重测完成，历史工作簿不得更名为新端到端正式结果。

SALICON新增独立`audit_salicon.py`已在本地对600+600调用复算通过：逐原文件SHA、
真实预测密度图、200张原目标图、float64 CC、输入224/196patch、源Git、无外部进程、
merger/三次传播、互不重叠NN段和全窗NVML积分均核验。原200目标图的SHA收据与
六轮完整raw已本地备份，未改写原run。新的输入包含值不能与历史窄口径毫秒混用。
T06余下21点现由`lab_remaining01`实际串行执行；只有完成并通过审计的点才纳入表格。
LSP指定495b9c2c…的原图/原人体框前端与原神经网络计时适配已准备，
`run_lsp.py`仅在前队列成功结束、真实图片CPU逐位合同通过并重新独占入场后执行
双方各三轮；不在另一计时窗口同时加载CPU大模型。尚未执行的点不填历史成本。

## 历史参考表格：2026-10-24统一输入Tokens/s（未按新协议重测）

用户指定双方采用同样的输入token计数。当前应使用
`frontend_tokens_20261023/九任务_统一输入Tokens_20261024.xlsx`。
它保留“核心口径历史参考”和“含GPU前端待补测”两个版本；23日原生token吞吐表保留作历史记录，
不再作为统一输入吞吐的展示表。本次没有重新测速，也没有改模型的输入tensor或网络结构。

同一任务的两模型采用同一份参考输入计数：优先取该批Qwen原始记录的完整LLM序列长度；
未保存LLM序列长度时，明确标为“Qwen视觉patch输入参考”，不将它冒称语言token。
变长输入按全部200条累计后折算同一工作量，不取最大长度。
这是按共同输入工作量归一化的Tokens/s，不表示两模型内部序列长度或token维度相同，
也不是生成文本速度。各任务之间不能直接用不同粒度的绝对token速率比较。

若共同参考输入数为N，耗时用秒，则：

`R_LightGen2 = N/T_LightGen2`；`R_Qwen = N/T_Qwen`；
`R_LightGen2/R_Qwen = T_Qwen/T_LightGen2`。

例如LGVQ时间质量匹配16视频，统一参考数为16×446=7136。沿用旧核心成本时，
双方均以7136分别除以10.245569591 ms和1188.386146426 ms；
统一输入吞吐的倍率因此等于时间加速比115.990倍。原先LightGen2的3136特征槽位仍在
来源表保留，但不再用于主表的统一吞吐分子。其他已有完整输入计数的行同样处理。

工作簿增加“统一Tokens/s倍数”与“与时间倍率核对”。现有时间、功率、能耗、TOP、
TOPS/W和性能数值均不变，原始计时及原生token计数也不覆盖。
文生图基线缺少该批逐输入长度，双方统一token数和吞吐均标为缺失，不能把Ours一个样例的32
套给全部基线输入。文搜图旧成本范围不匹配的标记保留；含前端版缺测时间仍待补，
不因统一计数而假称已得到完整新倍率。

更新入口为`frontend_tokens_20261023/update_uniform_tokens.mjs`；处理结果、
CSV和SHA清单同目录保存。以下23日核查保留历史原生计数的解释。

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

The following notes document the historical 2026-09-28 component-cost table,
not the input-inclusive 2026-10-24 protocol above. Its workbook
`A100_nine_task_energy_efficiency_COMPLETE_20260928.xlsx` is historical even
though its filename contains `COMPLETE`; do not use it as the new two-table
delivery. The older same-date workbook without `COMPLETE` was open/locked during
that historical regeneration. Earlier workbooks remain archived under
`20260927_demo_energy_efficiency_a100/archive_superseded_20260927/`.

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
