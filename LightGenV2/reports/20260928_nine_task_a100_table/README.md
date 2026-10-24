# A100 nine-metric timing and energy table — 2026-09-28

## 2026-10-25 03:03补记：人工新租80GB已启动DS/Kimi，原40GB缺失队列继续

人工新提供的租机端点已只读核为A100-SXM4-80GB、400W，真实UUID
`GPU-6d1916d1-fbd3-524b-3d98-7fb38512c388`；租期至05:46，计算硬截止05:25
（Unix1792877100），留21分钟取回。复用该新实例挂载的原模型、独立环境、原head和
LGVQ原DATA，不训练或重新下载；源码只Git干净ff，凭据/连接记录仅私有保存。
原manifest与已有mapped manifest核2808条除路径外字段完全相同；原绝对视频路径
仅新增到既有视频的DATA符号链接，未复制工程或替换输入，路径映射收据独立保留。

新T06 `lgvq_mps_e2e_a100_20261024/rental_80gb_20261025_01` controller381，
原`measure_mps_e2e_20261024 --profile rental`派发DS完整版/Kimi各100/50MPS、各三fresh，
共12点。冻结启动HEAD `fd5f8fe7e`，不改原B1/200/零dummy/全窗成本协议。
03:00已完成DS100 r1，r2 worker7108已进入正式200计时；尚不称12点完成。
首轮连续146.5077926814556秒/200、估算系统81340.70337846555焦耳，server原raw审计通过；
10文件14709251字节已逐SHA本地备份，`local_independent_audit01.json`独立复算通过。
正式准确度仍原full558的DS .7167352584517089/Kimi .805465271935947，200预测只作核验。

同机Qwen100三fresh对照已在`rental_qwen_anchor_20261025_01` controller4294等待前驱释放。
已发布`175df3229`的独立wrapper仅绑定新GPU/400W和原模型/head路径，调用原E2E worker，
无网络/输入/精度/计时数学变更；3纯CPU测试双端通过。Qwen原参考report/calls按原SHA
data-only传入canonical，原头/模型和Torch2.6cu124/TF4.57.3环境均已有并核准。
它是80GB平台对照，不替代两表既有40GB唯一共享三轮；单独role receipt明确该边界。
新机ff175df前后DS/Kimi活动记录的47源文件SHA全等，原活动runtime不热改。

实验室原1728678/1737924仍保留原12次idle守卫及T12后继，最后实际检查仍有hmh
1708990/1710971占A100；不杀对方、不混测、不卡住其他新机工作。空出即自动接续。
九行仍13/18、多baseline40GB24/24；新增80GB此刻仅1/12。两新正式Excel仍未生成。
每完整point先逐SHA取回再本地CPU重算，不等租期末才备份；05:46前必须保住原raw与日志。

## 02:40补记：原形状算量已补OpenMoji双方，新的外占只重排缺失工作

02:43后继队列补核：T12 `_02`在两个控制器同时启动时早于前驱首次status创建而退出，
未加载模型；1文件1365字节与1304字节外置日志已SHA备份。`1737cf603`仅修队列读入：
尚未创建/瞬间未写完的status作为有界pending，真正stopped仍拒绝，不改原NN。
4测试双端通过，新T12 `_03` controller1737924、deadline1792874553.3409386，
显式绑定仍活动的T08 `_02`；不再等待1728679，也没有重复CPU神经合同。
server精确干净ff到已发布1737cf603，不顺带覆盖后续peer T21的d9b739fc…源码；
T08冻结runtime前后SHA不变。下文T12 `_02`排队描述由本段替代。

时间/能耗仍13/18，正式准确度仍完整TEST，未训练、未重跑任何合格轮或CPU合同。
按人工最新要求，OpenMoji双方NN dense-MAC已直接从原网络和既有全部600调用形状推导，
新增模型/GPU前向均为0，双端CPU计算解析对象一致。Qwen含原完整24 vision/28 language、
QK/AV主项、原main+3 deepstack merger及完整共享头；每输入0.31419706459456TOP，
与原成本结合为4.2812823067797074TOPS、0.010946241244016098TOPS/W。
Ours原PT metadata与已封`nn_module_shapes.json`逐module计算：实际执行NN
0.00087843005568TOP/输入，排除并行残差的charged-serial NN为0.00081687207936TOP/输入。
它们是明确的**NN MAC小计**，不是完整scalar OP、不是把光学满孔径容量塞成任务OP；
光传播dense-equivalent与GPU FFT仍单独列，未知值不填0，不以NN小计冒充完整光学TOPS。
原非MAC类/功能与残差执行小计保留。私有原件：
`openmoji_qwen_network_shape_MAC_01.json` SHA
`5a7846efdbe199d633353a4f44b0589424b45e1741493a78ae089224f1c60d12`，
server原件`openmoji_qwen_network_shape_MAC_server_01.json` SHA
`c2fcd8613bb30a59718459790eb7b25ab19a8b104118c3b016147c5ab4785ab9`；
`openmoji_e45_saved_shape_MAC_01.json` SHA
`7969ca3eed4fc1b09ea4b932609e052d74dfdf6a45022dc14eda1e5d0580f85e`，
server原件`openmoji_e45_saved_shape_MAC_server_01.json` SHA
`5f6645f6863449ec5a5afc1258764acb50d06992fbc71591b079ec9c577db1bf`。
双端6+3纯CPU测试通过，原raw不改。初次两个纯算术CLI分别遇相对路径和summary字段名错误，
均在写输出前退出，已按实际路径/原schema修复；不是模型失败或GPU重推。
新本地全量重放`measured_index_partial_10.json` SHA
`0077f829ad3b3e117033f6570fd9979ac7455ac6df432d3e936bb8c6ea6a0cbc`，
builder`c5cbf53b2`；只增加07双方NN小计，其他八行及sharedQwen与09解析对象完全相同。
仍PARTIAL、无正式两Excel；server没有生成这个私有index，旧09和以前保留。

旧T08 image-student `_01`在三次idle后开始独立标题库模型加载，但02:28:15实际守卫捕获
新非MPS进程1708990/hmh；未生成标题库/合格查询轮，controller1675400/worker1706059已退出。
实际命令`resume_training.py`于02:28:12启动，后其child1710971在02:29:24启动
`fit_trainable_wave.py`占23062MiB，A100原始UUID明确，不是3090或旧PID猜测。
只停止自己，不杀对方；原11文件80416字节及785字节外置日志已SHA备份。
独立`post_foreign_admission_release_check01.json` SHA
`38355f6717262a2178fefb316b0370f225736bddb540d44f23a7bc625d4b3ff2`
确认controller/worker/私有MPS server1706890退出、privatecontrol无daemon，原release_error
来自外国仍占卡，不是自己的MPS残留。原T12 `_01`仅随前驱stopped退出，未加载CPU模型，
1文件1238字节及1196字节日志SHA保存，不再等这两个旧控制器。

新不可变续队T08 `a100_image_student_input_inclusive_20261025_02`，controller1728678，
deadline1792872485.5070028，冻结`5f83543d1`。仍精确复用已过CPU合同，仅补未完成的库和三轮；
现默认连续12次十秒idle/显存核验（至少两分钟），任一外部进入清零，防止短空档反复加载。
这不改变正式计时/能耗/NN源、250ms支持年龄或任何已合格点。
新T12 CPU `_02` controller1728679、deadline1792874285.509181，显式`--previous`绑定新T08，
前驱全部合格并释放MPS才执行。11静态测试双端通过，两新队列先看真实PID/status勿重启。
本次外占变化需要连续独占窗口协调；无变化不重复通知。runtime/源码均已三端发布，
peer T21的87d0351e7由其窗口先行同步，B100三暂存原样保留。

## 最新人工口径与完成清单（2026-10-25 02:19，北京时间）

本节覆盖下方旧的12/18、Qwen CPU等待和必须每项200次GPU算量回放描述。
人工最新明确：正式准确度沿用已确认的原代码**完整TEST**结果；first200×三fresh只测
时间/能耗，子集预测指标仅作输出核验，不替换完整TEST准确度。OP按原网络和既有实际
形状/长度记录推导，仅补确实缺失的形状；TOPS=OP/(1e12×匹配秒数)，
TOPS/W=OP/(1e12×匹配焦耳数)，不另要求每项整套GPU重放，不训练、不重跑合格点。
原protocol和所有raw不改，索引新增reporting_policy记录该人工覆盖。

现在九行时间/能耗已合格**13/18模型项**，多baseline40GB仍24/24。
新增OpenMoji Qwen固定T04 `a100_qwen_input_inclusive_20261025_01/qwen_repeat01,02,03`，
测量源码`b069f70b3`，controller1639931及worker1640928/1643978/1648295全部complete、退出，
MPS释放。原PNG/native完整Vision+Language/原BF16 CPU feature return/原head及CPU网格，
连续200、零专门预热、首条及199真间隙全部保留；三轮服务器和本地独立CPU审计均通过，
解析对象相同。63原文件15688776字节全SHA备份`backup_receipt_complete01.json`，
外置0字节日志也已取回。600调用73.38854158180766ms/输入、28.703648822496017J/输入，
时间加权board52.91894314590646W；正式完整1000准确度`.8120`，first200`.8`仅诊断。
三份本地独立审计SHA依次`85348829e95aa79955ba9758171edf6e58f330dbb79f6bb563ad0b00019641ca`、
`a6e602f131d1b552abfcc55f922f0b01700231b5ef52c397ea717f54c4e124ce`、
`143a07f0fb1234f710107293836b328e04d410e030c1ddc428824cc0008f70d3`。
原Qwen CPU `_04`真实200/999已通过，contract SHA
`1bf5d3e0a761d7a72b5aa0308568959af10fc0d51565a1002335a752dc40c3bd`，6文件28534字节已SHA取回，
不再等待或重跑它。OpenMoji双方同PNG/UTF8逐条配对，实际Qwen每轮12807有效token，
总38421；由新索引补充Ours共同2298.231084958699tokens/s，Qwen872.5476568929895，
倍率2.6339318738673287恰等于时间反比；Ours原raw里当时诚实记录的null不改。

已本地全量独立重放`measurement_protocol_20261024/measured_index_partial_09.json`，
SHA `6393a72b911235a7537e0bfb6ab6f89e8b6900175ac82b03b3601b744842b65c`，
builder已发布`f77e9a02e`，仍PARTIAL；server没有生成这份私有index，旧08及以前保留。
正式两Excel尚未生成，不用旧COMPLETE/frontend_tokens代替。

余5项成本为LGVQ Ours temporal/spatial、T08图搜文Ours、T12双方。
图搜文`.8058`原PT a2aa9a93…和原optimized配置已绑定；真实图200/2399及标题0/99
原/插桩完整CPU逐位通过，contract SHA
`24869bddce997cf26f0ce3dd2b2a89c7d08d956d25665d27cbf04bd0ad008e8a`，
6文件155083字节已SHA备份，不重复该合同。唯一GPU队列controller1675400，
T08 `a100_image_student_input_inclusive_20261025_01`，冻结runtime`d41653743`，
deadline1792869065.7180984。独立fresh进程先建100标题FP32库（非正式成本），
再三fresh原图200；CPU原非stable排序、原centered bicubic fit224、真实merger与6pass不改。
最新检查还在waiting_foreign_process、无worker/合格轮，不假称已开始计时。
真实A100占用PID1671965/hmh，02:09:17启动`fit_frozen_wave.py`，35606MiB；
独立观察SHA `13fe7b4e8cd63b0dfafed1782b0f50151cbcfa3eeaf00780f9defb3cba00b5fa`。
不杀对方、不混测，不将历史占用解释成用户看到空卡时也被占。

T12原5b4f主体92274638字节、599805baseline1365318182字节和原TEST/instruction/embedding
均重新全SHA核准，原t12_assets/venv依赖齐全。CPU前端/6pass观察入口`da85e0ec8`，
静态3测试双端通过；控制器1690147位于T12 `a100_live_frontend_contract_20261025_01`，
等待前述三轮全部完成/MPS释放和同一锁独占后才执行，不与计时重CPU骨干并发。
这是CPU合同准备，非正式200×3，尚未称神经通过或双方正式队列均已排好。
所有新源码已三端Git同步；peer T21的c83b7fa0f由其窗口自行先同步，B100三暂存未动。

## 新旧版入口与完成清单（2026-10-25 01:32，北京时间）

OpenMoji Qwen原生网络CPU入口正在独立核验，唯一当前CPU控制器1597350位于
T04 `runs/simulation/a100_qwen_live_frontend_contract_20261025_04`，冻结源码`06134eb87`，
有限deadline1792868467.4165387。共享锁及连续12次真实独占检查后才执行TEST200/999的原/观察版
完整神经前向逐位比较，全程隐藏CUDA、不启动MPS；尚未形成通过合同或正式200×3，
不能据此升级13/18，后续先读实际PID/父子status，不重复启动。
实际baseline主体epoch30/PT0be775…3907085字节已fresh全SHA核准，原完整1000指标`.8120`，
原resolved配置SHA52093084…、性能报告SHAdd22cbdc…与新布局/原Instruct快照一致。
入口保留原`qwen_shared.extract_native_one`的PNG/chat processor、全部vision/language层、
pre-merger capture、BF16 CPU feature return，再执行原head的float32/H2D及原网格后处理；
没有将缓存训练占位Identity当真实前端，也不静默改为省略原CPU返回的另一快路径。
此次CPU合同核验和后续正式同输入Qwen调用完成前，OpenMoji共同Tokens/s仍待补。
旧CPU `_01`仅原训练legacy路径迁移差异被拦截，发生在模型加载前；已按原metadata恢复
该未被Qwen分支使用的训练路径，未换权重或模型，4文件3658字节和721字节外置日志保留。
旧CPU `_02`在真实第200索引核验阶段被外部占卡守卫终止，4文件1925字节与704字节日志已
SHA备份；没有合格合同，不据status推断具体前向次数。新`_03`也在加载期被短占用终止，
追加原证据记录PID1591226/618MiB/1792862864.7003164，5文件2075字节及569字节日志已SHA取回。
后续快照compute又空，不编造该短暂进程的用户/命令，不把空快照当未来独占。
因此新`_04`将入场观察由3次加强为连续12次十秒检查（至少两分钟），有外部即清零等待，
不是反复加载模型盲试，也不放宽任何计时/250ms功率门槛。新静态检查补齐原三参compose
调用和外占快照/进程明细的留存，11项纯CPU测试双方通过；旧`_01/02/03`均停止，不重启旧PID。

以下OpenMoji LightGen结果与partial08已正式独立核验；Qwen CPU准备不改变其12/18完成度：

新布局实验室`.8655`所对应epoch45/PT03cb861c的LightGen真实输入正式三轮已完成：
T04 `runs/simulation/a100_e45_input_inclusive_20261025_01/ours_repeat01,02,03`，
冻结测量源码`cef807950`。controller1551824和三worker1552804/1554725/1557067均退出，
私有MPS已释放；三轮各200调用完整保留首条及全部199间隙，服务器和本地独立CPU审计全通过，
两端审计解析对象逐项相同。原66文件43339307字节逐SHA本地备份，根
`backup_receipt_complete01.json`；外置controller日志0字节也已精确取回。
三个本地`ours_repeatNN/local_independent_audit01.json` SHA依次为
`63bc770ea5f0da27767575d364d7848599361194b6b9d24af254a90d377b009d`、
`66516e318aa80f37485528017fb24b55d75ac19b40bf3c0b5c34f803132ad771`、
`4ce158d12594e83ef45e642af9dfb83da83c7df9f3bbc5afa371dd7d9be4a4e6`。
从原PNG/UTF8、实际输出/网格指标、源Git、原NN无重叠分段/实际merger、进程/MPS及
全连续窗NVML独立复算；三轮投影合600调用为27.8627333949ms/输入、
分阶段估算3.01392649295J/输入。这里包含每输入6次参数推算光传播，
是光电混合硬件投影，不是装置实测E2E；诊断软件wall/板卡积分与所选阶段能耗分列。
三轮新first200 changed-cell accuracy均`.855`；历史同PT完整1000仿真`.8765`及实拍`.8655`
分别保留，不混称全量性能重现或新实拍。尚未完成同输入Qwen正式调用，
共同输入token和Tokens/s保持null/待实测，不用Ours plain内部token替代或填0。

正式测量前，原TEST200/999的完整原前向与选定NN插桩前向逐位相同，
`a100_e45_stage_contract_20261025_02/original_instrumented_live_CPU_contract/contract.json`
SHA`8aacd85678b9baff63f82d845f226abd86f0ee757e1af5cc22ff47ab6bb10a9f`，
六传播、实际bridge/readout和两个并行残差均正确观察；全程CPU/noCUDA。
其5原文件158648字节已SHA备份，不重做已合格的前端/插桩合同或正式三轮。
旧stage `_01`的父类观察绑定错误在CPU插桩调用处被拦截，未开始GPU；
保留原5文件56108字节及732字节外置日志，修复实际实例绑定后才以新`_02`通过，
没有替换原神经计算、忽略真实merger或将缓存作为在线输入。

本地全量独立重放新`measurement_protocol_20261024/measured_index_partial_08.json`，
SHA`b5e3313e79f85419cb8d47ce5a7f4e47dbd4142ff31c617b31e1079fd67f4289`；
builder源码为`6be485bb1`所发布版本，生成时仓库HEAD为同主线peer的`841909a94`，
该peer仅T21文档/新增协议与测试，builder字节未变。除第07行新Ours数据外，
其他八行及共享Qwen anchor与partial07解析对象完全相同；旧07及以前均保留。
服务器未生成这个私有新索引。当前12/18模型任务项时间/能量合格，余6项是
LGVQ Ours时间/空间、T08图搜文Ours、OpenMoji Qwen、T12双方。
只有04文搜图Qwen实际200输入dense-MAC小计已验证，其余OP/TOPS/TOPS/W仍待匹配回放。
正式两份新Excel仍未生成；旧COMPLETE/frontend_tokens均不能作为新交付。

以下00:38及以前状态为历史，本段和不可变raw为最新事实：

文搜图LightGen现在固定复用`a100_title_student_input_inclusive_20261024_02/ours_repeat01`，
加`a100_title_student_input_inclusive_20261025_04/ours_repeat02,03`，三轮各200真实调用已
服务器/本地独立CPU复算全部通过；新controller1481157已complete、退出并释放MPS。
后两轮39原文件7888741字节已逐SHA备份，原首轮及所有中断/功率拒绝证据均原样保留。
合600调用，所选混合投影11.7767602413ms/查询、分阶段估算1.28238861555J/查询，
其中实际原NN同步wall、3次参数光学成本和全部真实199间隙均按协议计入；不是装置实测E2E。
三轮100唯一标题Hit@1均`.85`，历史完整`.86`另列，不为匹配旧精度改B1/模型/精度/候选库。
共同输入28728token，4065.634268 tokens/s；同批Qwen 1269.810077 tokens/s，
倍率3.201765634等于时间倍率，不是生成吞吐，也不据此声称100倍。
本地新`measurement_protocol_20261024/measured_index_partial_07.json`已完整独立重放生成，
SHA`15acf76ad97bde3ec15be3562d0356e3bb8910b624b534c8e197ddd95e2e621b`；
其builder已绑定发布的`2d86ba593`。除第04行新Ours数据/匹配吞吐外，其他八行及共享Qwen anchor
与partial06解析对象全同；旧06及以前不覆盖。服务器未生成该私有新索引。
目前11/18模型任务项时间/能量合格，余7项与其余真实输入OP仍待补；正式两Excel尚未生成。

新布局实验室`.8655`对应epoch45的真实输入CPU前端核验也已通过：
T04 `runs/simulation/a100_e45_live_frontend_contract_20261025_04` controller1493640、
CPU worker1495262均complete并退出，真实TEST200/999的原PNG+plain tokenizer/原BF16查表
与原缓存生产者的输入、原完整模型各输出及原网格后处理逐位相同。实际使用的原权重文件
fresh全SHA与原metadata一致，实际主体03cb861c…仍为同实验室/仿真版本；全程无CUDA/MPS。
原6文件38918字节已data-only逐SHA取回，`backup_receipt_complete01.json`；
`original_live_input_CPU_contract/contract.json` SHA
`ee9d944958ecab3d5eb44d05864ee25670d9c89197d37b6518c3b459b9dd628f`。
本地核原字节SHA/采用身份及两例检查记录，不冒称本地又做了一次神经前向。
这不是选定NN插桩合同、正式200×3、全TEST重评或新实拍，不能把此项升级为已测时间/能量。
旧CPU`_20261025_03`在外部1489635出现时停止，原件和外置日志已SHA备份；不重启旧PID。
目前没有本聊天活动GPU/CPU队列；下一步推进余7项真实适配、选定NN合同和匹配OP回放，
新GPU仍重新检查真实独占，不重复任何已合格标题/图库/CPU前端合同。

以下00:28及以前为历史，故障/运行状态以上面及真实原始数据为准：

标题缺失轮续测`a100_title_student_input_inclusive_20261025_03`的第二轮完成200调用，
但独立采样源在首条真实查询内出现366.370ms的观测间隙，超过250ms，因此整轮拒绝，
未生成合格summary，也不把其200条预测拼入正式结果。原向量/得分/全部排名与已合格
首轮逐位相同；错误限于功率覆盖，不通过更改模型、删首条、插值或放宽阈值来补救。
实际独立observer原始接收时戳也有这个缺口，不能归咎于模型端IPC读取延迟；
现有证据尚不能区分驱动查询阻塞和主机调度，不编造更细的根因。
19原文件3924878字节、外置日志及T04随前驱退出的原状态均已逐SHA取回保留。
相同采样器随后20秒CPU只读诊断396条、最大间隙54.351ms通过；没有GPU/模型调用，
也不把诊断通过当成下一正式首条必然覆盖。未改任何计时源码、神经计算或250ms门槛。
新有限T08控制器1481157在`a100_title_student_input_inclusive_20261025_04`仅补2/3轮；
若同因再发，保留证据继续查采样源，不做无诊断密集重试。第一轮仍固定复用原`_20261024_02`。
新T04 CPU控制器1481204在`a100_e45_live_frontend_contract_20261025_03`，
明确引用新`_20261025_04`和已合格旧首轮，仍待完整三轮及释放后才执行真实前端CPU合同。
旧1460638/1464413均已停止退出/MPS释放，不再等待或重启；新状态以实际原始队列为准。
私有诊断`measurement_protocol_20261024/abo_title_power_gap_diagnosis_20261025_01.json`
保留实际缺口及恢复诊断的边界；九行仍10/18完整项，partial06不覆盖，暂无两份新正式Excel。

以下00:18及以前状态保留为历史：

T08标题`a100_title_student_input_inclusive_20261024_02`已完成第一轮200调用，
服务器及本地独立CPU从原始输出、完整排名、NN分段、进程/MPS及连续窗功率复算均通过。
此轮100唯一标题Hit@1为`.85`，历史`.86`单独保留，不把两者冒称逐位性能复现；
共同输入9576token、所选混合投影11.9546907907ms/查询、分阶段估算1.3009343167J/查询。
这只是1/3轮，尚不升级整项合格数，也不是装置实测E2E或新完整算量。
第二轮在加载期、0计时调用时被外部`hmh/verify_tasks.py` PID1453028打断，
该进程00:06:45启动，当前约12210MiB；只停止自有worker1451986。
原controller1411319及其私有MPS均已退出，T04旧等待控制器1443601随前驱停止而退出，
不再等待或重启这两个旧controller。独立释放收据SHA
`e9279078b74296b597ae46dfa9c6a086ebce6810852c2ca2bec1a71d75c30ba6`
确认无自有进程/私有daemon；原`GPU not released`报错是对方仍占卡，不是自有MPS残留。
T08完整29份原文件4000762字节逐SHA备份到原run的
`backup_receipt_partial_foreign_interruption01.json`，第一轮
`ours_repeat01/local_independent_audit01.json`保留；两控制器外置日志也均原SHA取回。

按人工“失败后直接补缺失轮”指示，新有限controller1460638位于
T08 `runs/simulation/a100_title_student_input_inclusive_20261025_03`，只`--repeats 2 3`，
精确复用已合格`_02/ours_repeat01`、原图库和CPU合同，不重跑第一轮/图库/合同。
原计时runtime及全部被冻结依赖SHA核对不变；外部未退出时仍只等待真实独占。
OpenMoji串行入口增加显式固定保留首轮的参数；只有新前驱恰好完成2/3轮、MPS已释放，
旧首轮独立释放证据不变，且三轮原raw再次独立CPU复算通过，才允许执行新布局真实CPU合同。
不能把新前驱两轮直接当作三轮，也不能绕过停止的旧队列。
九指标行仍10/18整项时间能量合格；partial06及旧索引不覆盖，新正式两Excel尚未生成。

以下00:02及以前为历史，实际状态以上面和队列原始状态为准：

A100外部占用已变化：原`hmh/encode_data.py`退场后，`hmh/fit_quality.py`新进程1440470
（父1423904、23:54:58启动、12214MiB）进入；不把中间只有一次idle检查当作已独占。
T08标题续测1411319仍`waiting_foreign_process`、无新GPUworker或合格轮，私有MPS未启动。
T04真实前端CPU控制器1443601仍`waiting_previous_queue`，不会在当前占用期间加载CPU骨干。
追加只读原始证据`T08/runs/simulation/a100_title_student_input_inclusive_20261024_02/`
`foreign_admission_observation_20261025_0001.json`已原SHA取回，SHA
`331e5a29268fd141589f9ee35e5487b7b7f66bf71fb85b279b95b034a3d39133`。
未中止对方、未重建图库或重复合格测量；仍10/18项时间能量合格，无新正式Excel。

以下身份结论和旧进度保留；运行状态以上面及实际队列为准：

人工已确认OpenMoji使用新布局，并指定实验室`.8655`所对应的版本；旧`.8880`是未更新表项。
本次重新读取并SHA核验已封存的实验室原始complete报告、全部比较证据及实际主体PT，
确认实验室`.8655`对应epoch45/PT`03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21`，
同PT仿真`.8765`，新布局为`layered_anchor6_svg_v3`；Qwen仍epoch30/PT`0be77579…`、`.8120`。
实验室报告为1000 TEST/6000 CCD、无下游重训；报告SHA
`ba51d8b425fc556f55e3c0506675c4223755673701854ad5c1d54b1f62b08dfc`。
使用已封存原件而非新连接设备、采光或重评，因此不称本次新实拍；原报告缺少TEST清单SHA
和逐样本预测ID的限制仍保留，不将报告级权重绑定冒称已完成逐例CCD配对审计。
新增不可变`measurement_protocol_20261024/protocol_amendment_openmoji_e45.json`记录人工身份修订，
原`protocol.json`及全部旧索引/raw原样保留。`protocol_identity.py`只叠加第07行身份，
计时边界/200×3/能耗/MPS/共同Tokens/s及其他任务身份不改；仍不是第07行的新时间结果。
新索引将分别列仿真`.8765`、实验室`.8655`、Qwen`.8120`，后续只用这个epoch45主体补测。
原“待人工版本选择”已解除，不再重复提问，也不再使用旧网格`.8880`测速。
本地已真实重放生成`measurement_protocol_20261024/measured_index_partial_06.json`，SHA
`2ea517da347732aa80cab1bddbdc13c7ea1a6201a1cea9e691a62edeb90654ba`，
其余八行及共享Qwen anchor与partial05解析对象完全相同；旧05及以前不覆盖。
新版索引仍为PARTIAL、10/18项时间能量合格，第07行新计时/OP待做，不冒称正式两Excel已生成。
服务器未生成这个私有index。本次源修订双端29项CPU测试通过，原报告审计及4项既有量化检查
也通过；实际主体PT本机/服务器均26562234字节、同完整SHA。私有独立确认收据为
`measurement_protocol_20261024/openmoji_e45_user_confirmation_01.json`。

另准备第07行新布局epoch45的真实输入CPU前端合同`prepare_openmoji_e45.py`及有限串行入口。
仅在固定文搜图三轮确实完成、释放锁/MPS并连续三次独占确认后，才加载CPU模型；
不与任何GPU计时窗并行重CPU，也不训练、创建MPS或使用CUDA。真实TEST200/999的
原PNG/缓存生产者作为数值参照，在线输入另从PNG和原plain-text tokenizer/embedding重建，
逐位核输入、原完整模型输出和原网格后处理；target只用于参照元数据、不传给模型。
缓存只在不计时CPU合同中作为原数值参照，不能作为后续在线计时输入。
14项纯CPU/静态合同通过仍不等于这两张真实模型前向已执行，更不是选定NN插桩或200×3新测。
有限控制器1443601现位于T04 `runs/simulation/a100_e45_live_frontend_contract_20261024_01`，
实际`waiting_previous_queue`，明确引用T08标题`a100_title_student_input_inclusive_20261024_02`；
尚未加载CPU模型或执行真实CPU前向。新源码`c953a06fb`双端14项测试通过；
服务器原token缓存JSON声明plain text/no chat、零语言TF及原Instruct权重文件SHA
`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`。
本轮只读检查确认原/新所有推理配置字段相同、原光学base配置字节相同；
上述权重文件整文件fresh SHA、实际live lookup与原缓存及完整模型逐位合同仍由排队worker执行。

以下23:27及以前状态作为历史；未完成的正式测量仍以实际队列和原始审计为准：

独立`.8600`图库`a100_title_student_gallery_20261024_02`已完成原B1/2400张真实图
编码，父1398431、子1398856均退出，`controller_status`确认complete及私有MPS释放。
不重建该已完成图库，实际标题/图片CPU合同同样保留复用。
图库13份原文件904353字节已data-only逐SHA取回，`backup_receipt_complete01.json`；
双端独立CPU检查2400个原CSV身份、FP32[2400,64]、原文件SHA、78份源Git及全程进程/MPS通过。
两端`*_independent_gallery_audit01.json`解析对象一致，图库NPZ SHA
`d653f84d6509872911c0a725fa1eb4247d8d06e74b6f232b18a45954fd669251`。
正式`a100_title_student_input_inclusive_20261024_01`首轮在117个查询后，
守卫发现外部非MPS GPU进程而拒绝；原控制器1398473及worker1403037已退出。
只读后查A100外部PID1404356、2384MiB，hmh的`CLIPSalNet/bin/python encode_data.py`，
服务器启动时间23:20:02；本聊天没有终止对方。
首轮部分CSV/NPZ/功率/进程原件拒绝保留，不将其中任何部分拼成合格200调用。
20份拒绝原文件2382385字节已逐SHA备份，`backup_receipt_external_interruption01.json`。
原清理函数因外部仍占卡报告`GPU not released`；追加独立只读收据确认自有三个PID
均退出，私有control返回无daemon，不能误报为自有MPS未释放。收据SHA
`7440367a90bf31e6c56fa7e980e74bb517557b9f10a5b7ab4aa4c585b8d6025e`。
新有限控制器1411319在`a100_title_student_input_inclusive_20261024_02`引用同已完成图库，
不重建图库或CPU合同，等待三次真实独占后才补齐尚无合格轮的标题三fresh进程。
冻结运行源码`41d468474`；不重做其他任务合格轮。
当前仍10/18项合格，未发布新正式Excel。

下列23:17及以前为历史；现有运行状态以上面和实际子status/原始数据为准：

当前独立图库为`a100_title_student_gallery_20261024_02`，控制器1398431、GPU子进程1398856；
复用原通过的图片CPU合同DATA，没有再次执行该CPU前向。实际状态为
`encoding_original_gallery2400`，原整库函数完成前不提供逐张进度，不把状态中的初始0张误判成未运行。
正式标题三fresh有限控制器1398473已在
`a100_title_student_input_inclusive_20261024_01`等待该图库完整通过并释放锁/MPS后再入场。
源`41d468474`已本机/GitHub/服务器同步，23项纯CPU合同双端通过；
当前仍10/18合格，不提前称图库或标题200×3完成。

文搜图LightGen图库`a100_title_student_gallery_20261024_01`已完成两张真实图片的
原/分段CPU逐位合同，但GPU模型加载后，在来源登记时把Torch声明的虚拟相对
`_classes.py`误当作仓库实体文件而停止；原2400图库编码函数尚未调用。
父/子进程退出、私有MPS释放，原12份数据150185字节及外置完整日志已SHA备份。
来源登记修复只将虚拟相对模块单列说明，缺失的绝对仓库实体源码仍失败；不放宽真实源码SHA。
新不可变run可精确复用已通过的原图CPU合同DATA，不重复该CPU前向或原标题合同。
新`.8600`正式标题入口`measure_abo_title_student.py`、有限三fresh入口及独立CPU审计
已准备，23项无模型/无GPU单测通过。在线仍从UTF8开始，保存tokenizer/H2D、
原选择NN同步wall和CUDA Event、CPU输出/FP32余弦完整稳定排名及199真实间隔；
三光pass及固定分阶段能耗不变，FFT/纯layout/并行残差均保留原结果但不计所选投影。
仅查询进程不做图库或dummy前向。共同token由已合格Qwen同轮同标题原IDs/mask逐条绑定，
两者实际文本输入IDs/mask另行核对；不拿内部token数改变Tokens/s倍率。
这里只是新入口准备，图库及LightGen200×3完成前不升级10/18或发布正式Excel。

图搜文Qwen缺失第三轮已在新`a100_image_input_inclusive_20261024_03/qwen_repeat03`
完成并释放MPS；原前两轮固定复用`_02/qwen_repeat01/02`，拒绝的`_02/r3`不入表。
新20份原文件9535787字节已逐SHA备份，三轮均从原JPEG字节、FP32向量/得分/完整排名、
源Git、进程/MPS及全窗功率在本地独立复算通过。合并600调用85.694760ms/查询、
33.287886J/查询，时间加权A100板卡50.247157W，共同输入41184token，800.982460 tokens/s。
三轮首200子集Hit@1均`.705`，完整2400历史`.7358333333`仍单独标注。
当前为10/18项时间能量合格，其他八项仍待补，不冒称两张最终表已完成。
新本地私有`measured_index_partial_05.json`已真实重放生成，SHA
`e283f0a204a4dfae2c66ef6978b7f215920f546125749cbcba0f0d0d63bf223a`；
旧04及以前索引保留，服务器没有生成此私有新索引。
本地LSP审计保留已有SciPy运行时，图搜文审计显式调用已有匹配JPEG解码器的bundled运行时，
各自原审计函数不改、数据检查不放宽，没有安装/升级包、伪造loadmat或重跑模型。

文搜图LightGen `.8600`的实际标题0/99原完整前向与分段前向已逐位通过CPU合同，
`a100_title_student_contract_20261024_01`有限控制器已完成、零GPU。
下一步`prepare_abo_student_gallery.py`只生成同主体原B1/2400图候选库：
先用首200以外两张真实图片做原/分段CPU逐位合同，再在独立fresh GPU进程调用原
`reverse_runtime.encode_student_samples`，保存FP32而非冒用旧FP16保存件。
该离线图库没有标题查询/预热，不计正式时间能量；后续标题三fresh进程各自从第一条计时。
准备入口14项纯CPU测试通过，尚未据此声称图库完成或LightGen200×3测完。

以下22:43及以前记录为历史，运行状态和完成数以上面新段为准：

OpenMoji第07行的原始版本核对发现输入身份冲突，尚未启动该行新推理。
采用`.8880`的PT `579b19e7…`及原完整头确认报告确实存在，但原生产者实际读取
`routerfill_shared_s73/resolved_config.json`，对应旧`openmoji_grid_v2`，
不是新分层布局。Qwen `.8120`的PT `0be77579…`及既有全量导出则绑定
`openmoji_layered_anchor6_svg_v3`；该布局已有对应的epoch45 Ours `.8765`，PT `03cb861c…`。
两份实际服务器TEST清单已data-only逐SHA取回：旧网格`c86765b9…`，新分层`93aa528c…`。
各1000条的编号顺序完全相同，但源/目标网格1000/1000不同、指令992/1000不同；
首200条的源/目标网格也全部不同。不能只按`test_000000`等同名编号判定共同输入。
原配置、完整清单及只读核对收据保存在私有
`measurement_protocol_20261024/openmoji_identity_20261024_01/`，未修改模型/训练/批准协议。
已询问人工选择：新分层`.8765/.8120`对应配对，或保留旧网格`.8880`并让Qwen在同一旧网格重新评估测速。
人工答复前仅此行不混测/不静默换PT，其他任务继续；此核对不否定已合格的其他时间数据。

另准备有限`run_abo_student_contract.py`串行入口：只在现有图搜文Qwen续测队列确实
完成第三轮、释放同一把锁并连续三次验证独占后，才启动`.8600`原标题CPU逐位合同。
它不提前加载大模型、不创建MPS或执行GPU，禁止与其他计时窗并行重CPU；
前队列停止时不绕过或重跑Qwen。源身份路径及队列/结果门槛10项纯CPU测试通过。
该准备本身不是实际神经合同通过，更不是LightGen三轮已测。

图搜文Qwen原`_02`的前两轮200调用已完成并通过服务器独立CPU审计；
首轮也已本地从原JPEG字节、完整FP32向量/得分/排名、源Git及全窗功率复算通过。
新首200子集Hit@1为`.705`，与完整2400历史`.7358333333`分别标注，不混作同一精度。
第三轮在模型加载、零计时样本时遭遇外部`hmh`进程1341516，22:19:11启动、
命令`/home/hmh/anaconda3/envs/CLIPSalNet/bin/python encode_stages.py`。
仅停止自有第三轮worker1341118；1323283控制器退出，未混测或终止对方。
原私有MPS已退出。原清理函数因对方仍占卡而报`GPU not released`，不是私有MPS仍在：
后续只读收据确认自有controller/worker/server全退出、私有control返回无daemon，
GPU剩余只有对方3138MiB。原报错、加载守卫及拒绝原件保留。

新增`run_abo_image_remainder.py`固定复用原前两轮和原FP32库/CPU合同，不重新推理
候选库或合格轮；先等真实进程和显存释放，再只补第三轮到新不可变run。
这仍是9/18模型任务项完成，不把2/3轮升级为完整一项。LightGen`.86`的CPU神经合同
准备源码已发布，但因前队列中断未加载模型，也未开始正式LightGen测量。

本地旧Pillow10/JPEG9对179/200个原JPEG解码产生不同RGB字节；原文件SHA和尺寸
均相同，不是换了输入或GPU结果。改用已有bundled Python3.12/Pillow12.3/libjpeg-turbo
后，200个解码RGB SHA全部逐位匹配原测量记录，原审计器一行未放宽即通过。
没有安装/修改包或重跑模型；诊断留在私有`local_abo_image_decoder_binding01.json`。

图搜文Qwen的新`a100_image_input_inclusive_20261024_02`已实际通过两张原TEST图片的
原/新完整CPU前向逐位合同，离线fresh进程生成原B1 FP32标题库已完成，接着执行
独占MPS100三轮。冻结测量源码`19d0e4be4`；尚未三轮审计前不升级完成数。
先前`_01`在模型加载/标题推理及正式计时开始前发现服务器少一份原性能JSON，
原6文件16155字节已SHA备份；缺失的3750字节DATA按本地原SHA
`dc1cc8d23895bc35ff9fcbbac0ceae4f95efd8495395b169565d4ea9ac0198ba`
恢复且未覆盖任何差异文件，随后直接使用新不可变`_02`继续，没有等待人工再次批准。
原参考报告与源Git、模型/精度/提示词/统计范围未改。

另已准备采用`.8600`的T08原`build_student/load_checkpoint/student_embeddings`
完整前端及语言光路计时分段；`abo_student_graph.py`只提供真实标题的CPU逐位合同，
不是已完成的200×3结果。原`.86`纯标题在线3次传播，视觉侧和原生merger应无在线
调用；实际输入embedding、语言adapter/CCD非线性/读出均保留，原并行残差照算出
原结果但不计所选投影。CPU真实前向通过前不会启动正式LightGen测量。

文搜图Qwen的200输入MAC审计已在服务器独立完成，原GPU回放没有重跑；
`server_independent_MAC_audit01.json`已按SHA取回，与本地审计解析对象完全一致。
服务器原件SHA为`943e176839aa518d0e5e050e1dbe3ecfa2b9c5e093abad268aa5e2f6bf39280e`。
两个文件的换行字节差异保留，不冒称文件SHA也相同。

下一项是图搜文Qwen的原`.7358333333`版本。新增独立适配先将原Git生产者的
`_template/_inputs/_embed_title`逐AST核对，原teacher特征源码逐SHA相同；
候选标题库在单独fresh进程按原B1生成FP32、归一化2048维并保存SHA，而不是
使用旧FP16保存件或在正式查询进程中先推理标题。随后用不在首200计时集合中的
两张真实TEST图核对原/新完整CPU前向，再从文件输入到CPU完整100候选排名执行
三fresh×200调用。保留原动态EXIF/RGB、50176像素processor和GPU非稳定排序。
15项纯CPU合同测试已通过；这个准备与随后入场不提前计入完成项。
当前仍是9/18项时间能量合格，新正式Excel尚未生成。

T08文搜图Qwen恢复后的三轮已完成，控制器1243879退出、私有MPS已释放。
原run `a100_title_input_inclusive_20261024_02`的52份数据原件（11,331,196字节）
已逐SHA本地备份；四份原CSV也逐SHA保存为该run的`audit_original/data/`。
三轮均在服务器审计通过，并在本地从原CSV、600调用、FP32向量/得分/完整稳定排名、
源Git、进程/MPS及全连续窗功率重新独立复算通过。标题在线查询不需要重新解码或
构造2400张离线gallery图；候选库仍使用原FP32数据的SHA，不因本地审计而重建。
两整遍100标题的三轮Hit@1均`.82`，合并600调用平均37.706426ms/查询、
全连续窗能量估算14.923917J/查询，全窗时间加权A100板卡57.592411W，共同输入token总计28728。
这只新增Qwen一项，文搜图LightGen还没测完，不能提前给该行完整新加速比。

新私有`measured_index_partial_04.json`已在本地真正重放生成，SHA
`f013fb4f74ecf85e3f04e7442261424e2d73fb8a0a4ff7d4de91be9147fdc2eb`；
保留partial01/02/03，不覆盖旧索引。当前九行是9/18项时间/能量合格，
仅文搜图Qwen新增真实200输入的dense-MAC小计，其余算量仍待匹配回放，
不把中间JSON冒称两份正式新Excel。

T08 Qwen独立不计时回放在`a100_title_operations_20261024_01`，200次真实输入的
FP32向量、得分及全排名与正式三轮逐位相同。13,550,379,339,264 MAC/200查询，
逐查询平均67.751896696 GMAC，按1MAC=2OP为0.135503793 TOP（MAC口径）。
`audit_abo_text_ops.py`在本地从逐算子实际形状独立复算，并用原config的Q/K/V/O、
SwiGLU、QK/AV及2400候选点积主项交叉核对；差额1,838,592 MAC精确对应原RoPE位置构造。
非MAC与未分类算子逐项保留，不冒称完整scalar-OP或硬件峰值。匹配上述正式时间/能量
得到3.593653 TOPS、0.009079640 TOPS/W和1269.810077共同输入tokens/s。
该回放的时间/功率不参与正式比较。其计算完成后发生同进程CUDA客户端无法关闭自己
MPS的退出清理错误；worker退出后另用CPU确认私有clients空并释放，A100重新无计算
进程。原错误日志及一次收据序列化未完成原件都保留，未重复推理或覆盖数据；
新`post_worker_mps_release02.json`及退出后封存清单记录该区别。
9份raw共20,864,001字节已逐SHA本地备份，原外置controller日志也单独取回。
通过的MAC数据与退出故障分开标记；后续入口改为CPU父进程在GPU子进程退出后关闭MPS，
不以删除错误或重跑合格数据的方式掩盖问题。

21:08人工明确纠正：失败原因修复并核验后直接重跑，不再逐批等待额外恢复确认。
已合格轮次仍不重复，失败原件与拒绝证据不覆盖；外部占用时仍不能混测或终止他人。
T08文搜图Qwen的`_02`先通过真实标题原/新CPU前向逐位合同，再执行独立MPS100三轮。
执行冻结源码为`0f458cab6`，原`_01`的零GPU调用前检失败记录保留。
后续同类可修复错误按此人工指示处理，不再逐次等待额外确认。

LGVQ前端身份准备已完成一项新核验：`bind_lgvq_front.py`在CPU只读取原patch、
position及embedding四个参数张量，逐dtype/shape/字节SHA与原归档全部相同；
原位置插值函数SHA相同，空间与时间原prompt的真实tokenizer/embedding输出也均与
采用的正式release语言输入逐位一致。未加载完整Qwen或执行视觉/Transformer/GPU前向。
`AutoConfig`的vision dtype为null，而原缓存生产者显式传入BF16；只恢复这个明确的
构造参数，其余config字段逐项相同，四张量的实际BF16字节另行校验，不放宽模型身份。
最初config差异检查与后续通过记录均留原件，私有通过收据为T06
`runs/smoke/lgvq_frontend_identity_recovery_20261024_01/front_parameter_binding_02.json`，
SHA `6bb38fe494d35cc568fc3e00e1bd9901ce6f9e88e5e5dd8d3d7041292d8d4ed0`。
原224raw/Conv5视图实际使用`.10/.37/.63/.90`及全图crop1.0，Qwen原视图使用
`.10/.366666…/.633333…/.90`及crop.65；不能把两种原视图合并为一种取巧。
两项LGVQ Ours真实视频的patch/quality/Stem数值合同及正式计时仍待补。

42项纯CPU单测已双端通过；算量台账新增GQA按query头计数的真实CPU小张量合同，
插桩前后逐位一致。私有`operation_ledger_cpu_contract_02.json` SHA
`b4467544aaaaffb2e58aaede5c44bee678b6c8e6ae45c0913c64ff9ae0230952`，
仍只是工具测试，不是任一任务200样本OP结果；全部正式TOP/TOPS/TOPS/W仍待补。

### 20:35及以前的阶段记录（以下等待恢复文字已由21:08人工指示替代）

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
`measured_index_partial_05.json`和`temporal_matrix_partial_03.json`是当前可审计中间索引，
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
| 02 | ABO图搜文 | 原模型计时适配/正式测量待补 | `_02/r1,r2`+`_03/r3`三轮已独立审计 |
| 03 | ABO图搜图 | 三轮已审计 | 三轮已审计 |
| 04 | ABO文搜图 | 原标题CPU逐位合同已通过；图库及正式测量待补 | 三轮完成，本地独立审计通过 |
| 05 | LSP | 三轮完成、本地独立审计通过 | 三轮完成、本地独立审计通过 |
| 06 | SALICON | 三轮已审计 | 三轮已审计 |
| 07 | OpenMoji布局待确认 | `.8880`对应旧网格；与新布局配对版本待人工选择 | `.8120`对应新分层，需与Ours共用实际输入 |
| 08 | ABO文生图 | 原模型计时适配/正式测量待补 | 原模型计时适配/正式测量待补 |

这相当于18个模型—任务项中10项已有合格三轮时间/能量，其余八项仍待适配/正式测量。
匹配双方完成的任务是图搜图、LSP和SALICON；LGVQ时间/空间目前仅Qwen完成。
除文搜图Qwen已验证的dense-MAC小计外，TOP及其派生TOPS/TOPS/W仍需匹配真实输入形状的独立不计时回放；共同输入
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
- 文搜图Qwen完成：`tasks/t08_abo_image_text_retrieval/runs/simulation/a100_title_input_inclusive_20261024_02/`。

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
