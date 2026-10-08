# Editor16 原架构三组累计 robust（2026-10-03 新授权）

## 当前状态

### 2026-10-05 北京14:49 温和CCD+DC完成交付（新监督11/12，已删除）

完整TEST1000新6000CCD真实直接Changed-cell **.6915**，同PTstrictCPU clean仿真 **.8950**；相比旧CCD+DC .704低1.25pp，达到用户65–70区间，不是电子微调。保存新旧两组不覆盖。当前可展示直接链.5490→.5980→.6915→.7315，各自sim .895/.897/.895/.912；旧.704完整保留。
任务返回0/pipelinecomplete/adaptation explicitlydisabled，Python均退出/SDK StoplogOK Result0 FreeCStdLoadCti。auditPASS精确PT44e40243...3ebe，六层各1000PNG+收据总6000，minp99最低59/maxsat.021275<批准.15，phase与固定光学合同一致。report/audit/contract/deploymentidentity/progress五文件本地ccd_dc_noise075/physical备份，report SHA49ab981fb14096fbebd8c72f204803f169f99963e2705a8aef14b8f3e58aee69。bestlast与训练报告已备份，本轮GPU先前释放，旧有效CCD/PT全部保护，外部上传暂停。
监督openmoji-ccd-dc工具确认deleted，11/12实际完成，不再延长或新训。本次仍为TEST-selected开发口径非独立泛化；shot/mix元数据差异按上段源码解释，非严格sensor标定；本轮没有最终电子适配。

### 2026-10-05 北京14:29 温和CCD+DC全量采集实际核查（新监督10/12）

实际2700→5292PNG及同数CCD收据，五层各1000齐，vision_global292最新14:28:36。唯一任务267009/PID5412与11452保持，pipelineTEST，日志继续无完整report，无停滞迹象。不重复启动，剩708帧与终了审计，下一轮优先完整report/audit/相位信号合同/SHA本地备份/SDK释放及监督删除。累计10/12余2次，不报中途精度，旧结果/曝光合同保护。

### 2026-10-05 北京14:08 温和CCD+DC全量采集实际核查（新监督9/12）

唯一任务Running/267009、实际launcher5412/main11452精确新manifest/prefix保持。pilot24已完成，pipeline进入r2/test；全TEST实际2640PNG（语言router/expert各1000，global640）→日志643，最新14:08:29，持续实际增量，无异常或完整报告。最新p99=61/sat0，2000usGainX4wait240/flip_v不变，phase17650915...07a4有收据。继续原采集不重复启动，不报中途精度；pipelineelapsed42为阶段固定值不能当停滞。累计9/12余3次，完成核各层1000收据/report/audit/SHA备份释放SDK并删除监督，旧全部结果保护。

### 2026-10-05 北京13:49 温和CCD+DC新实拍已主动启动（新监督8/12）

上传session8047退出0，师弟PT SHA44e40243...3ebe逐字一致，无Python占SDK、新prefix不存在后注册/启动唯一OpenMoji_ObservedCCDDC_Noise075_1005。复用旧任务XML仅替换新manifest/prefix/log，无改源码/旧任务。新prefix editor16_observed_ccddc_noise075_20261005，单r2/capture-only/all，完整新6000TEST流程（先strictCPU六层桥接/pilot24）。Running仅为启动状态，须核实际Python/日志/收据，不冒称验收或实拍完成。累计8/12余4次，下一轮优先signal/phaseSHA/pilot及实际数量mtime/异常，完整后报告SHA本地备份及SDK释放删除监督。不TRAIN/电子适配，曝光合同和旧结果保护不变。

### 2026-10-05 北京13:29–13:32 温和CCD+DC备份/审计通过，权重传实验台（新监督7/12）

原7文件下载全部退出0，本地best SHA44e40243...3ebe/last9300be32...9c65与report一致，7文件齐。服务器CPU tensor审计PASS：best12/12phase更新，7/7alpha/frontend保护tensor不变。本地torch c10.dll初始化失败未改环境，改用原服务器venv审计成功。
新manifest本地ccd_dc_noise075/deployment_manifest.json已上传师弟weights/editor16_observed_ccddc_noise075_manifest_20261005.json；精确单r2/模拟.895/fixedalpha与best对应。best上传仍在exec session8047进行，禁止边传边加载；远端目标weights/editor16_observed_ccddc_noise075_best_20261005.pt，完成后必须Hash44e40243...3ebe。尚未注册/启动新采集，不能称已实拍。
下一轮优先确认8047退出0及远端SHA、无Python/SDK占用/newprefix不存在，再用旧任务XML仅替换manifest/prefix/log路径，注册新OpenMoji_ObservedCCDDC_Noise075_1005一次；prefix editor16_observed_ccddc_noise075_20261005/r2/capture-only/all，CPU桥接/pilot24后新6000TEST。源码未改，不需要Git同步，无TRAIN/电子适配，GPU已释放。累计7/12余5次，不重复上传/任务或覆盖旧结果。

### 2026-10-05 北京13:09–13:12 温和CCD+DC完训/备份交接（新监督6/12）

完整120epoch结束loss .137420，report complete，最佳GPU .896/e15，fresh strictCPU完整TEST1000 .8950；不是初始e0，尚无新实拍。best SHA44e40243d8802a1159c0e1597dde0472cafe0113b24140a58db21ccb27913ebe，last9300be320991175f709ca2b530b25cef1fe787d47efa86efc88b144b2ee99c65。nvidia-smi本轮GPU释放，仅他人GPU0PID1246190保留。
已开始7文件本地observed_ablation_20261005/ccd_dc_noise075备份，下载尚未完成：exec sessions68777/32588/8414/82932在运行，bestlast仍被下载进程占用，不可提前Hash/上传加载；小JSON部分完成。下一轮必须核传输结束与本地SHA，避免重复下载覆盖。CPU tensor审计命令因PowerShell引用转义未执行（并非模型错误），须安全重做12phase更新和alpha/frontend保护审计，不能声称已审计。
师弟只读确认无Python占SDK，旧r2manifest/任务动作已读取供精确新任务复用，但新PT未上传、新manifest/任务未注册。下一轮主动完成审计/备份→上传新best核44e40243...3ebe→新单r2manifest模拟.895/原fixedalpha→新prefix editor16_observed_ccddc_noise075_20261005，capture-only/all。不得运行旧manifest/prefix或重采旧CCD。旧任务OpenMoji_ObservedCCDDC_1005的EncodedCommand只替换精确manifest/prefix/log路径，可注册新OpenMoji_ObservedCCDDC_Noise075_1005，单SDKstrictCPU桥接/pilot/新6000TEST。累计6/12，目标区间不保证，无电子适配，旧结果保护。

### 2026-10-05 北京12:49 温和CCD+DC实际核查（新监督5/12）

实际epoch81→101/120，loss .148403→.139260；最新progress/log12:48:16，查询12:49:03，无异常或终了report。真实PID1090830精确原命令、UUIDe8837b85/8086MiB保持，他人GPU0保护。最高clean GPU开发TEST .896/e15，e100 .8585非实拍，尚未开始新CCD采集。剩19轮及freshCPU终审，原任务健康继续，不重复启动；下一轮优先核完整report、bestlast/phase与保护SHA、备份释放，并主动交接新6000TEST。累计5/12，仅写本handoff，旧结果及光学合同不变。

### 2026-10-05 北京12:29 温和CCD+DC实际核查（新监督4/12）

实际epoch61→81/120，loss .163709→.148403；最新progress/log12:28:11，查询12:29:01，无异常/终了report。真实PID1090830精确命令与UUIDe8837b85/8086MiB保持，他人GPU0保护。最高clean GPU开发TEST .896/e15，e80 .865非实拍，尚未新CCD采集。继续原任务，剩39轮及freshCPU终审后主动新实拍交接；累计4/12。仅写本handoff，旧结果及光学合同不变。

### 2026-10-05 北京12:09 温和CCD+DC实际核查（新监督3/12）

实际epoch41→61/120，loss .193947→.163709；最新progress/log12:08:04，查询12:08:59，持续约60秒/epoch，无异常或终了report。真实PID1090830精确原命令、UUIDe8837b85/8086MiB，GPU0他人1246190保护。最高clean GPU开发TEST .896/e15，e60 .8695非实拍，不据此判断实拍成功失败；尚未新采集。继续原训练不重复启动，剩59轮及freshCPU审计后主动交接新6000TEST；累计3/12，仅写本handoff不操作共享源码/Git事务。

### 2026-10-05 北京11:49 温和CCD+DC实际核查（新监督2/12）

实际epoch22→41/120，loss .234856→.193947；最新progress/log11:48:02，查询11:48:58，无异常或终了report。PID1090830精确原命令、UUIDe8837b85/8086MiB，GPU0他人1246190未动。最高clean GPU开发TEST仍.896/e15，e40 .8585不用于提前判断实拍；尚未新采集。保持原任务继续，剩79轮和strictCPU审计后交接；累计2/12。只更新本handoff，不修改任何源码/Git事务或旧结果。

### 2026-10-05 北京11:29 温和CCD+DC实际核查（新监督1/12）

唯一PID1090830/UUIDe8837b85命令精确，实际epoch1→22/120，loss .405988→.234856；history/log/progress最近11:29:03，查询11:29:15，无异常或终了report，GPU8020MiB、他人GPU0保留。最高clean GPU开发TEST .896/e15（非实拍），尚未开始新CCD采集。持续约59秒/epoch，健康继续同一任务，不重复启动或提前判成功失败。剩98轮及freshCPU审计后主动交接新实拍；累计1/12。旧全部结果保护，源码与光学合同未改。

### 2026-10-05 用户新授权：CCD+DC温和扰动单因素对照（已实际启动）

保留基线.549/CCD.598/CCD+DC.704/对齐.7315全部旧PT与CCD。用户希望另试CCD+DC .65–.70，不能挑差PT或改变光路压分。预声明只把observed_response noise_scale 1→.75：gain范围.925–1.075、低通混合上限.375、背景/随机噪声幅度乘.75；DC仍30%、grid0、pairedclean.5、一致性.05、初始01f7fc4a...eb05a/seed73/120epoch157step完整TRAIN5000均不变。检验温和响应代理的匹配，不保证实拍区间。
已在既有服务器t04工作树启动唯一PID1090830，GPU1 UUIDGPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d先验空闲后使用，实际8018MiB，GPU0他人1246190不动。输出runs/simulation/editor16_observed_ablation_20261005/ccd_dc_noise075及同级.log；实际e0完整TEST.895/status training，尚无新物理分数。
源码完全未改：Git9d6b24f01入口SHA5ae413bba6bebe34496e38e923877104f2243664e9373fe708bedf1445abf140，runtime3b956503+已保护overlay。原光电参数共同训练/12phase，alpha/frontend冻结，无新增架构/非decoder微调；TEST每5最高选开发/noTEST梯度/noVAL。
代码核验发现protocol noise_shot_fraction对observed_response登记0但实际代码有.01*scale*sqrt(max(code,0)*mean)信号相关Gaussian项，说明以源码为准，不能宣称read/shot真正标定；spatial_response_max_mix字段固定.5，新run实际.375。旧报告不覆盖，收尾审计须明确这两个元数据差异。
完成后严核history/log/epoch/CPU最佳重载/12phase更新/alpha frontend保护，备份bestlast报告协议SHA并释放GPU。best若e0不可当robust部署，须明确并测实际训练last。只该新PT于师弟既有rank64项目新manifest/prefix、仅r2/capture-only/all，strictCPU桥接0/pilot24后全新TEST1000六层6000，绝不复用旧CCD；不TRAIN不电子适配。单SDK/2000usGainX4wait240/ROI方向BMP不变，TEST15%饱和记录，暗p99<15有界诊断不盲重采。
有限20分钟监督最多12次，阶段完成/失败通知，正常静默，结束删除；源码Git-only无新分支worktree工程，外部上传暂停，其他最终封存。

### 2026-10-05 北京10:25–10:28 两组实拍最终交付（监督12/12，已删除）

r1 CCD完整TEST1000直接Changed-cell **.5980**，同PTcleanCPU **.8970**；r2 CCD+DC **.7040** / cleanCPU **.9000**。两组均为120epoch原光电共同重训后部署，无电子适配。保留对齐组cleanCPU .9120 / 直接 .7315及无trick基线.8950 / .5490。
直接实拍链为 **.5490 → .5980 → .7040 → .7315**。CCD达到用户.55–.65范围，CCD+DC高于.60–.70上沿.004，不人为降分。旧最终适配.8705属于另一PT，不混作本轮最终。
r1任务OpenMoji_ObservedCCD_1005返回0，pipeline complete，全部Python退出、SDK StoplogOK/Result0/FreeCStdLoadCti；六层各1000 PNG+1000收据共6000，capture_audit PASS，最低p99=61/maxsat=.030116，权重/相位及正常光学合同一致。r2已同样完整审计并释放。服务器本轮两GPU先前确认释放，不动他人GPU0。
r1精确PT SHA c7365fe084b94b64a731e756ce5cfcdf69cd126ee0491670db3081b780c5c761；report/audit/contract/deployment_identity/progress五文件本地observed_ablation_20261005/ccd_paired/physical与师弟SHA逐一一致，report SHA 281976e774a9546eaa280ccf4bda0167c8cb67e9da6f623aafb09db086b1b502。训练best/last/protocol/history/report及原CCD保护。
本轮12/12实际完成，openmoji-ccd-dc监督工具确认deleted；不再采集或启动微调。TEST选择为用户授权开发口径，非独立泛化；外部上传暂停，ABO/G2/G5及历史结果封存不动。

### 2026-10-05 北京10:05 实际核查（配对CCD/DC监督11/12）

r1实际2680→5296 PNG+同数收据，五层各1000齐/vision_global296最新10:05:52。
唯一PID6420/3916精确命令保持，任务267009/pipelineTEST，日志逐帧，无完整report或退出，健康继续最后704。
未重复采集/训练，不报中途精度；累计11/12，下一次必须核完整报告、audit/SHA备份/释放并实际交付删除监督；未完则如实说明不无限延长。
旧r2.704/对齐.7315及ABO/G2/G5保护，固定光学合同不变。


### 2026-10-05 北京09:45 实际核查（配对CCD/DC监督10/12）

r1唯一任务267009，实际6420/3916精确命令保持；pilot24审计PASS精确SHA c7365fe0...c761，各层minp99>=84/maxsat.026085。
全TEST实际2680 PNG+同数收据，language_router/expert各1000齐/global680→日志683，最近09:45:52，信号p99=159/sat.00281正常。
持续无停滞或异常，全报告尚无，不提前报r1实拍分数。pipelineelapsed42固定不作为卡死证据。
旧r2已交付.704且SDK释放，当前只有r1光路；累计10/12余2次，结束须全6000/report审计/SHA备份释放并删除本监督。


### 2026-10-05 北京09:25–09:27 r2交付并主动交接r1（监督9/12）

r2完整全TEST直接Changed-cell **.7040**，同PTcleanCPU **.9000**，无电子适配，较旧DC.5985提高10.55pp，略高目标.60–.70上沿.4pp，不挑差PT降低它。
pipelinecomplete/capture-only explicitlydisabled适配、任务返回0、Python23224/12536退出，SDK StoplogOK/Result0/FreeCStdLoadCti。
captureauditPASS精确SHAfcc7e5b6...69b97，六层各1000共6000，minp99最低60/maxsat.021612在批准.15内，phase合同一致。
report/audit/contract/deploymentidentity/progress五文件已本地observed_ablation_20261005/ccd_dc_paired/physical备份；原CCD留师弟保护。
r1权重上传完成远端SHAc7365fe0...c761核验通过，确认无Python占SDK、全新prefix不存在后唯一新OpenMoji_ObservedCCD_1005一次启动。
精确单组manifest weights/editor16_observed_ccd_manifest_20261005.json，prefix editor16_observed_ccd_20261005，r1_ccd/capture-only/all，桥接→pilot24→全新6000TEST，不TRAIN不电子适配不复用旧CCD。
已Running，后续须实际查pilot和全量增量/phase信号任务返回；不要重复启动。剩3/12监督足够优先核新r1完成备份/设备释放并交付删除监督。
旧对齐.7315及ABO/G2/G5不动，外部上传暂停。


### 2026-10-05 北京09:05 实际核查（配对CCD/DC监督8/12）

r2全TEST3000→5568 PNG+同数收据，五层各1000齐，vision_global568最新09:05:52。
实际23224/12536保持，任务267009、日志逐帧，无异常；曝光/相位/方向保持，最新p99=184/sat.000525正常。
尚无完整report，不报精度；r1权重与单组manifest开始预传准备，不占SDK，不在r2退出前启动r1。
累计8/12；上传可能较慢，必须确认进程完成及远端SHA c7365fe0...c761后注册新任务，禁止边传边加载。


### 2026-10-05 北京08:46 实际核查（配对CCD/DC监督7/12）

唯一r2任务Running/267009，真实23224/12536与精确manifest/prefix保持。
全TEST实际124→3000PNG及3000CCD收据，语言router/expert/global各1000齐，最新global08:46:02，正切换视觉层。
新增空视觉目录无收据导致诊断LastWriteTime.ToString空值提示，只是盘点格式化问题不是采集失败；原进程和日志持续。
无全量report，不报实拍分数；pipelineelapsed42阶段固定不作为停滞证据。
不重复SDK，r1待r2完整审计和释放后交接，累计7/12，旧.7315保护。


### 2026-10-05 北京08:25–08:26 实际核查（配对CCD/DC监督6/12）

唯一OpenMoji_ObservedCCDDC_1005 Running/267009，实际Python23224/12536命令身份精确r2新manifest/prefix/capture-only/all。
pipeline由pilot进入r2/test；全量首层language_router实际56→124PNG及124收据，最新08:26:17持续，尚无全量报告或实拍分数。
日志曝光2000/GainX4wait240/flip_v保持，最近p99=248/maxsat.005786在批准.15守卫内，无暗帧或异常。
未重复SDK/旧任务；r1待r2完整报告+六层收据及SDK释放后主动交接。累计6/12，不以pipeline elapsed42未刷新判断停滞。


### 2026-10-05 北京08:17–08:25 实际交接（配对CCD/DC监督5/12）

本次实际执行客户端已08:17，不冒称heartbeat18:55UTC即查询时间；两任务完整120轮结束、PID880622/880878退出，两UUID compute释放，GPU0他人1246190保留。
r2 strictCPU完整TEST1000 .9000，best e15 GPU.901，SHA fcc7e5b6cb2347c9ace930d1fb99db256ebdea40109dfe6edc70321c67b69b97；last eee65e21fb56c3051e1c16be27fdaec2bddfeb5e5db5e66385b9624d5a55e65f。
r1 strictCPU .8970，best e15 GPU.899，SHA c7365fe084b94b64a731e756ce5cfcdf69cd126ee0491670db3081b780c5c761；last57c78dc504bf19df0dd6dc7553edbba81ced292d2717329ea4c73c5f9d9f364f。
两组best/last/protocol/history/report/resolved_config/progress本地observed_ablation_20261005备份，四PT SHA与服务器报告逐一一致。
实际CPU tensor审计两best各12/12原phase确实更新；7个frontend/alpha保护tensor与初始不变。
师弟已有入口SHA0adc3fc6保持，无Python运行占用。r2 PT传输曾未完成触发SHA前置守卫，未启动SDK；待上传完成后远端SHA通过才注册/启动唯一OpenMoji_ObservedCCDDC_1005。
新manifest weights/editor16_observed_ccddc_manifest_20261005.json只r2；prefix editor16_observed_ccddc_20261005，capture-only/all，无TRAIN或decoder微调。
任务已Running，新selftest/pilot/全TEST流程待实际核验，不以Running称成功，不重复启动。
CCD r1 PT尚未上传到师弟，单组manifest已本地ccd_manifest.json，r2全6000收据/report齐并SDK释放后主动上传核SHA再注册新r1任务，不能用双组manifest（入口要求manifest精确对应selectedgroups）。
本地deployment_manifest.json双组汇总不用于采集；远端editor16_observed_ablation_manifest_20261005.json也是非运行汇总，避免误用。
累计5/12，旧.7315/ABO/G2/G5封存，曝光合同/外部上传暂停不变。


### 2026-10-05 北京02:37 实际核查（配对CCD/DC监督4/12）

首SSH握手No existing session失败，安全只读重连成功，未重启训练。
r2实际62→82/120、loss .15040121；r1实际63→84/120、loss .13026774。
日志/进度02:36:34/37更新，真实PID880622/880878均Rl，原UUID对应8086/8006MiB，GPU0他人保留。
最高clean GPU开发TEST仍.901/.899/e15，未有终了报告或新实拍，不冒称训练或实拍完成。
剩38/36轮与freshCPU审计，健康继续原任务；累计4/12，随后主动备份释放与r2→r1新光测交接。


### 2026-10-05 北京02:16 实际核查（配对CCD/DC监督3/12）

r2 CCD+DC实际48→62/120，loss .16848942；r1 CCD49→63/120，loss .14222226。
最新progress02:16:18/02:15:53距查询13/38秒，日志持续、无异常或终了报告，尚无新实拍。
最高clean GPU TEST仍.901/.899均e15，是开发仿真不是物理结果；不据此宣布改善成功或失败。
真实PID880622/880878与原UUID/命令保持，各8086/8006MiB，GPU0他人1246190保护。
剩58/57轮及CPU审计，继续原任务，不提前重训/部署或占SDK；累计3/12。


### 2026-10-05 北京02:02 实际核查（配对CCD/DC监督2/12）

r2实际22→48/120，loss .18626713；r1实际22→49/120，loss .15706814。
日志/进度持续更新，最近02:02:00/05距查询15/10秒，无异常/提前退出，未生成终了报告或开始新实拍。
最高clean GPU TEST仍r2 .901/e15、r1 .899/e15，仅仿真开发指标，不判断实拍成败。
真实PID880622/880878、命令与原UUID对应保持，占用8086/8006MiB，GPU0他人1246190保护。
按约60秒/轮健康推进，余72/71轮及freshCPU审计；不重复任务、不提前部署中途权重。
累计2/12，训练完主动备份释放并先r2后r1全新光测，旧.7315不动。


### 2026-10-05 北京01:36 实际核查（配对CCD/DC监督1/12）

两组实际epoch1→22/120，日志/历史/进度持续递增，最近progress01:35:31/26，无异常。
r2 CCD+DC loss .23854395，暂最高clean GPU TEST .901/e15；r1 CCD loss .20614039，暂最高.899/e15。
这些仅训练中仿真开发数，不是新实拍效果，两组尚未采集。
真实PID880622/880878保持原命令，分别4090 UUIDe8837b85/4d8bfdb9占用8020/8006MiB；GPU0他人1246190未动。
每轮约60秒，余98轮，健康继续原120轮，不重复启动/提前拿中途PT交接。
本轮监督累计1/12，完成后按上一授权先r2后r1主动新实拍交接，.7315版本封存不动。


### 2026-10-05 新授权：保留 .7315，配对CCD/DC去对齐同预算重训已启动

用户保留 observed alignment .912仿真/.7315直接实拍，要求同方法重训去对齐CCD+DC目标直接.60–.70，再CCD单独目标.55–.65。区间是目标，不挑差PT或造低baseline。
已有发布9d6b24f01入口SHA5ae413bb...bf140不改源码，运行原t04 worktree HEAD3b956503+保护overlay。
两个独立120epoch/完整TRAIN5000/157step、seed73、共同初始01f7fc4a...eb05a，从初始重训非73.15PT续调、非末端decoder适配。
均 observed_response/noise_scale1、同TRAIN-only sensor_audit、paired_clean_weight.5/consistency.05；原光电12phase共同训，alpha/frontend冻结，无新推理层。
输出 runs/simulation/editor16_observed_ablation_20261005：ccd_dc_paired(r2) PID880622/4090 UUIDe8837b85；ccd_paired(r1) PID880878/4090 UUID4d8bfdb9。
只移除grid，两组grid_probability0；CCD单独再去DC，其余学习率/预算/方法保持。GPU0他人1246190未动。
已核实际PID→UUID；r2初始化e0=.895正常，r1初始化中。不得仅PID称训练健康，下一核验实际epoch/loss/文件mtime/log/协议配置差异。
TEST每5最高选开发、不进梯度/noVAL；若best仍e0不能当训练robust结果，必须说明并光测实际训练last，不以干净仿真判实拍失败。
训练完strict CPU、12phase/alpha/frontend审计、bestlast/history/report/SHA本地备份及GPU释放；先r2后r1在既有bench项目新prefix分别桥接/pilot/全新6000TEST，capture-only不做decoder适配、不复用旧CCD。
正常曝光2000usGainX4wait240/ROI方向BMP不变、TEST15%饱和记录/暗帧守卫保留、单SDK。旧结果/ABO/G2/G5封存、外部上传暂停。


### 2026-10-05 用户询问后完整交付：对齐新实拍 .7315，本轮结束

唯一OpenMoji_ObservedAlign_1004已Ready返回0，pipeline complete，主12008/launcher21568均退出，
SDK日志Stop log OK/Result0/Free CStdLoadCti；训练服务器GPU在上轮已实际核验释放。
六层各1000PNG+1000CCD收据，共6000完整，capture_audit PASS，精确best SHA d4481ef8...56f653与pilot相位一致。
同权重strictCPU仿真 **.9120**，全量直接真实TEST Changed-cell **.7315**，无TRAIN采集/decoder适配。
保留格.9880445354、sceneexact.517；曝光2000/GainX4wait240/方向ROI/BMP不变。
直接比无trick .5490高18.25pp，比旧对齐.5730高15.85pp，比上一对齐.5810高15.05pp，已突破.60。
仍距自身仿真18.05pp；本轮没有新的最终微调分数，不能将旧.8705套在新PT上，不能称全链最终恢复达标。
report含1000逐样本与audit/contract/deployment_identity/progress五文件已本地observed_alignment/physical备份，
大小SHA两端逐一一致；report SHA5df5441f3f46fb7871488c70f446548fe64afdc6564201b11d8b1d0a489d10a5，
audit SHAd7701addad5fb8fbe1d444382b99943abdbb7b7c09a935e5f960d1eec4431a77。
原CCD/收据留师弟，best/last/训练历史已本地完整备份，ABO/G2/G5/DC旧最终封存，外部上传暂停。
此前quiet只表示该次监督未通知，不是实验完成；00:24仍4876，现查询已完整结束。
本轮实际交付后删除openmoji监督，不追加无限候选/重复采集/电子适配，后续需用户授权。

### 2026-10-05 北京00:23–00:24 实际核查（对齐监督9/12）

唯一OpenMoji_ObservedAlign_1004仍Running/267009，主12008/launcher21568命令身份保持。
实际全TEST2284→4868→4876 CCD PNG及同数收据递增；四层各1000齐，vision_expert876，
vision_global待采，最近收据00:23:55，日志逐帧持续无异常，全量report尚无，不提前报实拍分数。
遍历4868收据合同/phase/信号bad0，minp99=47/maxsat=.00573782；曝光2000/GainX4wait240/flip_v不变。
不重复SDK或重启服务器训练；继续最后一层及完整固定回放，随后备份审计/逐样本SHA并确认释放。
本轮累计9/12余3次，保持有限监督及旧结果封存、外部上传暂停。

### 2026-10-05 北京00:03–00:04 实际核查（对齐监督8/12）

唯一OpenMoji_ObservedAlign_1004保持Running/267009，主12008与launcher21568命令身份一致。
全TEST实际68→2172→2284 PNG及对应CCD收据递增，语言router/expert各1000齐，global284；
实际最新收据00:04:34，日志逐帧更新，未见停滞或异常，全量report尚未生成，不提前报精度。
遍历当时2276张CCD旁json收据（不是文件名receipt筛选），合同/phase/信号bad0，
minp99=47/maxsat=.00277919，曝光2000/GainX4wait240/flip_v与pilot相位一致。
首次*receipt*.json计数0只是命名筛选错误，已定位并使用ccd各层sample json正确核验，无数据丢失。
pipeline状态elapsed42只在进入TEST时写入，不作为停滞证据；真实文件增量正常。
不重启训练/SDK或混旧CCD，继续唯一流程至六层6000及完整回放审计，累计8/12余4次。

### 2026-10-04 北京23:43–23:48 主动交接（对齐监督7/12，训练完成→新实拍已启动）

align_paired完整120轮及fresh CPU TEST1000完成，elapsed7658.3秒，selected e15，
CPU Changed-cell **.9120**（GPU .9130），保留格.99355798/sceneexact.754。不是实拍成绩。
best 12个原phase全部确实更新；alpha/frontend tensor与精确初始一致，无新增架构或decoder适配。
原PID610089退出，原4090 UUID无compute进程；GPU0他人1246190保留。
best/last/report/history/protocol/config/progress七文件已本地observed_alignment备份，大小SHA逐一两端一致。
best SHA d4481ef804dbd8f27f2f24f98c08dc7bf161b3137cefcc2613310291fe56f653；
last SHA960e9b39583c5d407ea914e863eec2de1ce788ce2f6a5c644ec763f105886253；
report SHA8e7e591bc54271425a53f6a4ece76c3c450c1c3214a5f051555a9dd73c67751f。
本轮best为训练后e15，无需部署e0，保存last不删除。

已向既有师弟项目传精确best+manifest并远端SHA验证；没有源码写入，既有入口SHA0adc3fc6...bccb62保持。
manifest weights/editor16_observed_align_manifest_20261004.json，SHAe053813ef3042e467b187392518f80c799149ce6d0f029b38f7e0b79e7b23065。
确认无OpenMoji/ABO Running和Python残留后，唯一新任务 **OpenMoji_ObservedAlign_1004** 一次启动，
主12008/launcher21568，Running返回267009；不再启动该任务。新prefix editor16_observed_align_20261004，
仅r3_ccd_dc30_grid/capture-only/all，无TRAIN/末端适配，全新TEST6000。
理想六层selftest已通过并进入pilot，SLM SDK正常连接；当前尚未核验完整pilot24或全量实拍分数。
随后实查selftest error0/replay3.8147e-6 PASS，pilot24/24审计PASS，六层minp99
195/65/61/229/139/115、最大饱和.0046393，权重/相位/曝光合同一致，已串行进入全量TEST。
后续必须查新pilot收据/信号/phase及完整六层1000PNG/收据实际增量，不仅PID；
正常合同不改、暗帧守卫保持、TEST15%饱和记录，保护旧CCD/权重，外部上传暂停。
本轮累计7/12，余5次；下一监督仅跟进新硬件任务，不复活服务器训练。

### 2026-10-04 北京23:23 实际核查（对齐监督6/12）

align_paired已114/120轮，前轮95后新增19；history115条，loss .1398694098，
累计7215.4秒，progress距查询54秒（每轮约61秒），日志无异常，尚无终了report。
第105/110轮clean TEST GPU .8735/.8725，最高仍训练后e15 .9130，未获得新实拍结果。
实际PID610089 Rl、原命令/输出保持，4090 UUID e8837b85占用9844MiB；GPU0他人未动。
剩最后6轮及freshCPU复载仍在同进程，不能提前释放或把即将完成说成已完成。
下一次必须主动核验report/保护alpha/相位/bestlast SHA、本地备份和GPU释放，然后新manifest/PT光测交接，
不能再停在只读训练状态或重启旧任务；本轮累计6/12，剩6次有限监督不延长。

### 2026-10-04 北京23:03 实际核查（对齐监督5/12）

align_paired已95/120轮，较前轮76新增19；history96条，loss .1441599081，
累计6037.5秒，progress距查询38.7秒，日志持续无异常，尚无终了report。
第85/90/95轮clean TEST GPU .8665/.8820/.8765，暂存最高仍e15 .9130；不当作实拍结果。
实际PID610089 Rl、原命令/输出一致，4090 UUID e8837b85占用9844MiB；GPU0他人保持。
继续最后25轮及freshCPU完整复载，无重复训练/SDK。下一轮重点核验终了报告和GPU释放，
随后备份best/last及精确SHA，主动交接新权重的桥接/pilot/全新6000TEST，不用旧CCD或电子适配替代。

### 2026-10-04 北京22:43 实际核查（对齐监督4/12）

align_paired已76/120轮，比前轮57新增19；history77条，loss .1557196376，
累计4852.6秒，progress距查询15.6秒，日志连续无异常，无终了report。
第65/70/75轮clean TEST GPU .8665/.8775/.8835，暂存最高仍e15 .9130；没有新实拍精度。
真实PID610089 Sl、命令与输出身份保持，原4090 UUID e8837b85占用9844MiB；GPU0他人未动。
继续原120轮，无重复训练/SDK任务，不用loss下降或中途仿真宣称实拍改善。
下一步完成严格CPU、best/last身份和相位保护审计后主动交接新6000TEST光测。

### 2026-10-04 北京22:23 实际核查（对齐监督3/12）

align_paired已57/120轮，前轮38后新增19；history58条，loss .1753176395，
累计3662.5秒，progress距查询32秒，实际日志持续、无异常，无终了report。
第45/50/55轮clean TEST GPU .8805/.8625/.8730，最高仍e15 .9130；不是实拍结果。
实际PID610089 Rl、完整命令与输出一致，4090 UUID e8837b85占用9844MiB，GPU0他人未动。
健康继续原120轮，未重复训练/采集或提前用中途best冻结；结束后主动交接新全量光测。

### 2026-10-04 北京22:03 实际核查（对齐监督2/12）

align_paired已38/120轮，比前轮19新增19；history39条，loss .1990417022，
累计2439.1秒、progress距查询32秒，日志连续无异常，无终了report。
第30/35轮clean TEST GPU .8600/.8875，暂存最高仍e15 .9130；仅中途开发仿真，尚无新实拍。
真实PID610089 Rl、命令/目录身份保持，原4090 UUID e8837b85占用9844MiB，GPU0他人未动。
保持原120轮，不重复启动、不提前释放/占SDK、不用中途分数判断实拍成败。
完成后仍主动核验freshCPU和训练后mask身份、备份权重并以新CCD实际部署验证。

### 2026-10-04 北京21:43 实际核查（对齐监督1/12）

唯一align_paired正式训练19/120轮，比启动核查epoch1新增18轮；history20条，
最新loss .2479039888，累计1225.9秒，progress/history距查询61秒（每轮约63秒），日志连续无异常。
第15轮clean TEST GPU开发值.9130，暂存最高已是训练后的e15，不是初始e0；
这仅是中途仿真值，尚无最终strictCPU报告或新实拍成绩，不据此称robust部署提高。
实际PID610089 Rl、完整命令/输出身份一致，4090精确UUID e8837b85占用9844MiB；
入口SHA5ae413bb...bf140保持。GPU0他人1246190未动，没有重复启动或提前占SDK。
按既定120轮继续充分训练，终了核验best/last保护/相位/strictCPU与备份释放后主动交接新全量实拍。

### 2026-10-04 北京21:23 用户授权继续对齐组：正式训练已启动

用户在 observed_response CCD last 实拍 .4830 后要求先继续对齐组。
不是末端电子适配；同一初始 editor16/decoder64，原光电参数与12相位共同训练，alpha/frontend冻结，无新增推理层。
采用已发布 Git 9d6b24f01 的 observed_response + DC30 + TRAIN概率.5的17→8→17插值，
clean/noisy配对监督各.5及一致性.05。传感响应仍是TRAIN统计约束的代理模型，未声称真正read/shot/PSF标定。
4090空闲UUID GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d 上 smoke 已完整退出，
strictCPU TEST1000 .895（初始e0，仅加载预检，不称新mask改善），alpha保护通过。
唯一正式PID **610089** 已启动并核对实际GPU UUID/9842MiB，GPU0他人1246190未动。
服务器既有t04工作树输出 runs/simulation/editor16_observed_response_20261004/align_paired，
日志同级align_paired.log；协议120epoch×157steps，完整TRAIN5000梯度、TEST1000每5选择开发，无VAL/noTEST梯度。
精确初始SHA01f7fc4a8de4f20901fae09e8be55c67fce2177db0a7ed3975a7fc90853eb05a，
入口SHA5ae413bba6bebe34496e38e923877104f2243664e9373fe708bedf1445abf140；原runtimeHEAD3b956503+保护overlay保持。
完成保留best/last/history/protocol/report/SHA并释放GPU；若干净最高仍e0，不能用e0代替robust mask：
以实际训练last做strictCPU和12相位更新审计，明确其不是最高干净仿真PT，再部署新权重实测。
只在既有师弟rank64项目用新manifest/prefix、r3_ccd_dc30_grid/capture-only，桥接/pilot后全新TEST6000CCD；
不复用旧CCD、不采TRAIN、不用电子微调掩盖，曝光/ROI/方向合同保持。
目标直接实拍优于旧对齐.5730/.5810并尽量突破.60，不保证、不强凑单调。
新有限监督每20分钟最多12次，仅本轮训练→实际光测，完成交付即删除；旧任务不复活、外部上传暂停。

### 2026-10-04 北京20:58 实际完整交付（新监督3/6）

last120新TEST6000采集及固定CPU回放完成，任务Ready返回0，pipeline complete，
主6264/launcher3204均退出，无Python残留，SDK日志Stop log OK/Result0/Free CStdLoadCti。
六层各1000PNG+1000收据，审计PASS且精确SHA ffadf195...c744b，各层相位同pilot；
minp99 254/65/63/255/223/213，最大饱和.039158，原2000us/GainX4wait240/方向ROI/BMP不变。
真实直接Changed-cell **.4830**，同权重strictCPU仿真 **.8685**；保留格.9986613，sceneexact.352。
比无trick直接.5490 **低6.60pp**，比旧CCD.5175 **低3.45pp**；
仅这次真实结果支持此候选没有改善直接部署，不能泛化为所有CCD建模无效，原因尚未唯一定位。
没有TRAIN/decoder适配，不用旧CCD评新上游。旧DC.5985/最终.8705、ABO/G2/G5仍封存。
report（含1000逐样本）、capture_audit/contract/deployment_identity/progress共5文件本地physical备份，
逐一两端SHA匹配；首report下载SSH连接短暂失败，单文件重试成功，未重复采集。
report SHA de293f7b04dbfb50645b9d2945a9a7fa69625886f93f477717493628a5b1b233，
audit SHA86f37ed44dd66a3dd7d6ea420c2908e58b11fd940a10f126b189b2dc0de86d6e。
全部原CCD/收据保持师弟原目录，权重/训练历史已备份，外部上传暂停。
本轮实际光测交付完成，删除openmoji-mask监督，不无限复活旧任务/追加TEST扫分。

### 2026-10-04 北京20:38 实际光测核查（新监督2/6）

唯一last120任务Running/返回267009，主6264/launcher3204命令身份一致。
实际 **5544/6000 PNG及同数收据**：language三层、vision router/expert各1000齐，
vision_global544持续采集，最新mtime20:38:50；较前轮3028新增2516，不是只看状态JSON。
日志vision_global test_00543，phase SHA07ea8be6...2eb6987同pilot，p99=225，
饱和.00184，曝光2000/GainX4wait240/flip_v不变，无暗帧或异常停机。
尚无完整report和最终精度；继续唯一任务完成最后一层及固定回放评价，不重复启动/提前报数。

### 2026-10-04 北京20:18 实际光测核查（新监督1/6）

唯一OpenMoji_ObservedResponse_Last120_1004实际Running，返回267009为运行中，
主6264/launcher3204命令与last120 manifest/prefix/capture-only一致，无新任务。
实际全量CCD **2952 PNG +2952收据**（前轮48后明显增量），最新mtime20:18:44；
日志language_global test_00954，当前帧p99=180/上一帧171，饱和.00377/.00115，
相位SHA ba3bcfd5...2f55b与pilot一致，曝光2000/GainX4wait240/flip_v保持。
pipeline_status仍r1_ccd/test且其elapsed41.7为进入阶段时写入，不当成停滞；
实际文件与逐帧日志持续更新，无异常/暗守卫停机。随后逐层复核：language router/expert/global
各1000PNG+1000收据齐，vision_router28+28，合计3028，已正常交接第四层。
完整六层尚未结束，无最终实拍精度。
保持本任务继续，不重复启动、不复用旧CCD、不以仿真初始best门槛中止。

### 2026-10-04 用户纠正后新授权：已训练last实际光测已启动

覆盖19:02的停止结论：干净仿真最高选到e0不能判定robust mask实拍失败，用户要求立即继续实际部署。
现有仅best/last，best全部模型tensor等同初始，真正训练120轮last的12原phase均更新；
中间PT未保存，不能声称last是训练检查点中仿真最高，也不再以超过初始仿真为光测准入。
last精确SHA ffadf195cfecdc0ffd7d17e33d6d340ffbf008e475951a18c4bce20bbb0c744b，
新独立strictCPU完整TEST1000 **.8685**（末轮GPU .870），报告last_cpu_report.json与manifest已本地备份。
CPU审计首次SSH等待超时未留下报告，确认无该进程后有界后台审计完成退出，无GPU占用。
权重artifact传师弟SHA通过；既有Git捕获入口未修改，源码同步没有裸文件上传。

唯一任务 **OpenMoji_ObservedResponse_Last120_1004** 实际Running，主PID6264/launcher3204；
既有项目runs/editor16_observed_response_last120_20261004_r1_ccd_test1000，全新6000TEST CCD。
manifest weights/editor16_observed_response_last120_manifest_20261004.json，SHA
ccff725189494b98c125b3d1a7e001a635558dcb1779dc458693139d4b558120。
六层理想桥接error/replay_error **0/0 PASS**，pilot **24/24 PASS**，各层minp99
27/111/112/255/234/226，最大饱和.01679；原曝光2000/GainX4wait240/方向ROI/BMP不变。
已经进入全量，实际48 PNG且日志language_router test_00048，帧p99=255，饱和约.01308，
不是只注册或只看PID。capture-only无TRAIN或decoder适配，完成后实际数与基线.549/旧CCD.5175比较。
计划任务CIM注册报XML错误两次但未启动，改用已验旧任务XML仅替换新manifest/groups/prefix成功，
原任务不动，当前仅一次新任务，无重复SDK。暗帧仍停，不调曝光或造baseline。
恢复20分钟有限6次实际监督 **openmoji-mask**，正常静默，完整后审计备份释放并交付/删除监督。
旧best/last/CCD、ABO/G2/G5/DC及最终.8705封存，外部上传仍暂停。

### 2026-10-04 北京19:02 完整核验交付（监督20/24，配对response轮未获得新best）

正式120epoch完成，progress/report complete，累计9136秒；fresh CPU完整TEST1000最高.895，
selected_epoch=0。严格比较best与精确初始模型所有tensor：差异0，虽然PT容器SHA不同，
不能把它当成新robust mask。last的12原phase确实更新，最后clean GPU .870；未证明实拍提升。
PID3803883退出，精确3090 UUID无该compute进程，GPU已释放，其他用户进程保持不动。
best/last/report/history/protocol/resolved_config/progress共7文件已本地observed_response备份并逐一两端SHA匹配。
best SHA84ba893d48d080ecd903f9e25c17d0538d4f534a31c80e0dc5a7d797b5270d95，
last SHAffadf195cfecdc0ffd7d17e33d6d340ffbf008e475951a18c4bce20bbb0c744b，
report SHA0486c2e5503ff77b55cd5cc3e04e7fbc08605851efb6b6441651f2aaa7ea428e。
本轮模型选模未通过新候选门槛，不重采初始best、不换成更差last凑结果、不追加电子适配掩盖。
完整报告告知用户本次response+paired目标没有得到新最高clean权重，直接精度未新增，旧最终.8705保留。
本轮有限充分尝试结束；停止本监督，后续需基于本轮失败重新确定有限实验，不复活旧任务或无限扫TEST。

### 2026-10-04 北京18:42 实际核查（监督19/24）

observed_response正式训练 **106/120 epoch**，前轮90后新增16轮；history107条，
loss **.1256895427**，progress/history距查询53.1秒，累计8002.2秒，日志持续无异常。
第95/100/105轮clean TEST .8690/.8635/.8700，最高仍初始e0 .895，无终了report或新实拍精度。
实际PID3803883 Rsl、命令及输出身份一致，精确3090 UUID6dcca91a-8e08-1a50-9aa6-81defeaed50b
占用7856MiB；其他GPU任务未触碰，未重复启动训练/SDK。保留原定120轮完成，不改选模规则。
下一轮重点检查终了freshCPU报告/best/last与GPU释放，备份完整历史和SHA；
若最终最高仍初始e0，如实报告本轮未得到新的最高clean权重，不重采原权重冒充改进。

### 2026-10-04 北京18:22 实际核查（监督18/24）

observed_response正式训练 **90/120 epoch**，比前轮75新增15轮；history91条，
loss **.1278159266**，progress/history距查询61.1秒（正常每轮约75秒），累计6801.1秒。
第80/85/90轮clean TEST .8645/.8655/.8705，最高仍初始e0 .895，尚无终了报告或新实拍结果。
实际PID3803883 Rsl、完整参数/输出身份一致，3090精确UUID6dcca91a-8e08-1a50-9aa6-81defeaed50b
占用7856MiB；日志持续无异常，其他任务GPU进程未触碰，没有重复训练/采集。
保留原定120轮，结束后完整审计备份；若最高仍e0，如实交付没有新最高clean权重，
不得重拍原初始权重伪装新robust成果或用末端微调掩盖本轮建模效果。

### 2026-10-04 北京18:02 实际核查（监督17/24）

observed_response正式训练 **75/120 epoch**，比前轮59新增16轮；history76条，
loss **.1332194043**，progress/history距查询2秒，累计5659.3秒，实际PID3803883 Rsl。
第65/70/75轮clean TEST .8640/.8735/.8705，最高仍初始e0 .895；尚无终了report或新实拍结果。
日志持续更新无异常，精确3090 UUID6dcca91a-8e08-1a50-9aa6-81defeaed50b占用7856MiB；
其他用户GPU进程保留，不重复启动SDK/训练。原定120轮继续，未把loss下降称为性能提高。
结束后完整备份并strict CPU审计；如果最高仍初始，不冒充产生新的robust权重或重采原权重凑进展。

### 2026-10-04 北京17:42 实际核查（监督16/24）

observed_response正式训练 **59/120 epoch**，前轮45后新增14轮；history60条，
最新loss **.1474832462**，progress/history距查询14.2秒，累计4454.3秒。
第50/55轮clean TEST .8700/.8675，最高仍初始e0 .895，无终了报告或新实拍结果。
PID3803883实际Ssl、完整参数/输出一致，精确3090 UUID6dcca91a-8e08-1a50-9aa6-81defeaed50b
占用7856MiB；日志逐轮持续，无停滞/异常。未重复启动，不触碰其他GPU进程或SDK。
继续原120轮充分训练；最终若仍选初始，明确该轮没有新的最高clean权重，不将loss降低当成部署提升。
旧CCD/权重/报告均保持，后续新相位需严格身份审计和全新CCD验收，不以旧CCD替代。

### 2026-10-04 北京17:22 实际核查（监督15/24）

现有observed_response正式训练 **45/120 epoch**，比前轮27新增18轮；history46条，
loss **.1629889978**，progress/history距查询31.4秒，累计3406.7秒，日志连续无异常。
clean TEST第35/40/45轮分别.8825/.8610/.8680，最高仍初始e0 .895；
不能把loss下降说成准确率或实拍提高，完整充分训练尚未结束，无终了report。
PID3803883实际Ssl、命令身份一致，精确3090 UUID6dcca91a-8e08-1a50-9aa6-81defeaed50b
占用7856MiB；其他用户GPU进程未触碰。未新启SDK、未重复旧采集或额外训练。
继续原120轮；若最终best仍e0，必须报告此轮未获得新的最高clean PT，不能以原权重冒充robust训练成果。
后续仅据最终权重身份/相位更新审计安排桥接、pilot和新CCD，不用旧CCD评改变的上游。

### 2026-10-04 北京17:02 实际核查（监督14/24）

observed_response配对正式训练 **27/120 epoch**，比前轮12新增15轮；history28条，
最新loss **.1952970689**，progress/history距查询7.5秒，累计2045.8秒，日志持续增量无异常。
第20/25轮clean TEST分别.8800/.8775；最高仍初始e0 **.895**，不能称新权重改善。
PID3803883实际Ssl、命令和既有输出一致；3090精确UUID6dcca91a-8e08-1a50-9aa6-81defeaed50b
占用7856MiB，无终了report，无新部署/实拍数。其余GPU进程属其他任务，未触碰。
继续原120轮充分训练，不重复启动或提前用初始best充当robust结果；结束后核验相位更新、
strict CPU与best/last完整备份，再按新权重身份决定实际光路验收，旧结果保持不动。

### 2026-10-04 北京16:42 实际核查（监督13/24）

新observed_response配对训练实际推进至 **12/120 epoch**，history13条，最新loss
**.2422458233**（epoch11 .2475866），progress更新时间距查询42.9秒，累计训练918.9秒。
PID3803883实际Rsl、命令与输出目录一致，唯一本任务3090 UUID
GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b占用7856MiB，日志连续且未见异常。
最高clean TEST仍 **.895、selected epoch0**，尚未形成超过初始的完整结果；无终了报告或新实拍分数。
不重复启动、不提前部署早期PT、不碰其他GPU任务；原两组已结束输出和旧CCD保持封存。
下一步继续核查真实epoch/日志增量，终了执行freshCPU审计与best/last/SHA备份后再新相位全量实拍。

### 2026-10-04 北京16:22 主动推进（监督12/24，新充分重训已启动）

Git9d6b24f01正式发布observed_response TRAIN-only代理与原mask配对训练，服务器旧入口先核验
3c265864...e68f21f与d6a27fd5c一致，再仅restore本任务训练入口和test；其他overlay/他人任务不动。
空间核15混合比例0–.5（不把候选边界说成精确PSF）；按TRAIN各层median mean/p01映射camera code，
背景0–p01、增益.9–1.1，read/shot .01仍明确假设；[0,1]ADC截顶+8bit STE量化，映回原仿真单位。
专家/global用原detector phase_support明确识别对应层、router独立参数；六层参数来自只读TRAIN统计。
正常eval关闭全部扰动、不改推理图，不加电子层/不改曝光BMP/不加DC到CCD组。
零scale精确identity、零/常量输入finite梯度、paired teacher detach单元测试2/2 PASS。
sensor audit artifact两端SHA a037d2f46dd5f9c7b41a5b48899b77f66d4b468a70c47a73bcf8d9ed78800916。

新完整smoke PID3795264已完成退出，strictCPU最高.895（原e0，不能称提升），alpha保护PASS；
两step混合loss .39905、e1 clean GPU .8585，说明强扰动初期有性能代价，充分训练必须审计恢复情况。
启动正式前确认该3090 UUID无compute进程，未抢GPU0/他人/A100。
当前唯一新正式训练PID **3803883**，GPU **GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b**（3090），
服务器既有t04工作树runs/simulation/editor16_observed_response_20261004/ccd_paired，日志同级ccd_paired.log。
精确初始01f7fc4a...eb05a/editor16/decoder64，alpha/frontend冻结、原光电/相位共同120epoch完整TRAIN5000；
paired clean权重.5/一致性.05，seed73，完整clean TEST每5最高选development、无TEST梯度/noVAL，非末端适配。
下一轮需实际epoch/loss/history/mtime/异常/PID→UUID核验，不重复启动；早期最高不提前封存部署。
终了freshCPU best/last/report/protocol/history/SHA备份释放，再仅新CCD组strict桥接/pilot/6000新TEST，
不重采已结束两组或旧DC、不用旧CCD评新相位，benchmark normal合同不变。
这是TRAIN观测约束的联合response+paired目标改进，不是单因素noise对照或完整传感器标定；没有新实拍精度。

### 2026-10-04 北京16:02 实际推进（监督11/24，空间响应有限筛查完成）

本轮仅T04同输入诊断入口增加预声明box响应kernel1/3/7/15，Git81b4bf8d6发布并Git-only同步。
旧diag已结束无文件写入者；新独立runs/train_matched_sensor_spatial_20261004实际12TRAIN×六层完成，
72/72输入BMP身份匹配、保护上游不变、无SDK/GPU/梯度，全部旧CCD只读，没有重采或评新模型。
每操作前三个均匀样本中前2拟合、第三留出（8fit/4heldout TRAIN，不按TEST选response），
四预声明候选均在同输入上比较；所有层fit选择15，heldout residual RMS也降低：
language router .18293→.16085，expert .05750→.05537，global .05803→.05650；
vision router .18738→.15652，expert .16734→.14191，global .12308→.10833。
响应核15位于预声明边界，不无限加核扫分；数据支持“空间低通响应是残差的一部分”，
不证明完整PSF/读噪声标定，残差仍很大，不把改善残差说成改善实拍准确率。
报告已本地physical/train_spatial_response.json备份，诊断进程完成退出。

下一步已经有据可实现有限TRAIN-only detector response proxy：轻量空间响应混合（保持部分干净输入）、
按既有TRAIN各层动态范围/低分位约束背景与ADC截顶/8bit STE量化，再配对干净/扰动任务监督+.05一致性。
不得直接以affine截距充当纯暗电平，仍注明observed-bounded proxy非完整sensor calibration；
只改变训练扰动，不改normal inference/CCD曝光合同/不加电子层，原phase一起训。
先finite-gradient/identity clean/原.895严格复载smoke，再有限120epoch正式；目前尚未开始新正式训练。
仅重新计算噪声不足，必须保护干净任务性能与原mask扰动稳定性；不复活旧两组、不追加decoder适配掩盖。

### 2026-10-04 北京15:42 实际推进（监督10/24，同输入TRAIN响应诊断完成）

本轮写入仅T04 diagnose_splitrank_deployment.py，Git 9e4cbab8e/b0a03c077已测试语法并发布。
新增--scope train仅允许sealed original_g2，绕过TEST缓存选模、读取既有TRAIN CCD、无SDK/no参数更新。
师弟旧源码先LF归一SHA55293ca7...c0af完全匹配，再Git bundle仅restore该文件，未碰运行capture/tune。
首次诊断因为沿用ordinal五位sampleID而失败；已定位TRAIN是六位原始ID/非连续子集，
Git修复从data.records sample_id映射且只选已采TRAIN记录，保留失败目录train_matched_sensor_response_20261004。
修复后新retry1独立目录实际12样本（四操作各3均匀，不按分数选）×六层=72同输入比较完成，
全部输入BMP SHA与原收据72/72一致、保护上游SHA前后相同、响应identity gate PASS；进程结束。
原g2 TRAIN权重2acc2f58...194，绝未将旧CCD用于新模型正式评估。
报告本地physical/train_matched_response.json，两端SHA acd1ce061cd0d7da5b3261805cbb4b1c41ec18200000bbf9acc9029bf0aca00a。

实际各层PCC：language router/expert/global .395/.641/.658，vision .378/.469/.367；
剔除截顶像素后拟合的affine残差RMS约.184/.059/.060/.189/.165/.116（0–1CCD范围）。
输入身份正确但空间结构不完全匹配，简单白噪声或单一gain/offset不足以描述全部域差；
affine intercept混合背景与结构失配，不能直接当暗电平或纯read/shot噪声拟合参数。
后续优先用同TRAIN匹配输出分开检查空间响应/局部模糊/量化截顶对残差的贡献，
不能把低PCC结构误差都作为独立随机噪声注入，也不未经验证直接套affine slope/intercept训练。
配对训练入口smoke已通过、GPU已释放，但正式新重训尚未开始；不重复已结束原两组或宣称新精度。

### 2026-10-04 北京15:22 实查（监督9/24，原两组实拍完整交付；新授权继续）

唯一pipeline status complete、任务Ready返回0、原PID13068/4344均退出，日志Stop log OK/Result0/Free CStdLoadCti，SDK释放。
r3六层各1000PNG+1000收据=6000完整、capture_audit PASS，精确05126418...fbc4ca94；
clean CPU仿真.9265，直接真实TEST Changed-cell **.5810**/preserved.9893857/sceneexact.419。
比旧对齐.5730提高.8pp，但未突破.60，且低于历史旧DC.5985（噪声方案不同，不是单因素对照）。
新CCD .9155→.4940仍是失败的改善尝试；不能强凑递进，两组均未做末端适配。
原有效CCD和收据留师弟，新两组best/last已备份，旧DC/.8705及ABO/G2/G5全保留，外部上传暂停。

本地physical/r3_report.json SHA d4c5146e880b2f599736225cf13c1e2e57fd1cccd0a54aa7f6cc0ab901f92360，
r3_capture_audit.json SHA4d032a66bc66de6ee8273c84841dfdb27c7eaf88e21398e35615bd198a40bdfb，
comparison.json SHAb809a48b1c0865fbf8018d43ddcacf7bba8bd4f42f42d5bf0ef505417704f549，两端校验PASS。
audit六层最小p99=63，maxsat=.028973582，各相位与r3一致，无暗帧/信号失败。

用户新的传感器建模/robust mask授权尚未完成：配对入口smoke PASS已释放GPU，但还没有新正式重训。
已更新原openmoji-ccd监督为新授权诊断→有据模型→有限充分重训/实拍，保持20分钟、安静规则，
累计9/24只余15次，不无限延长、不复活已完成两组，不因原两组结束误删仍需推进的新授权监督。
下一轮优先推进同输入理想/真实TRAIN响应诊断与分层量化/截顶模型，不仅重复查看已完成任务。

### 2026-10-04 北京15:02 实查（新监督8/24，采集健康；配对smoke完成）

对齐r3实际3640→4760→4840收据；四层完整各1000、vision_expert756→840，最后vision_global待采。
遍历4840收据曝光/方向/相位/暗帧/15%饱和bad0，minp99=63/maxsat=.028973582；
真实PID13068/4344与唯一任务Running，最新收据持续写入，日志无Traceback/RuntimeError/AssertionError。
尚无完整对齐report，不报告临时精度、不重启或混旧CCD。

新配对smoke实际完成94.38sec，strict CPU最高复载.895（选择仍初始e0，不称性能改善）；
12个原phase参数在last确实变化，固定alpha审计PASS，原架构无新增层。
PID3483357已退出、原3090 UUID上下文不存在，GPU释放已实际核验，其他用户进程不动。
report/protocol已本地备份editor16_sensor_retrain_20261004/paired_smoke。
后续仍需有据的分层响应/量化/截顶建模及有限充分重训，不把smoke或单帧统计冒称传感器标定完成。

### 2026-10-04 用户新授权：完善传感器模型与 robust mask 训练（不等同末端适配）

用户明确CCD建模既模拟传感器误差，也训练耐误差mask；不能仅随机加噪声碰运气。
当前对齐r3采集保持不动（最近实际vision_router640、3640/6000）；新授权先做诊断与配对训练验收。
本地负责仅T04 train_editor16_robust_chain.py、audit_train_sensor.py、test_sensor_pairing.py；
源码Git d6a27fd5c已发布，服务器仅恢复这三文件，原入口先核验与370032c91 SHA完全一致，其他overlay不动。
新增可选paired-clean-weight/consistency-weight，默认0保持历史行为；同TRAIN批次干净/扰动监督损失加
类别KL+edit概率MSE稳定约束，干净teacher detach、相位原参数继续梯度，不加推理层/不使用TEST梯度。
CPU-only单元检查PASS：一致输出损失近0、noisy梯度有限、teacher无梯度；本地torch DLL不可用，测试在服务器完成。

已通过Git bundle仅在师弟新增无SDK TRAIN-only audit入口，未修改运行capture源码。
原g2_train1000合同scope=train、六层各1000收据，固定stride20共300帧统计已实际完成，
本地备份editor16_sensor_retrain_20261004/physical/train_sensor_audit.json。
语言expert/global median p01=8/255，router=11/255；router median饱和约.00972/.01629，
不同层背景/动态范围不同，旧统一无上限连续噪声未覆盖量化/截顶。
这些是空间观测，不能把单帧空间残差直接拟合成纯read/shot temporal noise或声称已完成标定；
后续应核验同输入理想/实拍的响应映射，分层背景/增益/量化/截顶，仅有证据才定训练噪声范围。
插值/DC留在独立组，避免把光学结构失配都归因于CCD。

实际短训练已启动：服务器既有t04 runs/smoke/editor16_sensor_pairing_20261004，
PID3483357，仅空闲3090 UUIDGPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b，启动前验无compute进程，
noise=randomized/scale1、paired_clean=.5/consistency=.05，原01f7fc4a...eb05a初始、1epoch2step及全TEST CPU检查。
这是入口/梯度验收，不是充分重训或改善结果；下一监督需实际log/progress/report核验完成并GPU释放，
失败先定位不可盲重复。验收与有据噪声模型准备后才安排有限正式重训，原结果全保留，外部上传暂停。
实际smoke epoch0 clean .895，epoch1两step loss .253209665/clean GPU .8825；最高仍初始.895，
仅验收不宣称改善。last与初始比较12个原router/expert/global phase参数确实改变，alpha保护未触发异常；
完整CPU逐样本报告仍计算中，PID3483357尚有GPU上下文，下一次必须核验完成/退出释放，不误称已释放。

### 2026-10-04 北京14:42–14:47 实查（新监督7/24，CCD重训未改善；对齐继续）

r1新CCD组六层各1000PNG/收据、6000完整，capture_audit PASS，精确权重be11e880...154f0d。
同组clean CPU仿真.9155，直接真实TEST Changed-cell **.4940**；低于旧CCD .5175及无trick基线.5490，
本轮噪声假设没有带来所需改善，不能称递进成功或用电子适配掩盖。原始CCD、旧结果均保留。
报告与capture_audit已备份到本地editor16_sensor_retrain_20261004/physical/r1_report.json及r1_capture_audit.json，
两端SHA一致：report ae0e66f264a13604b59190fed00007489ad94307c072aae5a2a088fe73f4ab42；
audit 7a71a1c7672c4301e5d2d2447aeb060cd747ad6ac6580c6f783052d339ae6098。
完整audit各层信号/相位PASS，minp99=64，最高饱和.024128604，无暗帧守卫失败。

同唯一pipeline已自动串行交接r3对齐组（权重05126418...fbc4ca94），selftest error/replay_error均0；
pilot24 audit PASS，各层minp99最低76，最高饱和.015900457，合同曝光/方向/编码保持。
实际r3收据2044→2440→2540持续增加；两语言层已各1000，正在language_global。
遍历2540收据曝光2000/GainX4/wait240/flip_v/对应r3相位SHA/暗帧及15%饱和bad0，
minp99=64、maxsat=.028973582，唯一任务Running，日志无新异常；尚无对齐全量报告或新直接精度。
继续健康流程，不重启、不采旧DC、不追加decoder适配或未经批准的候选；最终两组如实对照交付。

### 2026-10-04 北京14:23 实查（新监督6/24，健康采集尾段）

唯一任务Running267009/Python13068+4344，r1_ccd/test；实际3888→5864→5936配对收据。
前五层各1000PNG+1000收据，vision_global864/1000，随后总收据5936；elapsed2593sec。
遍历5936收据合同/曝光/相位/暗帧/15%饱和守卫bad0，minp99=64、maxsat=.0241286，
最新receipt14:23:15，尾日志无新异常。报告尚不存在，不能把几乎采完说成已评测。
继续同任务等待r1完整报告/audit与自动r3交接，不启动第二SDK/重拍/追加适配；下轮备份真实指标。

### 2026-10-04 北京14:07 实查（新监督5/24，健康采集中）

唯一任务Running267009，Python13068/4344保持；CCD r1/test，实际从672增至3800（随后3888收据）。
三语言层各1000PNG+1000收据完成，vision_router800/1000，progress elapsed1612sec，日志无异常。
遍历3888收据曝光2000/GainX4/wait240/flip_v/相位SHA/暗帧与15%饱和守卫bad0；
minp99=64、maxsat=.0241286，最近receipt时间14:06:58，证明仍出帧而非仅PID存活。
无完整r1报告或新实拍精度，r3仍同任务后续串行等待，不重复采集/提前宣称改善。
下一轮r1六层完整6000+报告/audit后备份实测，并核验同pipeline自动交接r3桥接/pilot/新TEST。

### 2026-10-04 北京13:42 实查（新监督4/24，健康采集中）

唯一OpenMoji_Editor16_SensorRetrain_1004仍Running267009，Python13068/4344同一流程；r1_ccd/test。
实际CCD PNG160→436→512，配对收据512，mtime当前、progress language_router436/1000/elapsed188sec。
已遍历当时512收据，曝光2000/GainX4/wait240/flip_v/六相位SHA/暗p99守卫bad0，最小p99≥246；
尾日志无Traceback/AssertionError/RuntimeError/暗帧失败。pipeline状态文件mtime在phase期间不逐帧更新是正常，
必须看capture progress/CCD收据实际增量，不把phase elapsed42秒当采集卡死。
随后计数增至672收据/676PNG（不同瞬时列举，采集中），再次全部收据bad0，
double统计最高饱和.0126267（1.263%，低于15%），minp99=246；首轮PS整数Max统计0不作有效饱和结论。
本轮尚无完整实拍指标，不称改善达标、不重启、不重复任务；继续单SDK串行两组。

### 2026-10-04 北京13:40 主动补交接（训练已完，新SDK任务已启动）

用户询问后实际核验两组120epoch已结束（history各121行），原PID3263304/3263305退出、原GPU上下文释放。
strict fresh CPU完整TEST：CCD **.9155**，开发选e30（GPU .918）；对齐 **.9265**，选e5（GPU .924）。
仅仿真结果，无新实拍分数，不能据此称达标。训练结束后的交接曾未及时推进，已向用户如实说明并补上。
其他新服务器GPU进程3313432/3/4/5不属于本轮旧PID，不得终止或误报本轮未释放。

本地editor16_sensor_retrain_20261004已备份两组best/last/report/protocol/history/progress/resolved_config全部14产物，
大小SHA逐项PASS；sealed_training_artifacts.tar 110346240 bytes，SHA feca25ea42817c1d5a1fc387a94a007ed9b8e0f004773f33f5b628610dbdf03b。
manifest SHA44713104b64ca1b1a1b092776a0d5a8a85cc374be428bb4f3d17e99a7ed066d3。
CCD best SHAbe11e8808e5e7cdd61c92608bfbf74f54315a86fe5cd460f70b802a37f154f0d，last6f98eccaf334de1230a28018c17f3534f904916744ef74e2580c98a836b9a474；
对齐 best SHA05126418d62408705806e76aca59b9bbc3dc9704d4ff31b67c44a9ddfbc4ca94，lastb208376da4e90c7290bf787cc1e0acefcfc88b2ced6045981b4dd3668b31d7a3。
两best及manifest已转师弟weights并校验完整SHA，源入口a935b87ae保持上轮Git同步。

新唯一任务 **OpenMoji_Editor16_SensorRetrain_1004** 已注册并一次启动，capture-only、仅r1/r3、新prefix；
注册New-ScheduledTask形式遇XML兼容错误，后采用旧已验证任务XML模板仅替换本轮参数注册成功，未修改旧任务。
实际State Running/返回267009（运行中），Python13068/4344同一launcher/model进程，
状态runs/editor16_sensor_retrain_20261004_pipeline_status.json，日志同prefix_pipeline.log。
首核验phase=r1_ccd/pilot，SDK已连SLM/CCD；随后实际读取selftest PASS/error0/replay_error3.814697e-6。
pilot capture_audit PASS，六层各4=24 CCD，minp99最低78，最高饱和比例.017227<.15，六层相位SHA已记录。
流程已进入r1_ccd/test，全量PNG实际28→160持续增加，不只是PID；暂未有完整TEST实拍指标。
24帧pilot通过才6000新TEST，r1结束SDK释放后同任务串行r3桥接/pilot/6000新TEST；不采旧DC、不追加TRAIN/decoder适配。
尚无全量采集数或新实拍指标，下一核验必须看真实文件/收据数量mtime/p99/相位SHA/日志，不能仅PID。
ABO/旧G2/G5与旧DC/.8705最终保留，外部上传暂停。

### 2026-10-04 北京10:57 实查（新监督3/24，健康；硬件入口就绪）

CCD epoch78→114/120，loss .122985；对齐epoch71→104/120，loss .144352，history115/105行。
progress最近27/3秒更新、真实PID/UUID不变、日志无异常；best仍.918/e30和.924/e5，暂无最终CPU报告。
不重启或提前部署。当前预计CCD先完成，对齐随后完成；下一轮须主动交接而非只看PID。

等待期间已核验师弟无OpenMoji/ABO Running硬件任务，并通过Git bundle导入a935b87ae，
仅restore部署入口lab_editor16_robust_chain.py，旧入口LF归一SHA4934b118...30f56身份核验通过，
新LF SHA78fd9bbb8ebaeb86b7df14fca0e9e5baeda072208a9ae5b1fb5dca8402a9edb9；
bundle SHA b8eae9726a9572d8c279830e6f370e296227f7a171c40eb0dd36eb8972994210，
在既有project根editor16_sensor_deploy_a935b87.bundle。首次fetch按分支名失败（bundle只有HEAD），
已改fetch HEAD成功，没有新分支、无源码SCP、未改backend/profile/其他窗口文件。
实际bench Python导入/--help成功，两个选择组/新prefix/capture-only选项存在，未触碰SDK或旧CCD。
权重和manifest仍待两组正式训练及fresh CPU报告完整后转运，不能取运行中best作为最终。

### 2026-10-04 北京10:37 实查（新监督2/24，健康）

CCD epoch40→78/120，loss .129793，history79行；对齐epoch37→71/120，loss .162924，history72行。
progress最近26/18秒更新，日志连续、无错误；真实PID3263304/3263305和原GPU UUID匹配，
只本任务两GPU4626/6618MiB，GPU0他人不动。两最高clean TEST开发仍.918/e30和.924/e5，
无最终report/fresh CPU结果，未提前冻结PT或启动SDK，也未重复训练。等待完整预算及严格复载，
部署入口a935b87ae已准备，下一轮预计接近训练尾段；最终最高可能是较早epoch，必须保留最高而非挑低分。

### 2026-10-04 北京10:17 实查（新监督1/24，健康）

CCD实际epoch4→40/120，loss .167165、最高clean TEST开发.918/e30；对齐epoch4→37/120，
loss .211772、最高clean TEST开发.924/e5。history连续增长、progress最近29/17秒更新，
两真实PID3263304/3263305和原UUID匹配，无异常、无最终report；不是实拍成绩，不重复启动。
其他GPU0他人进程保留。等待120epoch及fresh CPU全TEST最高复载，不能提前用临时best部署。

等待期间已完成本任务部署入口准备：Git **a935b87ae**（仅lab_editor16_robust_chain.py），
增加`--groups r1_ccd r3_ccd_dc30_grid --prefix editor16_sensor_retrain_20261004 --capture-only`。
manifest必须恰好包含这两组；部分重跑拒绝旧封存prefix，拒绝重复组/乱序/路径穿越/重复权重。
已通过语法及纯函数守卫测试，保持旧默认三组流程不变。该入口尚未同步师弟、未启动SDK。
后续仅Git同步新入口、权重manifest+SHA；strict loader/bridge/pilot后新6000×2 TEST，
capture-only明确禁止额外TRAIN或decoder适配，旧DC和旧最终保持。

### 2026-10-04 新轮实际启动：CCD与对齐重训（不是decoder适配）

用户明确突破.60指累计对齐组，不是无trick直接组；旧G4 .5985和最终.8705保留。
此前方案阶段未启动，已向用户如实纠正。现入口Git发布370032c91，既有服务器t04工作树仅同步该入口，其他overlay不变。
源初始PT仍01f7fc4a...eb05a，editor16/decoder64/alpha/frontend合同保持；光学相位和原电子参数共同仿真训练，非末端适配。

有限预声明两组：r1 CCD采用统一router/expert随机增益.9–1.1、偏置0–.03参考均值、读噪声.01、信号相关噪声.01；
r3采用同噪声+原DC30+TRAIN以.5概率做原17→8→17插值，其余为干净采样。正常eval全部关闭。
该模型仍是未标定假设，非根据TEST或真实CCD拟合出的传感器；不保证单调。保留旧DC作为历史参照，
由于新噪声与旧DC不同，不能把新对齐对旧DC的差异归因于单一插值因素。
各组计划120epoch完整TRAIN5000，同起点seed73，TEST每5最高开发选模/noTEST梯度/noVAL，无新层。

服务器新输出：既有t04/runs/simulation/editor16_sensor_retrain_20261004。
ccd_smoke PID3261929和align_smoke PID3261930均已结束：2步梯度loss .24257/.35126，
关闭扰动及fresh CPU全1000严格复载均.8950，报告smoke_complete，通过并释放。
正式ccd_randomized PID3263304（UUID e8837b85）和align_mixed PID3263305（UUID4d8bfdb9）已实际启动各120epoch；
首次实查epoch1，完整TRAIN epoch loss .27998495/.34401586，elapsed43.2/42.9sec，日志无异常，GPU分别4624/6618MiB。
不能将epoch0原基线.895当新结果；待每5epoch最高与最终fresh CPU报告。
监督openmoji-ccd已通过工具创建，20min最多24次，只本轮、健康静默，失败查因，完成实拍交付删除。
下一步部署原lab_editor16_robust_chain硬编码三组/旧prefix，须Git安全支持仅新r1/r3、新prefix、跳过decoder适配后测试，
不得盲运行旧pipeline覆盖结果或重复DC采集。
旧GPU0他人不碰；新上游不得复用旧CCD，选择PT后必须新桥接/pilot/独立新TEST采集。
外部上传暂停、ABO/其他封存模型不动。

### 2026-10-04 用户澄清最终比较基准及重训要求

用户明确最终应比较**无trick基线仿真.8950**，不是G5自身仿真.9165。
最终.8705相对此基准差**2.45个百分点**，按绝对2.5个百分点门槛.8700已通过；
相对降幅为2.7374%，若采用严格相对2.5%门槛.872625则仍差.002125。
以下六项按用户指定顺序整理，均为Changed-cell，最终为TEST选模开发指标，不称独立泛化：

| 项目 | 数值 | 口径 |
|---|---:|---|
| 无trick仿真 | .8950 | 原完整TEST1000 clean CPU |
| 直接部署 | .5490 | 无trick未微调真实CCD |
| CCD | .5175 | G3未微调真实CCD |
| DC | .5985 | G4累计CCD+DC未微调真实CCD |
| 对齐 | .5730 | G5累计CCD+DC+对齐未微调真实CCD |
| 最终 | .8705 | G5原decoder适配后strict CPU真实CCD回放 |

用户要求重新改善CCD噪声建模，期望真实直接结果处在无trick与DC之间；重训必须是仿真阶段原架构训练，不是只微调末端电子。
同时提出“直接训练的也需要再高一些，突破.6”，该词指无trick直接组还是DC/robust未微调组需确认，
否则与CCD要求位于.5490和.5985之间可能冲突。本轮尚未启动新训练或采集，旧结果全部保留，不承诺人为制造顺序。

### 2026-10-04 北京01:45 最终验收交付（监督第16/24次，本轮结束）

唯一pipeline完整结束，任务Ready/返回0，主3520和launcher21568均已退出；SDK已关闭，训练GPU上下文随进程释放。
G5完成160epoch，TEST开发选epoch105；strict CPU复载PASS **.8705**，原基线缓存 **.5730** 与直接报告一致。
上游保护前后SHA均6a2a938e40b3fe2c5561143cb786e89efdf25e99c9b2054398f892117b8d424a，
架构/相位/alpha不变、仅原decoder30162参数、TRAIN1000梯度/TEST每5选开发，无TEST梯度或VAL选择。

| 组别 | 原clean仿真 | 直接实拍 | 本轮适配后 |
|---|---:|---:|---:|
| 保存的无trick基线 | .8950 | .5490 | 未重跑（旧.8395保留） |
| G3 CCD噪声 | .9075 | .5175 | 未适配 |
| G4 再DC30% | .9110 | .5985 | 未适配 |
| G5 再TRAIN对齐proxy | .9165 | .5730 | **.8705** |

G5提升29.75个百分点，但距原仿真4.60个百分点/相对5.0191%，低于.8935875目标2.30875个百分点，**未达相对2.5%目标**。
最终保留格.985111、sceneexact.569；add/replace/move/remove Changed-cell .788/.824/.874/.996。
不降低原仿真基准，不强凑robust单调，不追加无限TEST搜索或未经批准逐层改动。

本地 `editor16_robust_chain_20261003/g5_adapter` 已备份best/last/last_decoder、report/strict_reload/history/execution/progress/test_samples，
九个文件大小及SHA与远端核验一致；最终comparison及G5 TRAIN报告也已取回，原CCD和收据留师弟完整保留。
best SHA **7f3268c8d94dd2ec1389a9ebc247b0a1a5301a9ae9e43315177364803cd5e40a**；
last SHA **67543d96d41f35ac7820e11737b4316bac7ddbaafb910809bc3f830e4ace6023**。
本轮监督 `openmoji-robust` 已删除，不再复活旧任务。ABO/旧G2/G5封存保持，外部上传仍暂停。

### 2026-10-04 北京01:24 实查（监督第15/24次）

唯一pipeline仍Running、主3520/launcher21568，G5/tune已完成缓存并实际训练epoch141，随后history递增至144。
progress最近6.36秒更新，loss约.1248，无异常；当前最高TEST开发 **.8705**，selected epoch105，
epoch140 TEST .8595，尚未结束160epoch或strict CPU复载，不称最终成绩。
较直接.5730提高29.75个百分点，距原仿真.9165仍4.60个百分点（相对5.02%），未达.8935875目标。
best.pt与last_decoder.pt存在，GPU4060主3520在用；不能提前释放或重启。下一轮核验正式完成、保护上游及best/last CPU报告并备份。

### 2026-10-04 北京01:04 实查（监督第14/24次）

G5独立TRAIN六层各1000PNG+1000收据完整6000，capture audit PASS、minp99≥49，相位同G5 TEST。
审计已备份physical/g5_train_capture_audit.json；SDK日志close dev ok，单pipeline自动进入G5/tune。
实际launcher21568/主3520保持，Running267009；execution精确G5 SHA094d0115...0489a4，
只原shared_readout.decoder 30162参数、FIT1000/VAL0/TEST1000，每5epoch TEST选开发、无TEST梯度。
protected_before=6a2a938e40b3fe2c5561143cb786e89efdf25e99c9b2054398f892117b8d424a，
split_audit SHA=d652a675abfdc23600531e206b6cefefdcbea4525d5101b4821aa4424efffd5c。
CPU TRAIN缓存已实际推进至101/1000，尚无梯度分数；下一轮查缓存增量、CPU基线必须.5730及epoch进度。
流程无停滞、未重复启动，原定目标.8935875不变，外部上传暂停。

### 2026-10-04 北京00:44 实查（监督第13/24次）

唯一pipeline仍Running（launcher21568/主3520），处于G5/train，日志无异常。
G5独立TRAIN前三层各1000PNG+1000收据，vision_router718对，总3718/6000；
较前轮1219增加2499对，最新收据0.028秒、p99=237/后续239、234。
各层相位同精确G5 TEST，曝光/增益/wait/方向保持，实际收据持续更新，无停滞。
TRAIN报告尚未生成，尚未进入decoder梯度适配，不提前报微调成绩；继续既有单SDK队列，无重复启动。

### 2026-10-04 北京00:24 实查（监督第12/24次）

G5全TEST六层各1000PNG+1000收据完成，capture audit PASS，minp99≥49，权重/相位正确。
G5 cleanCPU仿真 **.9165**，直接真实Changed-cell **.5730**，差34.35个百分点；
比无trick.5490高2.40个百分点，但比G4 .5985低2.55个百分点，三组不单调，不修饰结果。
保留格.991562、sceneexact.434。报告及审计已本地备份physical/g5_test_report.json、g5_capture_audit.json，
SHA分别2c5cc35779f2a69aaed96ff26874eb16c478221d075e957e6076851a52bbe09d、
d8ec590bff9694875aded020e5f4b5ca504690bf802648191a9e2d366aa0026a，与远端一致。
直接.5730低于原定.8935875，唯一pipeline已自动进入G5独立TRAIN采集，不是重复启动。
launcher21568/主3520保持，TRAIN language_router1000对、language_expert219对，共1219/6000；
最新收据0.013秒，p99=81/后续83，TRAIN相位同精确G5 TEST，日志正常递增。
等待独立TRAIN完整及审计后原decoder适配，暂无微调结果；ABO/旧权重不动，外部上传暂停。

### 2026-10-04 北京00:04 实查（监督第11/24次）

唯一pipeline仍Running，launcher21568/主3520保持，267009为运行中。
G5全TEST前五层各1000PNG+1000收据，vision_global100对，共5100/6000；
较前轮2120增加2980对。最新收据1.06秒，p99=112/后续135，各层相位保持、无异常或停滞。
G5完整报告尚未生成，不提前报精度。按原队列等待全量评价，再必要独立TRAIN及原decoder适配。
G3/G4已验收结果.5175/.5985不变，未重复任务、重拍或恢复外部上传。

### 2026-10-03 北京23:44 实查（监督第10/24次）

G4全TEST六层6000PNG+6000收据完成，capture audit PASS、各层minp99≥50，精确权重与相位核验通过。
G4 cleanCPU仿真 **.9110**，直接真实Changed-cell **.5985**，差31.25个百分点；
比G3 .5175提高8.10个百分点，比无trick基线.5490提高4.95个百分点。
保留格.990024、sceneexact.437。完整报告/审计已备份本地physical/g4_test_report.json、g4_capture_audit.json；
report SHA f91978ed517d52e1cab6b02c8d1a29e74157316a761cdb36636e7a9b9697a511与远端一致。
唯一pipeline仍Running（launcher21568/主3520），已自动进入G5全TEST；
language_router/expert各1000对，language_global120对，共至少2120/6000，最新收据0.62秒、p99=72。
日志继续更新、无异常；G5尚无完整指标，按原流程继续，不重复启动或改变合同。

### 2026-10-03 北京23:24 实查（监督第9/24次）

唯一pipeline仍Running，实际launcher21568/主3520保持，267009为运行中，无异常。
G4五层各1000PNG+1000收据，最后vision_global928对，日志继续929，合计5928/6000；
较上轮3380增加2548对。最新收据1.39秒，p99=113/后续115、141，phaseSHA保持，信号正常。
尚未有G4完整report，不提前报告精度；G5未开始。下一轮重点核验G4结果及G5自动交接。
没有重启、重拍、额外训练或恢复外部上传，继续既有单SDK流水线。

### 2026-10-03 北京23:04 实查（监督第8/24次）

唯一pipeline任务仍Running，launcher21568/主3520不变，返回267009为运行中。
G4全TEST前三层各1000PNG+1000收据，vision_router380对，总至少3380/6000；
较前轮452增加2928对，最新收据0.62秒、p99=233，前三层末帧p99=216/53/51。
相位SHA及2000us/GainX4/wait240/flip_v保持，日志持续逐帧更新，无异常或停滞。
G4全量报告尚未生成，G5尚未开始，不提前报指标；G3已交付结果.5175不变。
保持既有单SDK流程，无重启、额外训练或上传。

### 2026-10-03 北京22:44 实查（监督第7/24次）

唯一任务仍Running，launcher21568/主3520保持，已自动接续G4，无重复启动。
G3六层TEST各1000PNG+1000收据，共6000，capture audit PASS；各层minp99均≥50，phase/合同核验通过。
G3 cleanCPU仿真 **.9075**，直接真实Changed-cell **.5175**，差39.00个百分点；
比保存的无trick直接实拍.5490低3.15个百分点，CCD噪声单项本轮没有改善，不强凑递增趋势。
保留格.997535、sceneexact.403；报告与采集审计已下载至
`editor16_robust_chain_20261003/physical/g3_test_report.json`、`g3_capture_audit.json`。
G4已进入全TEST，language_router452对，日志继续455，最新收据1.21秒、p99=214；
相位SHA74718d4f6a343d356a03a4a03d35f60909f21bb126284917aac38c6b25fc885f，信号正常、无异常。
G5尚未开始。按原单SDK流程继续G4/G5及必要G5独立TRAIN和原decoder微调，不中途重训或改变对照。

### 2026-10-03 北京22:24 实查（监督第6/24次）

唯一硬件任务仍Running，launcher21568/主3520不变，无失败日志。
G3 TEST已完成language_router/expert/global和vision_router各1000PNG+1000收据，
vision_expert684对且日志继续687，总有效至少4684/6000（较前轮1692增加2992）。
最新收据距检查1.53秒，p99=154/后续164，phaseSHA与pilot一致，正常曝光/增益/wait/方向保持。
仍在G3全量第五层，不是停滞；G4/G5未开始，全量report未生成，不报告提前精度。
没有重复任务、重拍或占用服务器GPU；下一次检查应重点看G3报告和G4接续是否完成。

### 2026-10-03 北京22:04 实查（监督第5/24次）

唯一硬件任务 `OpenMoji_Editor16_RobustChain_1003` 仍Running，actual launcher21568/主3520保持；
LastTaskResult267009为正在运行，不是失败码。G3全TEST language_router已1000PNG+1000收据，
language_expert692PNG+692收据并继续693（最近收据0.83秒，p99=59/后续98）。
两层phaseSHA与pilot一致，曝光2000/GainX4/wait240/flip_v未变，日志有实际逐帧增量，无异常。
全量当前至少1692/6000，新收据持续产生，其他三层尚未开始；无report、不提前报实拍精度。
pipeline status只在阶段切换写入，phase `r1_ccd/test` 未更新elapsed39不代表停滞，务必看实际CCD/mtime。
G4/G5尚未启动采集，按原单SDK队列继续，不重复启动任务或服务器训练。

### 2026-10-03 北京21:44 实查及交接（监督第4/24次）

**本次交接随后完成：**三份PT已传完、远端SHA全PASS；manifest SHA
`43aa0cbf51c836c8bc14dd6736112acfe902e8a45ad39b992edbf4ce2ba8c1bd`。
师弟三组CPU selftest PASS：ideal桥接全0，replay误差0/3.8147e-6/5.7220e-6。
核实无其他硬件Running任务或SDK Python后，已启动唯一新任务 `OpenMoji_Editor16_RobustChain_1003` 一次，状态Running。
从此**不得再次启动**已注册任务、重新训练或重传已验证文件；改为查pipeline实际PID/log/status/CCD收据递增。
SDK流程先G3 pilot→全TEST→G4 pilot→全TEST→G5 pilot→全TEST，然后必要G5 TRAIN→原decoder适配。
G3 pilot现已24/24有效、六层minp99=201/67/62/232/144/126，最大饱和1.0513%，相位收据核验PASS。
pipeline已经进入 `r1_ccd/test`，实际Python launcher21568/主3520；首次全量初始化中，后续看实际CCD/收据增量。

三组120epoch及freshCPU全TEST1000严格复载全部完成，三个原PID已退出，nvidia计算进程仅他人GPU0/1246190，自己的三卡已释放。
G3/G4/G5 final cleanCPU = **.9075/.9110/.9165**，选e30/e15/e5（完整训练后保留最高，并非提前停训）。
G5相对2.5%恢复门槛 **.8935875**。本轮尚无实拍结果，不借仿真推断真实精度。

| 组 | best SHA256 | last SHA256 |
|---|---|---|
| G3 | bcdfac05e38ecafce3d6e454d2b55c617a08093f08f364ccd83285971e421c36 | 5ff83d1ca58100ad1791180731b8cbc2f938adc07ce608cc269a09070300d01d |
| G4 | a113be6c5143972d652b8801d9e89b767761b9661dff6bb71105e9583c6d7f6d | f674d6f133f640dfbb880c4881a67d4ab08d4b83c4dbe7f5f09c446dfbd1055c |
| G5 | 094d01157e63d56d195f154bc0671e9e2ce43ac32df34f47d893ea233c0489a4 | ced817d89e921bf8140ef8359cd4fa3b8142fd6c980bfc38487c1465c9b9229a |

本地 `editor16_robust_chain_20261003` 已备份best/last/protocol/report/history/config/progress各三组共21文件，大小和SHA全核验；
manifest `deployment_manifest.json`，sealed tar SHA78438950b82f441d425ad69921867ee563320cbb90406f7a78ebb9acf8c8aa67。
师弟已收到 `weights/editor16_robust_deployment_manifest_20261003.json`；三份best正传输，不得对未完成文件做模型加载。
唯一新任务 `OpenMoji_Editor16_RobustChain_1003` 已注册Ready、**尚未启动**，无trigger，命令为本轮部署module `--phase all --epochs160`。
必须等三份PT完成并逐份远端SHA，再先CPU `--phase selftest` 三组桥接通过，核实无其他SDK任务后启动已注册任务一次。
旧任务模板仅用于复制Interactive PS登录方式，不启动它。新任务输出 `runs/editor16_robust_chain_20261003_pipeline.log/status/comparison`，
每组run前缀 `editor16_robust_chain_20261003_<r1/r2/r3...>`，内部依次pilot24、TEST6000，最后G5 TRAIN6000及适配。

### 2026-10-03 北京21:24 实查（监督第3/24次）

G3已完成epoch120，正在进程内freshCPU完整TEST复载，report尚未生成，不提前释放或交接。
G4/G5 epoch115/102（较前轮+36/+32），loss .142440/.147699；G3最终TRAIN loss .123245。
暂存最高GPU TEST仍 .9110/.9120/.9185，selected30/15/5。三份日志无异常、progress最近4/30/32秒，
实际PID都在，三张UUID对应显存4626/4828/6702MiB，没有新任务或CCD；最慢G5仍需18轮。
下一轮应可核验三组最终CPU report/SHA/进程释放并主动开始manifest备份和师弟权重交接。

### 2026-10-03 北京21:04 实查（监督第2/24次）

G3/G4/G5 epoch82/79/70（较前轮+37/+36/+31），loss .128347/.154933/.170096；
三份 progress 最近更新距检查24/9/27秒，history83/80/71行。三任务真实PID→GPU UUID及显存与上轮一致，
没有异常或停滞，尚未到120 epoch及CPU最终复载，report均未生成，不能提前称选定最终PT。
暂存最高 GPU TEST仍 .9110/.9120/.9185；没有重复任务或开始采集，等待足额训练后交接。

### 2026-10-03 北京20:44 实查（监督第1/24次）

G3/G4/G5 分别完成 epoch45/43/39，均仍在正常训练，完整120 epoch预算未结束。
loss 分别 .161125/.206934/.213165；暂存最高 GPU clean TEST .9110/.9120/.9185，
选中 epoch30/15/5，仅为中途开发值，不是最终CPU复载或光路精度。
三份 progress 最近更新距检查21/7/6秒，history46/44/40行，日志仍持续递增、无异常；
三个实际PID存活且对应原先三张GPU UUID，显存4626/4828/6702MiB。
GPU0他人1246190保持不动；没有重复启动，没有最终report或新CCD，暂不进行硬件交接。

三组正式训练已启动；尚无本轮最终仿真或实拍结果。旧 baseline 保留且不改：
完整原 TEST1000 clean CPU 仿真 **0.8950**；同合同真实 CCD 未微调回放 **0.5490**。
旧末端适配 0.8395 也保留，但不是本轮 G5 结果。

基线初始 PT SHA256：
`01f7fc4a8de4f20901fae09e8be55c67fce2177db0a7ed3975a7fc90853eb05a`。
旧报告/权重在 `preserved_g2_20261003/physical_adapter`，不覆盖或重采旧数据。

## 已锁定协议

- 同一 editor16 + 原 decoder64，head 179224 / decoder 30162 参数，无新增层或支路。
- G3 `r1_ccd`；G4 `r2_ccd_dc30`；G5 `r3_ccd_dc30_grid`。
- 三组分别从同一个 clean baseline 开始，不以 G3 最佳 PT 暖启动 G4。
- 每组 120 epoch，每 epoch 完整 TRAIN5000；seed73、相同 optimizer 和预算。
  相比旧 baseline 的 head-only 初训，这是额外全模型 robust 续训；不冒称与旧 G2 总预算完全匹配。
- 原 frontend/alpha 冻结，alpha 保持已验证原值约 .49–.52；原光学相位、路由和电子参数可学习。
  因而每个选定 PT 必须全新采自己的六层 CCD，绝不复用旧 G2 CCD。
- CCD 采用历史 scale5 proxy：均值参照 offset .15 / read noise .05，gain1。
  不是已标定相机噪声；DC 为历史 expert/global 相干相位泄漏强度 .30；
  grid 为可微 17→8→17 raster proxy，不是真正 8µm 传播。
- grid/noise/DC 仅训练启用；正常仿真与上机使用 clean r0 有界振幅合同。
- 完整 TEST1000 每5 epoch 按最高选择 PT，无 TEST 梯度、无 VAL 选模。
  报告明确为 TEST-selected 开发指标，不能声称独立泛化。封存后 fresh CPU 全 TEST1000 严格复载。
- 最后 G5 若实拍低于 `G5原始clean仿真 *.975`，采独立 TRAIN1000×6，
  仅原 decoder30162 适配160 epoch，TRAIN 梯度、TEST每5最高选 PT，冻结全部上游。
  2.5% 采用相对差口径，同时报告百分点差；不保证达标或三组单调上升。

## 服务器运行身份

既有运行目录：
`/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t04_openmoji_robust_20260928`。
runtime HEAD 原 `3b956503` + 保留 overlay；新训练入口 Git 发布 `8edcbb0b7`。
不整体 checkout/pull，不覆盖脏 profiles/其他源码。

输出：`LightGenV2/tasks/t04_openmoji_robust_ablation/runs/simulation/editor16_robust_chain_20261003`。

| 组 | 实际 PID | GPU UUID | 预算 |
|---|---:|---|---|
| G3 | 2964315 | GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d | 120 epoch |
| G4 | 2964316 | GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd | 120 epoch |
| G5 | 2964317 | GPU-1b963983-7909-af6e-0528-f0f0661ab549 | 120 epoch |

三组 smoke 完成并退出：e0 clean 均 .895；两步训练有限 loss、alpha 不变。
正式作业启动前三张 GPU 无计算进程，GPU0 他人进程 1246190 不动，A100 不用。

## 后续主动交接（不能只看 PID）

1. 查 protocol/progress/history/log/mtime、实际 PID→UUID；失败保留输出深入定位，不重复启动。
2. 120 epoch 后核验 report complete、fresh CPU、best/last/SHA、alpha 和架构，备份必要产物；确认 PID 消失/GPU释放。
3. Git 发布部署入口 `lab_editor16_robust_chain.py`，权重用 manifest+SHA 转师弟，源码只 Git bundle/scoped restore。
4. 既有师弟 `OpenMoji_Robust_Rank64_SHS_20261002`，先三组严格加载/理想桥接，然后各组 pilot24→TEST6000，单SDK串行。
5. G5 独立 TRAIN6000→原decoder适配。pipeline 比较表保存三组仿真/直接及G5适配；严格 CPU best/last 复载保护上游。
6. 硬件正常合同 2000us/GainX4/wait240/原ROI方向/保零tanh与round255a；
   用户批准 TEST15%/TRAIN95% 饱和接受并记录，暗 p99<15 仍停，不能改曝光造 baseline。
7. 仅本地交付，ABO/旧G2/G5不动，师姐上传暂停。正常进展安静，仅重要结果/失败/必要用户动作通知。

部署入口已发布 `75eeaa29503b5e87d22b75ec4db939a7068ec67b`，并已通过 Git bundle
同步到既有师弟 source，只恢复两个本轮新入口；真实 bench import 和三个冻结 backend SHA 校验 PASS。
bundle SHA256 `8e50c23673f7014a9accb37c78d04db8afa1b4e05f1567579d2afd4cec6d818b`，
Git bundle 已导入，不需再次同步旧源码。师弟检查无 OpenMoji/ABO 硬件任务 Running。

当前尚未上传本轮 PT 或注册本轮硬件任务。此状态必须随实际交接更新，不能把“已准备入口”写成“已实拍”。
本轮 heartbeat `openmoji-robust` 已建立，每20分钟仅推进本轮，正常安静，完成后删除，最多24次。
它不等于已注册硬件任务，训练结束必须实际核验并主动交接 manifest/PT 和一次性串行 pipeline。
