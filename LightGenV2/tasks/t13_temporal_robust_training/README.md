# T13 时间一致性鲁棒训练与光路消融

这是基于导师审阅版的时间一致性鲁棒消融任务，四组仿真训练已完成，尚未完成新光路实采。旧导师包不修改；本目录使用内部固定runtime快照，不运行时依赖父仓库、原T06或其他任务。当前profile为schema=4，配置为configs/study_full2250_testbest.json：原2250 train/558 test、不划validation，每5epoch按共同8μm/DC30/无CCD噪声的test SRCC选best。主报告是固定选定权重在共同noise-scale=1下的最终评价，不是选模时的无噪声数字。所有组phase dropout=0.05、router noise=0.06，像素平移全部关闭；tanh/0.5振幅与固定255量化一致。schema=3及其报告是作废历史，旧产物已隔离；原导师包、公共缓存、schema=4正式结果和父阶段均保留。

**2026-10-03整理入口：**[唯一复现入口](reports/reproduction/README.md)集中列出源码、四份权重SHA、数据与原始报告位置。整理不重新训练、评估或采集。已保护服务器源码HEAD为`493eb38375f3bd9e4e8e7edfe278468b351365f1`，实际本轮训练记录commit为`1aa3842c01ddb54c8b0d61f8428de391e11cdaaa`；前者还包含结果文档与独立仿真导出，不能把导出包未经复核称为本轮训练运行时。

弱光修正与是否修改损失的正式说明见 [reports/BOUNDED_AMPLITUDE_FIX.md](reports/BOUNDED_AMPLITUDE_FIX.md)。同一振幅用于仿真与固定255量化，禁止逐帧峰值缩放；不是删除所有RMS。四组保持原MOS损失，不额外捆绑功率loss。旧teacher_reference保持历史模型不改，schema=3须重新训练。

## 当前正式结果与交付

当前有效run为`full2250_testbest_phase_only15_s163_uuid1256_20260927`，从各自上一阶段best继续15轮光学相位局部优化及共同评价完成。
test SRCC依次0.792741/0.785567/0.803539/0.800796；不是光路实测，test参与选模。选中best epoch5/15/1/0，前三组只改变6个phase张量且91个电子参数不变，第4组保留父best；不强造第4组最高。
主报告只放一张四组结果表，见[reports/FULL2250_TESTBEST_RESULTS.md](reports/FULL2250_TESTBEST_RESULTS.md)。其他评价条件仅为原run内诊断，不是额外实验组。
当前四份真PT工程位于`projects/temporal_full2250_phase_only15_s163_20260927`；父阶段run和工程保留历史证据，当前迁移只使用这个目录。旧schema3产物在`.codex_tmp/t13_retired_schema3_20260927`隔离，不用于部署。

## 来源与版本边界

本轮共同光学相位局部优化已完成：四组从low_lr30阶段各自best出发，电子参数全部冻结，只更新教师optimizer命名合同下的六个光学phase张量，15epoch，phase LR=0.0006、router phase LR=0.00096（原始LR乘0.03）；电子组配置LR不生效，因为requires_grad=False。2250/558、test-best、物理/增强条件及预算各组一致。保留epoch0父best，不为“第4组应该最高”改变评价条件。optimization_parameters.json记录真实可训练和冻结名称，逐张量核验已通过，结果及当前工程入口见上文。

已完成的父阶段小学习率优化：四组从各自schema4 best继续30epoch，三类学习率统一乘0.1，电子3e-5、普通相位0.002、router相位0.0032。重新建立AdamW及30轮cosine scheduler（权重warm start，不恢复旧优化器）。父PT组别及物理/数据/噪声合同一致；2250/558、test-best、四组条件不改。父best作为epoch0候选；本阶段是当前15轮phase-only的来源，不是待执行训练。

正式配置加载统一经过 `settings_adapter.py`：仅去除导师版“DC至少20%”的历史策略门槛，保留真实数据模式、几何及合法区间检查，不以synthetic模式绕过。训练前两组DC区间为0，后两组为0.30；共同部署各组区间固定0.30，理论参考固定0。原导师runtime保持哈希不变。

最新导师包位于：

```text
LightGenPublic/tasks/t06_lgvq_temporal_consistency/
  teacher_release_final_v2/lgvq_temporal_08044/
releases/LGVQ_Temporal_08044_teacher_final_v2_20260923.zip
```

其权重 SHA256 为 `5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c`。
已对实际 ZIP 的 77 个清单文件逐项校验，包含 81,776,084 字节的 `weights/best_checkpoint.pt`，证据见 `reports/teacher_zip_audit.json`。导师 ZIP 有完整结构，并非只有源码。本任务上一版是开发框架，不等同于该推理交付包。
该最新包的实际固定权重复评 SRCC 是 **0.8022806420**；名称中的 0.8044 是历史标识，不是当前包的重评结果。

`runtime/` 的 21 个 Python 文件是该包的字节级源快照，`reference/teacher_v2.yaml` 是原配置；`reference/source_manifest.json` 记录来源及哈希。新增行为只在本目录的适配器中实现，不暗改源快照。

## 最终设计：5 个展示条件，4 次训练

| 展示条件 | 训练 ID | 探测噪声训练 | 相干未调制直流训练 | 设备网格处理 | 是否上光路 |
| --- | --- | --- | --- | --- | --- |
| G1 理论仿真 | 复用 G2 权重 | 无 | 无 | 17 μm 理想网格，无设备转换 | 否 |
| G2 基础部署 | r0_post | 无 | 无 | 训练后插值到 8 μm | 是 |
| G3 +探测噪声 | r1_ccd_post | 有 | 无 | 训练后插值到 8 μm | 是 |
| G4 +相干直流 | r2_ccd_dc_post | 有 | 有 | 训练后插值到 8 μm | 是 |
| G5 +训练中插值 | r3_ccd_dc_intrain | 有 | 有 | 训练图内映射到 8 μm | 是 |

G1 是同权重理想条件参考，不保证是上界。G2→G3、G3→G4、G4→G5 相邻各改一个因素。四次训练从共同随机初始化独立进行，不串行继承上一组 best，也不用已经 robust-trained 的旧权重初始化来声称“没有鲁棒训练”。

这是**累计措施的条件增量消融**，不是完整 factorial；G3-G2 是无 DC 训练前提下的 CCD 收益，G4-G3 是已有 CCD 训练前提下的 DC 收益。不能宣称因素独立主效应或把增量跨排序泛化。seed=163；单 seed 只能做 demo，后续有预算再重复关键配对。

## 保持不变的合同

- 任务：LGVQ Temporal consistency，16 个视频×每视频 4 帧；4×4 整幅相干场，6 次传播，1 视频1 MOS；padding 保留在场中但不计指标。
- 架构、电子支路、归一化、优化器、训练轮数和相位参数量相同；保持 532 nm、10 cm、478 有效区、518 逻辑画布、17 μm 逻辑 pitch、0.5° 截止。
- 仅研究用户指定的三个措施；三类位移统一关闭。phase dropout=0.05、router logit noise=0.06四组共同开启，普通网络dropout/训练损失各组相同，不把共同增强算作消融增量。
- G2–G4 训练传播网格为 518/17 μm；G5 为 1101/8 μm，但可学习 mask 仍是原逻辑参数，不通过增加 mask 参数量取胜。
- 所有组选择 checkpoint 时都在同一8 μm、eta=0.30、无CCD噪声的原558 test评估，保留原test-best口径，不主张无偏泛化。最终同时报告DC30带噪声、DC30无噪声、DC20无噪声评价；后者只用于对齐旧版评价条件，不用它再次选权重。不会把G2的17μm和G5的8μm数字直接比较。
- 物理孔径保持约 8.126 mm；8 μm 设备有效图 round(478×17/8)=1016。不能保持像素数却改变物理孔径。

## 三项建模的实际含义

相干未调制直流继承导师后端：

```text
M = sqrt(1-eta) * exp(i*phase) + sqrt(eta)
```

原导师eta训练区间0.20–0.35；本轮G4/G5固定0.30、共同设备评估0.30。它是名义混合系数，不等于CCD实测功率恰占30%。不是相机黑电平，也不是phase DC正则。即使G2/G3训练忽略直流，部署评价仍保留共同直流。

导师包的可选噪声是场均值标度的有偏截断Gaussian，随后相对Poisson，正式配置关闭。新study使用ccd.py的独立Poisson-Gaussian：先采样信号/暗电子，再加一份零均值读噪声，扣期望暗电流后裁零。不额外叠加EAC Gaussian。原源码不改。当前schema=4沿用参数为k=3072、read RMS=10电子（此前schema=2为4096/8）；仍是未标定pilot，不是ACCEL的CCD参数。算子在逻辑CCD边界，不称raw sensor/像素积分已标定模型。正式标定需要暗场、平场、photon-transfer及模型强度到电子转换；本轮已明确允许增强pilot训练。详见BOUNDED_AMPLITUDE_FIX.md，ENGINEERING_AUDIT.md中的旧参数保留历史含义。

插值沿用导师包的幅值 bilinear + 单位复相位 nearest 设备网格映射，探测强度 area 回到逻辑网格。禁止 bilinear 直接插值跨 0/2π 的相位角。

**重要部署边界：**原导师实现对“已调制复场”重采样；真实装置分别播放振幅与相位面板。相干泄漏下，两种顺序不一定等价。当前保留此实现以建立可追溯基线，不能称严格的逐像素设备模拟。正式实采前必须验证独立振幅/相位编码与训练 forward 的一致性，必要时为四组统一升级同一 raster operator，形成新 profile，不能只修 G5。旧采集的逐图 p99.5 振幅缩放也不在模型 forward 中，不能直接照抄冒充等价；应先建立物理有界振幅合同与匹配导出。

## 架构

```text
t13_temporal_robust_training/
  reference/       导师原配置、源码哈希与实际指标
  runtime/         原导师兼容运行时快照，只读
  configs/study.json  四组及共同测试合同
  study.py         条件展开、数据预检、TRAIN留出划分
  experiment.py    显式CCD评估、共同设备验证适配器
  run.py           plan/preflight/smoke/train/evaluate/theory
  hardware.py      逻辑CCD替换、六层身份、相位映射
  build_lab_package.py  单组权重/相位/部署manifest暂存
  build_projects.py  由单一源码生成四份独立工程，可携带导师参考PT与输入
  project.py       每份工程锁定组别的入口
  ccd.py           Poisson信号/暗电子 + 零均值Gaussian读噪声适配器
  verify_source.py 原导师源码完整性检查
  tools/           Git服务器同步与真实结果绘图
  tests/           条件、梯度、网格与恢复行为
  reports/reproduction/  唯一复现入口
  assets/          缓存/数据身份；忽略Git
  runs/smoke|simulation|hardware/  独立run；忽略Git
  releases/        最终包；忽略Git
```

运行方法见 [COMMAND.md](COMMAND.md)，图表设计见 [reports/EXPERIMENT_AND_FIGURE_PLAN.md](reports/EXPERIMENT_AND_FIGURE_PLAN.md)。不同任务不维护两份可变模型：旧包只用于参考，消融只在本目录变更，最后从通过验证的该分支构建新的导师交付包。

## 四份工程与权重交付

`build_projects.py` 生成 `projects/<build_id>/01_baseline_post`、`02_ccd_post`、`03_ccd_dc_post`、`04_ccd_dc_intrain`。每份都有 runtime/configs/weights/inputs/teacher_reference、独立入口和SHA清单，不运行时依赖其他三组或父仓库。生成产物不提交Git，源码只维护本目录一份；服务器使用同一commit自行生成，禁止SCP覆盖源码。

`teacher_reference/weights/best_checkpoint.pt` 是旧导师模型，只用于 `project.py reference`；不能改名当四组消融权重。四组真实best/last已经存在，当前四份工程含组别核验后的真实PT，位置及SHA见唯一复现入口。构建器通过`--checkpoints`加入真实权重。训练所需缓存也已恢复并用于正式run；换机器仍须核验私有路径与数据身份，不能用参考35场输入冒充TRAIN。

## 训练与数据口径

原manifest 2250 train / 558 test原样使用，正式预检强制核对样本ID唯一性及2250/558数量，不划validation、不重排缓存。

训练器test字段保持原含义，每5epoch按test SRCC选择best。summary/checkpoint明确test_used_for_selection=true、validation_used=false。测试不参与反传，但参与选模，因此不是独立泛化估计；本任务沿用仓库DATA_SPLIT_AND_ROUTER_PROTOCOL.md的既有口径。

两个Qwen-front缓存已由原始视频及冻结Qwen重建，并严格验证数据身份；35个打包field仍只用于参考推理。新run通过--reuse-cache复用公共缓存，正式预检仍核验身份和哈希。

## 光路边界与下一步

`hardware.measured_ccd_boundary` 在 `_detector` 的逻辑518/478强度边界替换测量，不在1101设备传播网格里塞478 CCD；支持按六层依赖的连续prefix，保留原电子forward，不自动换相位。

`build_lab_package.py` 当前产生单组checkpoint、六张逻辑/设备相位与SHA清单，状态为 `staged_not_capture_ready`。**还不是能直接上设备的完整采集包**：必须接入经过验证的SDK、相位方向/LUT、物理有界振幅、raw CCD/ROI以及数据field身份。原 T06 的 pin-only load_model 和旧 propagator-level replay 不能直接用于新权重/8 μm网格。

采集按每组、每层全量完成，再进入下一层；不循环小批量六层，不混不同mask/曝光会话。558视频=35场，每组6×35=210次CCD采集，四组共840次（不含探针/复拍）。预计时长必须由实测端到端吞吐估算。

## 同步与安全

2026-10-06治理：主线 `maintenance/git_safety/readonly_server_policy.py` 明确拒绝
旧 `sync`／`publish-bundle` 自动创建工作树或任务分支的请求；只核验inspect/test参数，
不连接服务器、不操作Git。其12项纯CPU测试通过。
本机私有 `tools/server_sync.py` 已去除旧写入实现并绑定此规则，保留只读检查和指定commit的CPU测试；
该连接脚本被Git忽略，按根规则不公开提交或通过源码传输复制到其他机器。
不能称其他机器的私有旧脚本已被更新，也不能称此规则强制拦截所有窗口。
源码发布仍按根AGENTS逐项审核、串行同步main。旧实验数据、PT和教师runtime未更改。

遵守仓库根AGENTS.md的单main规则，不自动新建分支/worktree或独立工程。旧`codex/t13-temporal-robust-20260927`工作树作为已验证身份保护，待逐任务发布核验后收敛，不覆盖未提交修改。源码仅Git同步，资产另走SHA清单；私有连接脚本不进入新源码交付。历史正式训练获4卡授权并已结束，当前不启动训练；新运行重新查空闲资源并遵守当时预算。原协议无validation，退出只核验自己的PID。

当前阶段：四组仿真训练、共同评价与真实PT工程已完成；核心源码已在main的139d147bd发布，2026-10-08再次逐项核对62份原服务器运行文件与发布源码，无SHA差异（仅清单声明项作CRLF→LF转换）。硬件包仍为staged_not_capture_ready；四组实采及新版导师硬件交付未完成，不把源码发布或仿真完成冒称实采完成。
