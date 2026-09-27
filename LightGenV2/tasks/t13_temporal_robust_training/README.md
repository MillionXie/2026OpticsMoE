# T13 时间一致性鲁棒训练与光路消融

这是基于导师审阅版的独立研究分支。旧导师包不修改；本目录不运行时依赖父仓库、原 T06 或其他任务。当前profile为schema=4，配置为configs/study_full2250_testbest.json：恢复原2250 train/558 test，不划validation，每5epoch按共同8μm/DC30/无CCD噪声的test SRCC选best；所有组phase dropout=0.05、router noise=0.06，像素平移仍全部关闭。四组统一tanh/0.5振幅、直流训练组30%、增强CCD pilot。schema=3及其报告仅为作废历史口径，不用于部署。用户已要求清理该轮权重及派生工程，原导师包和公共缓存不删。状态以当前run的supervisor.json和comparison.json为准，尚未新光路采集。

弱光修正与是否修改损失的正式说明见 [reports/BOUNDED_AMPLITUDE_FIX.md](reports/BOUNDED_AMPLITUDE_FIX.md)。同一振幅用于仿真与固定255量化，禁止逐帧峰值缩放；不是删除所有RMS。四组保持原MOS损失，不额外捆绑功率loss。旧teacher_reference保持历史模型不改，schema=3须重新训练。

## 当前正式结果与交付

当前有效run为`full2250_testbest_dc30_ccd_s163_uuid2456_20260927`，四组各100epoch及共同评价完成。
带噪声test SRCC依次0.790527/0.786405/0.801299/0.800796；不是光路实测，test参与选模。
完整train/test及无噪声诊断见[reports/FULL2250_TESTBEST_RESULTS.md](reports/FULL2250_TESTBEST_RESULTS.md)。
当前四份真PT工程位于`projects/temporal_full2250_testbest_s163_20260927`，旧schema3产物仅在`.codex_tmp/t13_retired_schema3_20260927`隔离保存，不用于部署。

## 来源与版本边界

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

导师包的可选噪声是场均值标度的有偏截断Gaussian，随后相对Poisson，正式配置关闭。新study使用ccd.py的独立Poisson-Gaussian：先采样信号/暗电子，再加一份零均值读噪声，扣期望暗电流后裁零。不额外叠加EAC Gaussian。原源码不改。当前schema=3参数为k=3072、read RMS=10电子（此前schema=2为4096/8）；仍是未标定pilot，不是ACCEL的CCD参数。算子在逻辑CCD边界，不称raw sensor/像素积分已标定模型。正式标定需要暗场、平场、photon-transfer及模型强度到电子转换；本轮已明确允许增强pilot训练。详见BOUNDED_AMPLITUDE_FIX.md，ENGINEERING_AUDIT.md中的旧参数保留历史含义。

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

`teacher_reference/weights/best_checkpoint.pt` 是旧导师模型，只用于 `project.py reference`；不能把它改名为四组消融权重。各组 `weights/best_checkpoint.pt` 要在重新训练后通过组别核验打包，目前尚不存在。构建器可通过 `--checkpoints` 加入四组真正权重。当前没有假造PT或训练结果。导师参考推理不需要完整训练缓存；四组正式训练仍需要缺失缓存。

## 训练与数据口径

原manifest 2250 train / 558 test原样使用，正式预检强制核对样本ID唯一性及2250/558数量，不划validation、不重排缓存。

训练器test字段保持原含义，每5epoch按test SRCC选择best。summary/checkpoint明确test_used_for_selection=true、validation_used=false。测试不参与反传，但参与选模，因此不是独立泛化估计；本任务沿用仓库DATA_SPLIT_AND_ROUTER_PROTOCOL.md的既有口径。

两个Qwen-front缓存已由原始视频及冻结Qwen重建，并严格验证数据身份；35个打包field仍只用于参考推理。新run通过--reuse-cache复用公共缓存，正式预检仍核验身份和哈希。

## 光路边界与下一步

`hardware.measured_ccd_boundary` 在 `_detector` 的逻辑518/478强度边界替换测量，不在1101设备传播网格里塞478 CCD；支持按六层依赖的连续prefix，保留原电子forward，不自动换相位。

`build_lab_package.py` 当前产生单组checkpoint、六张逻辑/设备相位与SHA清单，状态为 `staged_not_capture_ready`。**还不是能直接上设备的完整采集包**：必须接入经过验证的SDK、相位方向/LUT、物理有界振幅、raw CCD/ROI以及数据field身份。原 T06 的 pin-only load_model 和旧 propagator-level replay 不能直接用于新权重/8 μm网格。

采集按每组、每层全量完成，再进入下一层；不循环小批量六层，不混不同mask/曝光会话。558视频=35场，每组6×35=210次CCD采集，四组共840次（不含探针/复拍）。预计时长必须由实测端到端吞吐估算。

## 同步与安全

专用分支 `codex/t13-temporal-robust-20260927`。本地和服务器共享工作树都有无关修改，使用隔离worktree，不pull/reset。源码仅通过Git同步，资产另走SHA清单。本轮用户明确授权4张GPU，各组单卡；train_four.py先检查空闲，缓存准备完成后并行四组，最终计算train/validation/test指标，退出检查自己的GPU PID，不清理其他人的进程。密码仅交互输入，不写入代码、配置、日志或交付包。

完成条件：源码测试通过并Git同步；训练资产和设备合同预检通过；四组训练后共同评价；四组实采后冻结结果；最后再构建新版导师包并隔离验证。当前不能把搭框架说成训练或实采已完成。
