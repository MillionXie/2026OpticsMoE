# T06当前结果、实际源码与待收敛边界

此页整理已有版本身份，不重训、不重复数据集评估、不改变设备合同。以下两项已有
实拍，是彼此独立的模型，不能把Spatial和Temporal的PT或数据复用为同一项结果。

## 已验证的实拍版本

| 版本 | 固定原模型仿真SRCC | 未适配实拍SRCC | PT SHA256 |
| --- | ---: | ---: | --- |
| Spatial，单视频4帧，原1M读出 | 0.6710968960 | 0.5786364901 | `95e12397ccf8c960fa30ba9dfb400b69d2c2ebd02288ac6a4879870ab592828b` |
| Temporal，16视频×4帧，同场并行 | 0.8043868643 | 0.7977138739 | `5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c` |

两项源数据均为原558条test；空间3348张、时间210张六层有效CCD。原实验报告在
[空间实测](reports/reproduction/SPATIAL_SHS_20260914.md)和
[时间实测](reports/reproduction/TEMPORAL_SHS_20260914.md)。空间末端适配的不同划分、
混合test适配、重代入指标另见原读出报告，不能用不同分母的最高分替代上表。

## 正式核心已收敛到main，历史入口和后续适配仍分开核验

两项实际离线推理包的源码commit为
`8e869473787f4ffceb2a6a77f4430b94c206f459`。各包15份运行Python与其内部SHA清单
及该固定Git源码全部一致；PT也与包声明及原服务器权重完全一致。2026-10-04在服务器
CPU上直接从既有ZIP读取源码和PT做strict reload，两模型均通过；不重新评估558条数据，
未写入/解压新的独立工程。

实拍包在训练服务器主工程 `LightGenV2/tasks/t06_video_quality_assessment/releases/`：

- `20260914_shs_spatial06710.zip`，SHA `fe90b0ac2cdedbe9b71f1d21c1f6cc5dcd6f2e237cefdd9bed1c9e94cde61d2c`。
- `20260914_shs_temporal08044.zip`，SHA `8421e7418ade4c9ec59f924cdad8e274db4100600c001cf802a8291ab7a67aff`。

源码从Git迁移、数据/PT从manifest+SHA管理；ZIP留作已验证旧部署与恢复证据。
主线 `8d6445e28c88346c692acfaef788cd9d4a4e86fe` 已按52项依赖闭包审核，
迁入19项缺失文件和10项正式模型/训练依赖，保留已有原子JSON设备读写修复。
服务器CPU直接加载发布Git树：90份Python编译通过，原57项模型/训练合同测试通过，
上表两份正式PT均strict reload通过；没有新建工程、调用设备或重评数据集。
机器可读来源见 [source_import_20261004.json](source_import_20261004.json)。
这些测试不等于新机硬件复现；旧默认profile及设备辅助入口仍待逐项兼容，不能把
主线核心恢复称为全部工程/资产迁移完毕。

### 后续离线微调入口已单独发布

主线 `5247f84b2b29a2b10616411fa7fb9af4c6a70148` 增补7项微调源码/配置，
来源为 `81fe403e896a4310c077a26e5c33ee8317752609`，不是原9月14日推理包。
原TRAIN2250交付ZIP内适配器仍绑定 `d89223b7d3b9dd3e1b76122fd26fb14cffb6a0a8`；
两种源码身份均保留。`adapt_measured_readout` 和 `tune_measured_readout` 是离线入口，
不调用相机/SLM；无新增模型分支，只更新原读出头。既有设备模块保持上一次发布版本，
未迁入后续自动换相位/曝光助手，也不赋予任何新的采集授权。

候选Git树93份Python编译、原57项模型合同及12项适配协议测试全部通过；两份原正式
PT仍严格加载通过。测试不训练、不重新评估视频数据、不占GPU。
20项源码/依赖SHA及CPU收据见
[适配来源清单](adaptation_source_import_20261004.json)。原实验各划分/选模方式、权重与
指标仍以[微调报告](reports/reproduction/SPATIAL_SHS_READOUT_ADAPT_20260914.md)为准；
尤其112条原test参与梯度的结果与2250纯TRAIN适配结果分开，混合558成绩非独立test。

## 历史默认profile的实际问题

服务器另有2026-09-10四阶段ResNet-E1历史Spatial候选，报告SRCC .666503，
PT为 `6b05961f...`。其两份结果说明已按服务器原字节收敛至兼容后端，旧VGG
报告仍在Git历史；见 [历史报告身份](../../../maintenance/storage/T06_HISTORICAL_SERVER_RECORD_ADOPTION_20261006.json)。
它不是上表六阶段 `95e12397...` 实拍模型，也不替换当前默认profile。
原 `lightgen_spatial_065` 的配置／实现闭包未据这两份报告证明可在main运行。

Temporal-36是“单视频36帧”独立历史baseline，不是上述16视频×4帧实拍版本。
当前默认profile及部分Spatial旧profile锁定的源码SHA与多个实际工作目录不匹配。
部分权重只是移到别处，并未丢失：paired-flip Spatial best/last在服务器
`/DATA/DATA1/guest3/lightgen_spatial_067/promote_s1201/`，两项SHA核验通过。
Temporal-36旧默认路径的PT在本次限定名称搜索中未找到；不能据此判定被删除。

因此目前不得把默认 `--phase evaluate` 视为已验证的最终实拍复现入口，也不修改SHA
或取消检查来强行启动。正式实拍版本按已固定包及原报告读取；默认入口兼容性仍在收敛。

固定实拍模型现在可从主线显式进行只读CPU权重验收：

```bash
python -m LightGenV2.tasks.t06_video_quality_assessment.verify_formal_checkpoint --target spatial --checkpoint /path/to/spatial_best_checkpoint.pt
python -m LightGenV2.tasks.t06_video_quality_assessment.verify_formal_checkpoint --target temporal --checkpoint /path/to/temporal_best_checkpoint.pt
```

沿用 `lab_runtime` 的原SHA、架构和strict state检查，不改默认Temporal-36，不加载数据、
不调用SDK或GPU。权重缺失/不匹配仍失败；通过也不代表性能、数据或光路复现完成。

2026-10-06已在服务器 main `f54e8b2d...` 对两个真实PT执行上面入口：
Spatial 3,786,407参数、Temporal 6,804,011参数均通过原SHA/架构/strict state检查。
未运行forward或数据集，未占GPU、未改文件；结果记入本任务[source_import清单](source_import_20261004.json)。
这补齐的是主线实际权重加载，不是恢复旧Temporal-36资产或重新证明实拍性能。

## 数据、baseline与测速保留

历史 A100 batch scaling 的源码和15份原始测速/遥测/预测文件统一从
[批量baseline身份清单](batch_baseline_source_import_20261006.json)进入。
清单记录原目录、运行子目录及每份文件的字节数/SHA；原文件仍保留在
清单的 `asset_root` 历史区，原目录为 `/DATA/DATA1/guest3/2026OpticsMoE_t06_a100_batch`。
该副本已完整归档后原生Git移动，所有原始测量字节保留；两处源码overlay与登记的Git恢复引用一致。
formal batch16 与 sweep batch1/2/4/8/16 各自保留，不能把sweep或历史schema混作当前正式结果。
本轮只读绑定文件身份，未重测速度、重算指标或将旧A100数字套给本页Spatial/Temporal模型。

### LightGenPublic 三份内部交付快照的区别（2026-10-06）

`LightGenPublic/tasks/t06_lgvq_temporal_consistency` 是保留的独立交付快照，
不是另一个长期开发入口。`teacher_release`、`teacher_release_final` 和
`teacher_release_final_v2` 的完整性检查分别通过75、77、77项文件；三份绑定
同一Temporal权重及35份冻结输入，共558条有效、唯一视频，padding不计入。

前两份记录的仿真SRCC为0.8043868643；`final_v2` 记录为0.8022806420。
后者新增8µm传播网格重采样，并改动探测器扰动实现；即使权重相同，也不是相同
推理实现。其继承的原运行commit标签不能单独证明新版运行源码身份。本页正式
0.8043868643/0.7977138739结果保持原身份，不用v2替换或混标。

这些数字来自既有报告，本轮没有重评数据或调用设备。三份原包、输入、权重、结果
与测速均原地保留；109份现存Python/YAML源码另以Git恢复引用封存，在本地及
训练服务器逐文件SHA验证。恢复引用不是新开发分支，增量bundle依赖已有main
基底，不是完整数据备份。具体身份见仓库
`maintenance/storage/T06_REVIEW_PACKAGE_IDENTITY_20261006.json`。

另有未跟踪的旧独立目录`LightGenV2/projects/lgvq_temporal`，其README记录的是
8µm重采样版本SRCC 0.8022806420，不是本页正式0.8043868643实现。本轮核对
46项manifest文件、同一权重和35场／558有效视频；发现`release.json`没有被
manifest绑定，完整性门不能宣称通过。目录内`sync_reference.py`会重写逐视频
参考及汇总，本轮没有运行、修补manifest或替换原报告。30份现存源码／配置已
在本地和训练服务器Git恢复引用内逐SHA保全，原资产仍原位；既有主入口中未发现
对该目录的直接静态路径引用，不代表排除了外部动态调用，暂不移除目录。
见`maintenance/storage/T06_WORKING_PROJECT_IDENTITY_20261006.json`。

2026-10-08该旧独立8µm重采样仿真副本已完整收拢至
`archive/legacy_lab_projects/lgvq_temporal_8um_snapshot`，不再作为第二个日常工程入口。
移动前本机未观察到使用该路径的进程；所有13份注册工作树的已跟踪运行脚本／配置
未发现对该路径的直接引用。309个文件、732,387,612字节（含原PT、输入、相位、
运行记录及全部测速）移动后文件集合、长度、SHA全部一致，没有删除或重生成资产。
本段覆盖下方10月7日“原位不动”的历史状态，仅此完整旧副本移动，正式Temporal
工程及其资产不动；若需运行旧命令，先按收据完整恢复原目录，不能套作正式.8044版本。
见`maintenance/storage/T06_WORKING_PROJECT_ARCHIVE_20261008.json`。

2026-10-07再次核验这30份旧源码/配置与本地、Linux既有Git恢复对象完全一致，
仅按具体路径退出日常Git待提交列表；原文件和全部资产不动，新源码仍可见。
此举不是将该8µm副本提升为正式模型，也不是修复其release完整性或批准删除目录。

2026-10-06历史Spatial收尾核验：两套旧工程的141份未跟踪源码及配置已完整Git归档，
原文件和所有结果／测速保留。主线57项模型合同测试通过；s586及旧卷积128读出候选的
历史／主线原有配置字段、架构和权重键一致，有界合成输入预测误差为0。
另检查101份旧release配置：92份原有参数和架构一致，9份早期RGB试验缺少额外Vision
采样视图对应的raw-frame缓存，主线拒绝加载；不取消对齐检查或把这些试错配置改成
正式默认。它们的源码／原配置仍能从归档恢复。此检查没有加载正式PT或数据集，
不是重测性能、光路复现或全部机器迁移验收。证据见仓库
`maintenance/storage/SPATIAL_SOURCE_CLOSURE_20261006.json`。

原数据/划分、缓存、best/last、全部原CCD/收据、逐视频结果、部署更新和诊断均不移动。
Temporal-36/9×4/16×4、Spatial不同电子容量、自研卷积及被否决的预训练上界各自保留
身份；上界仍标为不合规，不改为正式方案。全部历史A100/5090D测速和功率证据保留，
且只绑定当时的源码/PT/工作负载，不套给本页模型。

机器可读盘点在仓库 `maintenance/storage/T06_RUNTIME_IDENTITY_20261004.json`。
它是限定范围审计，不是全盘数据备份，也不是批准删除旧目录的清单。
