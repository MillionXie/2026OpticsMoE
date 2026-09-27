# T13 时间一致性鲁棒训练与光路消融

这是基于导师审阅版的独立研究分支。旧导师包不修改；本目录不运行时依赖父仓库、原 T06 或其他任务。当前状态：**代码框架与 CPU 测试；尚未正式训练、尚未部署采集**。

## 来源与版本边界

最新导师包位于：

```text
LightGenPublic/tasks/t06_lgvq_temporal_consistency/
  teacher_release_final_v2/lgvq_temporal_08044/
releases/LGVQ_Temporal_08044_teacher_final_v2_20260923.zip
```

其权重 SHA256 为 `5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c`。
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

这是**累计措施的条件增量消融**，不是完整 factorial；G3-G2 是无 DC 训练前提下的 CCD 收益，G4-G3 是已有 CCD 训练前提下的 DC 收益。不能宣称因素独立主效应或把增量跨排序泛化。第一轮 seed=163；单 seed 只能做 demo，后续有预算再重复关键配对。

## 保持不变的合同

- 任务：LGVQ Temporal consistency，16 个视频×每视频 4 帧；4×4 整幅相干场，6 次传播，1 视频1 MOS；padding 保留在场中但不计指标。
- 架构、电子支路、归一化、优化器、训练轮数和相位参数量相同；保持 532 nm、10 cm、478 有效区、518 逻辑画布、17 μm 逻辑 pitch、0.5° 截止。
- 仅研究用户指定的三个措施；三类位移、phase dropout、router logit noise 统一关闭，避免捆绑第四因素。普通网络 dropout/训练损失保留且各组相同。
- G2–G4 训练传播网格为 518/17 μm；G5 为 1101/8 μm，但可学习 mask 仍是原逻辑参数，不通过增加 mask 参数量取胜。
- 所有组选择 checkpoint 时都在同一 8 μm 网格、eta=0.20 的 TRAIN 留出验证集评估；测试也统一设备条件。不会在 G2 的 17 μm 名义性能和 G5 的 8 μm 性能之间直接作不公平比较。
- 物理孔径保持约 8.126 mm；8 μm 设备有效图 round(478×17/8)=1016。不能保持像素数却改变物理孔径。

## 三项建模的实际含义

相干未调制直流继承导师后端：

```text
M = sqrt(1-eta) * exp(i*phase) + sqrt(eta)
```

eta 训练区间 0.20–0.35，测试名义值 0.20，是待标定的名义系数；不等于 CCD 上固定百分比的实测功率。它不是相机黑电平，也不是 phase DC 正则。即使 G2/G3 训练忽略直流，部署评价也必须保留实际直流。

探测噪声沿用导师包可选模型：每个完整场均值标度的有偏截断 Gaussian，随后相对 photon-count Poisson + straight-through 梯度。参数为经验值，不称完整/已标定 CCD sensor model。正式训练前优先确认暗场、平场和重复帧统计；未标定只能显式授权为 pilot。`--allow-uncalibrated-noise` 是披露开关，不是标定。

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
  verify_source.py 原导师源码完整性检查
  tools/           Git服务器同步与真实结果绘图
  tests/           条件、梯度、网格与恢复行为
  reports/reproduction/  唯一复现入口
  assets/          缓存/数据身份；忽略Git
  runs/smoke|simulation|hardware/  独立run；忽略Git
  releases/        最终包；忽略Git
```

运行方法见 [COMMAND.md](COMMAND.md)，图表设计见 [reports/EXPERIMENT_AND_FIGURE_PLAN.md](reports/EXPERIMENT_AND_FIGURE_PLAN.md)。不同任务不维护两份可变模型：旧包只用于参考，消融只在本目录变更，最后从通过验证的该分支构建新的导师交付包。

## 训练与数据口径

原 manifest 2250 train / 558 test。本研究从原 train 按固定 source basename 分组抽 20% 验证，减少同源生成变体进入两侧；缓存张量和全局索引不改变。正式前要核对 basename 是否足以标识源视频/场景，不足则改为明确 source_id 再冻结。

旧训练器的内部 `test` 字段重定向到这个验证集；原 test 在训练 payload 中标 `sealed`，不会反传或选模。历史文件名 `test_*` 保留以不改源快照，但它们此时是验证输出；新 summary/checkpoint 明确纠正 selection metadata。最终原558仍有历史反复选模局限，不能宣称是全新未触碰 test。

源包注明两个完整 Qwen-front 训练缓存已缺失；35 个打包测试 field 只用于固定评估，不是训练资产。`preflight` 检查 manifest、视觉缓存、语言缓存、train-only soft targets；存在性检查之后还要用后端验证数据身份，不能只有“文件在”就宣称可复现训练。

## 光路边界与下一步

`hardware.measured_ccd_boundary` 在 `_detector` 的逻辑518/478强度边界替换测量，不在1101设备传播网格里塞478 CCD；支持按六层依赖的连续prefix，保留原电子forward，不自动换相位。

`build_lab_package.py` 当前产生单组checkpoint、六张逻辑/设备相位与SHA清单，状态为 `staged_not_capture_ready`。**还不是能直接上设备的完整采集包**：必须接入经过验证的SDK、相位方向/LUT、物理有界振幅、raw CCD/ROI以及数据field身份。原 T06 的 pin-only load_model 和旧 propagator-level replay 不能直接用于新权重/8 μm网格。

采集按每组、每层全量完成，再进入下一层；不循环小批量六层，不混不同mask/曝光会话。558视频=35场，每组6×35=210次CCD采集，四组共840次（不含探针/复拍）。预计时长必须由实测端到端吞吐估算。

## 同步与安全

专用分支 `codex/t13-temporal-robust-20260927`。本地和服务器共享工作树都有无关修改，使用隔离worktree，不 pull/reset 它们；源码仅通过Git同步，数据/权重另走SHA清单。默认单GPU，本轮不启动训练。服务器密码仅交互输入，不写入代码、配置、日志或交付包。

完成条件：源码测试通过并Git同步；训练资产和设备合同预检通过；四组训练后共同评价；四组实采后冻结结果；最后再构建新版导师包并隔离验证。当前不能把搭框架说成训练或实采已完成。
