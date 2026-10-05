# OpenMoji rank64：主线入口与历史调度边界

本次核验只读取源码，不启动任务、设备、训练或正式数据评估。原实验目录不移动。

## 可维护的任务入口

主线任务目录 `LightGenV2/tasks/t04_openmoji_robust_ablation` 已提供显式身份的
`lab_shs_capture.py`、显式 G2/G5 的 `lab_tune2000.py` 和
`lab_calibrate_rank64.py`，以及两轮独立补充 TRAIN 准备入口。
原 PT、数据、设备路径仍须按任务复现说明配置；源码已发布不等于现场迁移完成。

## 不应作为新实验默认入口的旧脚本

下列原字节源码封存在 Git 对象
`bc1951cfbd0bcbd60a6111884286b506790e20fb` 的 `runtime_entries/`。
原目录继续保留；归类不是删除许可。

| 脚本 | 原用途 | 不能直接用于新工程的原因 |
| --- | --- | --- |
| queue_rank64_bench_20261002.py | 串行 G5 TEST、条件式 TRAIN/微调及 G2 pilot/TEST | 写死实验台根目录、1002任务名、6000张数量和当时 G1=.9385；不是最终 TRAIN2000 工作流 |
| queue_rank64_extra_tune_20261002.py | G5补充TRAIN1000后派发TRAIN2000微调 | 写死原PT SHA、旧目录/任务名，已有状态或输出拒绝重复派发 |
| queue_rank64_g2_tune_20261002.py | G2首轮TRAIN1000后派发旧微调 | 指向 g2_decoder_testselected，不是最终TRAIN2000结果 |
| queue_rank64_g2_resume_tune_20261002.py | 首轮G2采集恢复后的同一微调派发 | 与上一脚本主要差异是恢复任务名，不能误当独立模型版本 |
| verify_g2_rank64_saved_decoder_20261002.py | 固定旧G2 decoder PT严格重载并核对缓存指标 | 指向旧1000 TRAIN输出和 lab_tune_g2_test，不能证明最终校准PT已重载 |
| verify_rank64_adapted_sim_20261002.py | 已封存G5未校准/校准PT的理想仿真审计 | 写死两份PT SHA；运行会重新评估旧TEST并写报告，本轮不执行 |

旧队列的相位、收据、曝光与暗帧检查具有追溯价值，但不能仅以计划任务 Ready 或
旧 report.json 存在证明设备当前释放或新版本运行成功。新的运行配置必须显式绑定
源码 commit、原/适配 PT SHA、数据划分、硬件合同及实际任务/进程。

## 仍待完成

- 按当前主线入口核对实验台启动命令与完整资产配置，不能复用旧队列默认值。
- 在不占用现用实验的前提下核对设备依赖；现场回归需要另行排期，不以合成测试代替。
- 最终 PT 的严格重载以正式封存报告及其精确 SHA 为证据，不借用上述旧G2验证脚本。
- 本地默认 checkout 切换仍待人工确认；Git引用同步不代表三端实际运行目录已切换。

本文件只登记入口归属与迁移缺口，不改变正式结果、光学合同或 TEST 开发选模口径。
