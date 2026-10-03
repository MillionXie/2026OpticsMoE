# T13 唯一复现入口

2026-10-03整理：四组仿真训练及共同评价已完成，尚未完成新光路实采。原数据、正式best/last、父阶段和有效报告保留原位置；整理不重训、不改指标。此前“只有架构、缓存缺失”是早期状态，不再作为当前结论。

## 代码与执行环境

- 服务器任务根：`/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t13_temporal_robust_20260927/LightGenV2/tasks/t13_temporal_robust_training`。
- 本轮训练commit：`1aa3842c01ddb54c8b0d61f8428de391e11cdaaa`。
- 已保护源码HEAD：`493eb38375f3bd9e4e8e7edfe278468b351365f1`；增加结果文档与`public_simulation/`独立导出。原runtime与适配器不通过导出包替换。
- 历史命令用`/home/guest3/miniconda3/envs/xml/bin/python`；依赖见任务requirements.txt（torch2.6.0、numpy2.1.3、PyYAML6.0.2），开发测试另见requirements-dev.txt。本地CPU测试环境不是历史训练GPU环境。
- `reference/source_manifest.json`固定导师源快照，`python -I verify_source.py`校验22项SHA；2026-10-03合同CPU测试33项通过，未重评558视频。
- 服务器未跟踪的`baseline_holdout_recovery.py`保留待审，不是本报告入口，不能删除或混入正式身份。

## 正式run、权重和指标

当前run：`runs/simulation/full2250_testbest_phase_only15_s163_uuid1256_20260927`。
父run：`full2250_testbest_low_lr30_s163_uuid1256_20260927`；初始训练与缓存恢复证据均保留。

| 组别 | best epoch | 固定报告test SRCC | best_checkpoint.pt SHA256 |
| --- | ---: | ---: | --- |
| r0_post 基础部署 | 5 | .7927406697 | `6a83dfcd1089905fb1d9702a2339b0528e18d54d806d67d19e11aca856a8bd76` |
| r1_ccd_post +CCD | 15 | .7855666534 | `bc7b8bdec17a96164ce8b4c517fb73605b06a9f1d4d683329c7e49ce01522e03` |
| r2_ccd_dc_post +DC30 | 1 | .8035385943 | `800564b426f1bbd5892c3a11faa5b8545f395f2ec25d5160a1c41f5b3d06b42a` |
| r3_ccd_dc_intrain +训练内插值 | 0（父best） | .8007958329 | `7b7d99706514a978d70619ef63d6893a545fbbeb6e7e6fa679ad13b002dcbb1d` |

每组best/last在该run对应group目录。原comparison.json SHA256为`bb204e87695992f1d184d1f56965276d03b2d720c5987f72a0a9e18df4f891f1`；本地原副本与服务器相同，不能用导师参考PT替换四组PT。

完整train/test指标及限制见[正式报告](../FULL2250_TESTBEST_RESULTS.md)。四份生成工程在任务`projects/temporal_full2250_phase_only15_s163_20260927`，生成产物不进Git，不另维护四套模型源码。

## 数据与评价合同

原manifest2250TRAIN/558TEST不变，无VAL；四组初训同随机起点，随后共同30轮小学习率，最后15轮仅6个phase张量可训练、91个电子张量冻结。G5保留epoch0父best。TEST参与选模、不参与梯度，指标属于开发评价，不是独立泛化。

选PT：共同8µm/DC30、noise-scale=0的TEST SRCC。主报告：固定该PT、同8µm/DC30、noise-scale=1、noise-seed20260927。`final_test_clean_dc30`和`final_test_clean_dc20`是诊断，不重选PT。

数据身份在run的selection_split.json、assets_sha256.json；私有路径在paths.json。原视频、Qwen快照、TRAIN-only教师soft targets及两种重建前端缓存均保留；换机器只改私有路径，不改划分或以35个参考field充当TRAIN。

完整历史训练命令与固定权重评价见[COMMAND](../../COMMAND.md)。命令用于说明复现，不授权现在启动4卡训练。

## 部署和计时边界

4×4整幅相干场、16视频×4帧、6层、1视频1MOS，padding不计指标；532nm/10cm、478有效区/518逻辑画布、17µm逻辑pitch、8µm设备网格、DC30、保零tanh(abs/.5)、BMP固定255量化。

硬件资产仅为staged_not_capture_ready：须核验SDK、LUT、相位方向、独立幅相raster与forward等价、raw CCD/ROI及field身份。旧.7977实拍不是新四组实拍。558视频需35field，每组210CCD、四组840CCD只是计划数；当前无绑定本权重的端到端延迟或功耗。

## 导师参考与可恢复历史

导师PT SHA `5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c`，实际固定复评SRCC .8022806420；包名.8044是历史标识，不当当前结果。

原release.json逐field预测是历史记录。当前首场golden来自独立CUDA验证，SHA见reference/golden_first_field.json；旧本地CPU与CUDA首场最大差.0084343MOS，按.01门限验证，不是bit-identical或全558复评。

schema3及旧派生工程已隔离，原日志保留，不是schema4当前run。后续清理先审运行占用、唯一资产及下游依赖，不能按目录时间删除。
