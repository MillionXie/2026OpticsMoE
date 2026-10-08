# 历史 DVP 调试区，不是封存 rank72 的运行入口

2026-10-08更新：本机27份旧CMD已完整归档到
`archive/legacy_lab_projects/t07_old_cmd_20261008`。原命令含两份MNIST测速启动器，
共12,485字节，全部逐文件SHA保留，未执行或删除任何测速。
主线已跟踪运行源码／配置未引用这些CMD，未观察到本机相应启动进程；
归档后27份文件集合、长度、SHA与原清单一致。下文“原CMD可见/原位不变”是旧状态。
需要复查旧命令时，按原清单完整恢复原目录及当时源码/设备合同，不能当成rank72入口。
清单 `.codex_tmp/storage_git_backup_20261002/t07_old_cmd_archive_20261008.json`，
SHA `1b2c11500b566c54a246de22c53dacf617f1cf0f2caf8744ff664c2dd854d151`。
Python依赖、实际数据和原报告未移动；未操作服务器旧运行目录或实验室电脑。

本目录原文件保留于本机，用于解释2026-09-23至26日的设备调试和历史命令。
静态MNIST/菲涅尔/双SLM对齐包的原始说明和原训练摘要，见
[历史测试包](../../../../MNIST_10cm_8um_Bench_Test_20260923/README.md)。
其67张BMP保留为私有资产，逐文件SHA见仓库
`maintenance/storage/MNIST_BMP_PAYLOAD_IDENTITY_20261006.json`；新克隆不包含BMP。
目录名在T07下，不表示里面所有实验都是ABO检索，也不表示这些本机文件已全部发布main。
不要直接从这里启动采集；封存rank72请从[任务页](../README.md)和
[正式硬件源码入口](../hardware/README.md)进入。

## 原先用于什么

| 脚本族 | 原用途 | 与最终版的区别 |
| --- | --- | --- |
| `run_abo_brightness_health`、`run_abo_illumination`、`run_abo_input_replay`、`run_abo_sdk_preload_probe` | 检查照明、输入显示及SDK初始化边界 | 诊断不是检索结果，不代表最终合同 |
| `run_exposure_scan`、`run_four_image_flow`、`run_full_query_flow*` | 旧DVP相机曝光扫描、四图流程和旧查询采集 | 固定旧几何及20,000µs等配置，不能套给最终SHS 400µs合同 |
| `run_mnist_*` | MNIST数字识别、相位对照、曝光及设备测速 | 独立的数字实验，不是ABO图搜图 |
| `run_mnist_timing*` | MNIST设备时序测量与重试 | 命令、原报告及所有测速副本保留；不能作为rank72专属测速 |
| `run_shs*`、`run_tune*` | 较早SHS采集及末端适配 | 历史权重/读出流程，不是仅换PT即可运行的rank72架构 |

这些CMD绑定实验室旧目录、固定Python环境、原输出位置；部分使用重定向覆盖日志。
`mnist_dvp_bench.py`还按旧独立工程层级寻找Camera、PhaseHDMI、Holoeye驱动和
仓库外MNIST相位/SDK。直接在现在的LightGenV2路径执行会采用不同路径基准，
**不能把文件已归到任务目录等同于接口已迁移可运行**。

## 保留和归并边界

原数据、有效CCD、收据、相位、结果及全部测速不移除。本页只是明确用途与入口，
没有执行设备、训练、重新评估或修改原脚本。历史Python恢复记录见仓库
`maintenance/storage/REMAINING_CLEANUP_PLAN_20261003.md`中具名源码备份；
恢复身份不是现场可运行证明，尚无恢复身份的CMD仍需继续核对，不能当作垃圾。

后续只在确认独有代码、依赖和占用并验证恢复包之后归档冗余启动副本。
正式rank72的仿真、实拍、权重和测速口径以任务页为准，不能从这些旧命令推断。

2026-10-06追加：已阅读的16份旧ABO/MNIST CMD保存于
`refs/archive/reviewed-t07-launchers-20261006`，恢复提交
`8afcf98702f35473cd1970fcbc1bc4f2ef30bcb5`。本机原文件不变，训练服务器和实验室
源码对象库均逐文件核验16份原始字节SHA相同；HEAD及开发分支未改变，不执行CMD。
增量恢复包3229字节，SHA256
`c0240a6fecdcbb9dabe22356a3bf6389b7cb099bc1318d65d16997812b3b3350`，
前置提交为`d8d9bf4015942f3cbf55cfd1ee0c37a2df052977`，不是完整数据或独立仓库备份。
私有清单和收据为`.codex_tmp/reviewed_t07_launchers_manifest_20261006.json`及
`.codex_tmp/reviewed_t07_launchers_recovery_sync_20261006.json`。
原CMD仍可见，尚未证明可删除或可现场运行；两份MNIST timing启动命令同样原样保留。
