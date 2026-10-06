# 历史 DVP 调试区，不是封存 rank72 的运行入口

本目录原文件保留于本机，用于解释2026-09-23至26日的设备调试和历史命令。
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
