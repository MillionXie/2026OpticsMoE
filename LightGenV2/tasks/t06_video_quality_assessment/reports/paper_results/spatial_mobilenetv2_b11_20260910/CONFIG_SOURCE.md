# 历史 MobileNet baseline 配置来源

本报告对应旧 Spatial s745 仿真对照，不是当前 Spatial/Temporal 实拍模型。
配置入口为仓库兼容后端的
`experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial_mobilenetv2_b11_rank_s745.yaml`。
其完整33级继承链中，五份此前未跟踪的配置现在纳入 main，原字段和相对路径不改：

- `spatial_mobilenetv2_b11_rank_s745.yaml`
- `spatial_mobilenetv2_b10_rank_s741.yaml`
- `spatial_mobilenetv2_b10_s740.yaml`
- `spatial_readout_moments_s640.yaml`
- `spatial_calibrated_s620.yaml`

2026-10-07实际读取训练服务器旧运行目录 `/DATA/DATA1/guest3/lightgen_spatial_065`
的这五份配置。本地三份与服务器原字节一致；s745及s740两份仅参数统计注释不同，
YAML计算字段一致。本地注释分别写361856及289024，服务器旧注释写361664及288896；
这不是模型容量变更。保留本地原文件，不覆盖用户独有修正，也不改服务器旧运行目录。
本地文件原字节可从既有Git提交 `ac7e3ecce0bef96ae7d91f8de6e478542913bd4e` 恢复。

此项只补齐配置依赖。报告中的数据、缓存、初始化PT及训练环境仍按原
`resolved_config.json` 识别；未重训、未评估、未重新测速，不能称完整新机复现通过。
