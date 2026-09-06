# Runs

本目录只保留说明，实际产物被 Git 忽略。服务器当前正式仿真只保留：

1. `simulation/multivideo9x4_contentroute_d30_s114`：9 个视频 × 4 帧；
2. `simulation/multivideo16x4_rank_s163`：16 个视频 × 4 帧；
3. `simulation/qwen3vl_quality_tokens_r448_dataset_once_5090d_20260906`：冻结
   Qwen3-VL 的 448×448 纯电子 baseline，4/9/16 帧、两种输出头的逐视频性能与计时。

每个正式 run 的 PT 权重只允许有 `best_checkpoint.pt` 和
`last_checkpoint.pt`。指标、实际配置、命令和相位可视化仍保留；探索候选、周期 PT、
`phase_snapshots` 和单独 polish 目录在正式候选确定后删除。论文结果的唯一入口是
`reports/paper_results/`，不需要逐个打开 `runs` 猜目录含义。

- `smoke/<run_id>`：CPU 结构和梯度检查，确认后可删除；
- `simulation/<run_id>`：正式仿真训练和测试；
- `hardware/<run_id>`：六次光传播采集、逐阶段微调和最终实测。

请使用 T06 入口自动创建 run，不要手工建立 `new2`、`final_final` 等目录。
