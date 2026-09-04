# Result evidence

本目录保存 2026-09-04 在 RTX 5090 D 上完成的正式 Temporal
4/9/16/25/36/49 帧速度测试。

- `per_video_measurements.csv`：4/9/16 帧底层证据，共 1,674 行；
- `per_video_measurements_36.csv`：36 帧底层证据，共 558 行；
- `per_video_measurements_25.csv`、`per_video_measurements_49.csv`：追加两档底层证据，
  各 558 行；
- `timing_summary_all.json/csv`：六档帧数最终统一汇总；
- `timing_summary_extended.json/csv`：第一轮四档汇总及当时的 49 帧条件判定；
- `RESULTS_EXTENDED.md`：服务器自动生成的可读摘要；
- `RESULTS_ALL.md`：最终六档服务器摘要；
- `extended_run_identity.json`、`additional_run_identity.json`：追加实验的配置、样本及
  文件身份记录。

每个 test 视频在每档帧数下只出现一次。第一轮按阈值跳过了 49 帧，后来用户明确要求
后又完成了 49 帧正式测试。结果解读与实验边界见上一级 `EXPERIMENT_RECORD.md` 和
`TIMING_REPORT.md`。
