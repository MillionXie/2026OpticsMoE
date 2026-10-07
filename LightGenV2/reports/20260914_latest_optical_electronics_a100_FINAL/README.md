# A100 最新光学 MoE 电子部分计时（2026-09-14）

## 正式口径

- GPU：NVIDIA A100-PCIE-40GB；FP32 eager，未使用 `torch.compile`。
- 每个组件每次实验先预热 50 次，再测 1000 次；两次独立纯延迟实验共保留 130,000 条逐调用记录。
- 论文表采用两次实验全部样本的 pooled CUDA Event median，不挑选较快的一次。
- 正式电子时间只包括 CCD 读出/非线性/融合、光路由权重计算、必要 bridge 和任务头。
- SLM 画布排版、scatter、文件 I/O 不计入神经网络推理；这些操作仍保存在各原始运行的 full-reload audit 中。
- 电残差与光过程并行，因此不加到延迟；其时间与能量均单独保留。
- 每次物理光过程为 0.714 + 0.300 + 0.0307 = 1.0447 ms。

## 文件

- `all_per_call_timings.csv`：两次纯延迟实验的全部逐调用 Event/Wall 数据。
- `pooled_component_summary.csv`：各组件 2000 次调用的 mean/median/std/P05/P95/min/max。
- `paper_summary.csv`：可直接填表的性能绑定、电子明细、物理时间、总时间和能耗代理。
- `consolidated_report.json`：完整机器可读合同、指标来源、能耗定义和组件次数。
- `power_samples.csv`：独立功率实验的 10 ms 板卡功率原始采样。
- `source_runs/`：两次延迟实验与一次功率实验各自的原报告、命令、环境和日志。
- `latest_lgvq_source_snapshot/`：0.6710 空间版本实际测速采用的最新源码快照。

## 必须披露的指标差异

- ABO 图搜文 0.8058 是固定 epoch-25 EMA 的历史 1934/2400；当前环境完整复核少 3 张，不能写成完全复现。
- LSP 当前绑定的 `final_report.json` 是 0.7347857143，截图的 0.7353 尚未找到逐样本/权重证据，论文定稿前需确认来源。
