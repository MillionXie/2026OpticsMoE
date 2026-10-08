# Ours 能耗重算

本目录是针对新高速相机功耗的独立重算结果；原始归档
`server_2026OpticsMoE_a100_formal` 没有修改。

## 光学功率

\[
P_{opt}=9.679+5.595+14.303+34.225+16.556=80.358\;W
\]

旧口径使用的相机功率为 14.333 W，因此旧光学功率为 80.388 W。新版只减少 0.030 W。

## 三种能耗口径

### 版本 1：总混合时间口径

这是原表的口径，但把光学功率更新为 80.358 W：

\[
E_{ours}^{(1)}=(P_{opt,new}+P_{GPU,hybrid})T_{hybrid}
\]

其中 `P_GPU,hybrid` 和 `T_hybrid` 来自 A100 原始报告。

### 版本 2：光学、电学分别计时

\[
E_{ours}^{(2)}=P_{opt,new}T_{optical}
 +P_{elec,serial}T_{elec,serial}
\]

`T_optical` 是物理传播/采集 pass 的总时间；`T_elec,serial` 是电子神经网络推理时间（含必要 bridge，不含排版、I/O、搬运）。电子功率使用对应原始报告的 `serial_electronic_measured_active_mean_power_w`。

这严格执行“各自功率乘以各自作用时间”。并行 residual 没有加入正式延迟；如需把 residual 的独立 GPU 能量也计入，可在此基础上另列一个保守上界，不能与本表混用。

为避免丢失信息，`energy_recalculation.csv` 另外保留了 `ours_energy_v2_with_parallel_residual_j`。它是“版本 2 + 已测并行 residual kernel 能量”的补充口径，不是本次主表的版本 2 数值。

### 版本 3：Baseline 按 250 W 估算

Ours 仍采用版本 2，Baseline 改为：

\[
E_{baseline}^{(250W)}=250\;W\times T_{baseline}
\]

这是 A100 板卡功率估算，不包含额外风扇、主机和外围设备，因此属于明确标注的估算上界/统一硬件口径，不是实测整机功耗。

## 任务级能效

\[
\eta=\frac{N_{workload}}{E(J)}
\]

LGVQ temporal 的 `N_workload=16`（16 个并行视频）；其余单样本任务取 1。单位为 `workload/J`，时间质量可直接写作 `video/J`。

能效提升倍数按：

- 版本 1：`E_baseline_measured / E_ours_v1`
- 版本 2：`E_baseline_measured / E_ours_v2`
- 版本 3：`E_baseline_250W / E_ours_v2`

逐任务完整数值见 `energy_recalculation.csv`，计算元数据和公式见 `energy_recalculation_report.json`。
