# 六 pass 导出样例（仅用于格式检查）

本目录的每个 pass 含 4 个 train + 4 个 test 样例：

- `compact_amplitude/*.png`：478×478 紧凑振幅；
- `amplitude_to_play/*.bmp`：1024×1024、17 μm 原生 1:1 振幅 SLM 文件；
- `phase_to_play/*.bmp`：1920×1200、8 μm 相位 SLM 文件；
- `theoretical_ccd/*.npz` 与 `*_log_preview.png`：仿真强度及可视化。

这六组样例都从同一个 epoch-75 初始 checkpoint 以 `--simulate-upstream` 生成，只用于确认
文件命名、尺寸、四帧布局和硬件能否读取。它们不是逐层实测闭环，也不能直接作为正式
微调数据。正式实验必须按根目录 `README_FIRST.md` 依次采集，且每完成一个融合 stage 后
使用新 checkpoint 导出下游 pass。

