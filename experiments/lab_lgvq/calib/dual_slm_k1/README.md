# 双 SLM 振幅与相位倍率扫描（normal）

只使用本目录中的两个编号文件夹。每个文件夹内先固定播放唯一的
`amplitude_bmp/*.bmp`，再按 `phase_bmp_scale_sweep/phase_00...phase_40` 的编号顺序
逐张测试相位。

- `phase_00_k1p0000`：无倍率修正，规则棋盘相位与旧版逐像素相同。
- 精细段：`+0.0005, -0.0005, ...`，直到 `±0.0050`。
- 大范围段：`+0.0100, -0.0100, ...`，直到 `±0.1000`。
- 振幅硬件合同：`255=白/透光`、`0=黑/遮光`；本目录
  `invert_before_export=false`，不要在播放软件中另行反相。
- 相位已经沿用旧版纵向翻转；不要在播放软件中再次翻转。
- 相位中心为 `(980,590)`。
- `02_large_blocks_c48_normal` 的 X/Y 相位分别位于两个目录；单张相位只有一个方向。
- `01_checker_c64_normal` 和 `02_large_blocks_c48_normal` 的振幅/相位不能交叉配对。

老师给出的关系按 `n=(8/17)×k×m` 实现，即 `m=17n/(8k)`。每一档实际尺寸、
量化误差和 SHA256 见 `scale_sweep_manifest.csv` 与
`alignment_scale_manifest.json`。
