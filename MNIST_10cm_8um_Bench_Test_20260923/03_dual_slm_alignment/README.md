# 双 SLM 配套格子

每组 `A.bmp` → 振幅 SLM（1920×1080）；`P.bmp` → 相位 SLM（1920×1200）。`P_V.bmp` 只作为相位上下翻转对照，不能与 `P.bmp` 同时用。相位已做反灰度编码。建议从 `01_check64` 开始，再用 `02_blocks_x` 和 `03_blocks_y` 判断水平方向和垂直方向。`04_check16` 是更细格子，先粗对齐后再试。具体参数和文件 SHA 见各组 `pair.json` 和 `pairs.json`。
