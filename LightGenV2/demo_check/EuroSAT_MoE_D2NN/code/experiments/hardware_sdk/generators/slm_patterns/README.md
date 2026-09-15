# SLM calibration BMP generator

当前 17 µm / 8 µm 双 SLM、k=1 即播套装及中心保持的 Fresnel v3 命令见
[V3_CALIBRATION_COMMANDS.md](V3_CALIBRATION_COMMANDS.md)。

生成居中的 8-bit 灰度标定 BMP。默认有效区为 8 µm 像素下的 956×956：振幅画布 1920×1080，相位画布 1920×1200。相位图按当前折叠光路在导出前上下翻转；振幅图不翻转。

振幅图包括均匀灰度/白场、棋盘格、十字、字母 A 和圆孔；相位图包括 0/π 平面、棋盘格、十字、字母 A，以及 532 nm 下的 5 cm/10 cm 薄透镜相位。尺寸、波长、焦距均可在 YAML 修改。

当前1024×1024 Meadowlark振幅SLM统一使用正常极性：`255=白/透光`、`0=黑/遮光`。
带 `inverted` / `_inv` 的旧目录只作历史留档，不用于新实验。

## 当前老师 MATLAB 方窗 Fresnel

新光路使用 `fresnel_full_panel_17um_8um.yaml` 与
`experiments.hardware_sdk.generators.fresnel_full_panel_array`。它按老师 MATLAB 的
方窗二次相位公式生成 P1/P4/P9，振幅固定全白，焦距 10 cm，并保持相位导出中心
`(980,590)`。P4 外间距严格为 `478×17/8=1015.75` 相位像素。详见上方当前命令文档。

## 历史 v2 ROI 顶点标定

旧封存实验仍使用 `fresnel_roi_vertex_array_17um_8um.yaml` 和
`experiments.hardware_sdk.generators.fresnel_roi_vertex_array`。它生成的四点直接位于
478×17 µm有效输入映射到8 µm相位SLM后的四个**精确物理顶点**，四点间隔为
`478×17/8 = 1015.75`个相位像素；九点覆盖四角、四个边中点和中心。

历史 `fresnel_phase_array_17um_8um.yaml` 的四点位于四个半区中心，间隔只有508像素，
仅作历史留档，不能再把它的四个焦点直接解释为ROI顶点。
