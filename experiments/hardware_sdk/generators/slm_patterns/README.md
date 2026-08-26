# SLM calibration BMP generator

生成居中的 8-bit 灰度标定 BMP。默认有效区为 8 µm 像素下的 956×956：振幅画布 1920×1080，相位画布 1920×1200。相位图按当前折叠光路在导出前上下翻转；振幅图不翻转。

振幅图包括均匀灰度/白场、棋盘格、十字、字母 A 和圆孔；相位图包括 0/π 平面、棋盘格、十字、字母 A，以及 532 nm 下的 5 cm/10 cm 薄透镜相位。尺寸、波长、焦距均可在 YAML 修改。

当前1024×1024 Meadowlark振幅SLM统一使用正常极性：`255=白/透光`、`0=黑/遮光`。
带 `inverted` / `_inv` 的旧目录只作历史留档，不用于新实验。

## 当前正式的菲涅尔 ROI 标定

请使用 `fresnel_roi_vertex_array_17um_8um.yaml` 和
`experiments.hardware_sdk.generators.fresnel_roi_vertex_array`。它生成的四点直接位于
478×17 µm有效输入映射到8 µm相位SLM后的四个**精确物理顶点**，四点间隔为
`478×17/8 = 1015.75`个相位像素；九点覆盖四角、四个边中点和中心。

历史 `fresnel_phase_array_17um_8um.yaml` 的四点位于四个半区中心，间隔只有508像素，
仅作历史留档，不能再把它的四个焦点直接解释为ROI顶点。
