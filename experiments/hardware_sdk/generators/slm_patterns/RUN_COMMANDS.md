# Commands

从仓库根目录执行：

```bash
python -m experiments.hardware_sdk.generators.slm_patterns \
  --config experiments/hardware_sdk/generators/slm_patterns/configs/slm_956.yaml
```

输出位于：

```text
experiments/hardware_sdk/generators/slm_patterns/generated/slm956_calibration/
```

## 17 µm / 8 µm：振幅反相、俄罗斯方块和相位倍率扫描

从仓库根目录执行：

```powershell
python -m experiments.hardware_sdk.generators.dual_slm_registration_sweep --config experiments/hardware_sdk/generators/slm_patterns/configs/dual_slm_17um_8um_inverted_scale_sweep.yaml
```

输出使用全新目录，不会覆盖旧的 `dual_slm_17um_8um_alignment`：

```text
generated/dual_slm_17um_8um_inv_large_blocks_k0p1/
├── 01_checker_c64_inv/
│   ├── amplitude_bmp/                 # 旧规则棋盘的整画布黑白取反
│   ├── phase_bmp_scale_sweep/         # 相位规则不变，21个k值
│   └── preview/
├── 02_large_blocks_c48_inv/
│   ├── amplitude_bmp/                 # 4/5/6/9格组成的简单大块
│   ├── phase_bmp_scale_sweep_x/       # 每张仅X方向0-pi光栅
│   ├── phase_bmp_scale_sweep_y/       # 每张仅Y方向0-pi光栅
│   └── preview/
├── scale_sweep_manifest.csv
└── alignment_scale_manifest.json
```

每组先固定播放它自己的振幅 BMP，再按 `phase_00`、`phase_01`……顺序测试相位。
大块组的 X 与 Y 光栅必须分开测试；任意一张相位图中都只有一个方向，不再在同一图案
内部交替横纵方向。规则棋盘组仍保留前一版相位不变。
播放顺序从无缩放开始，然后向正负方向逐步扩大。先保留近 `k=1` 的精细扫描，再扩展
到 `±0.1`：

```text
k = 1.0000,
    1.0005, 0.9995,
    1.0010, 0.9990,
    ...,
    1.0050, 0.9950,
    1.0100, 0.9900,
    1.0200, 0.9800,
    ...,
    1.1000, 0.9000
```

严格按照老师给出的关系：

```text
n = (8/17) × k × m
m = 17n / (8k)
```

其中 `n` 是振幅像素数，`m` 是相位 SLM 像素数。因此 `k>1` 会让相位图案略小，
`k<1` 会让相位图案略大。每一档实际得到的相位有效区像素数、理论单元尺寸、量化后
实现的 `k` 及其误差都记录在 CSV/JSON 中，避免只凭文件名猜测。

BMP 合同固定为：振幅 `1024×1024`、8-bit 灰度、仅0/255；相位
`1920×1200`、8-bit 灰度、仅0/128。相位继续使用原有纵向翻转和中心 `(980,590)`。

## 532 nm 菲涅尔阵列：焦面、翻转和 CCD ROI 标定

从仓库根目录执行：

```powershell
python -m experiments.hardware_sdk.generators.fresnel_phase_array --config experiments/hardware_sdk/generators/slm_patterns/configs/fresnel_phase_array_17um_8um.yaml
```

输出目录：

```text
experiments/hardware_sdk/generators/slm_patterns/generated/fresnel_phase_array_532nm_17um_8um/
├── amplitude_bmp/       # 1张全0、1024×1024、8-bit振幅BMP
├── phase_bmp/           # 15张1920×1200、8-bit相位BMP
├── preview/             # 相位有效区预览PNG，不用于硬件播放
├── fresnel_lens_centers.csv
├── fresnel_array_manifest.json
└── README.md
```

相位文件覆盖 `1/4/9` 个透镜和 `5/10/15 cm` 三个传播距离。4阵列和9阵列各有
两种版本：

- `uniform`：所有透镜口径相同，用于精确提取焦点中心和拟合ROI；
- `flip_coded`：仅“逻辑左上角”透镜的有效相位口径缩小到55%，焦点中心不变、
  强度较弱，用于消除完全对称阵列的上下/左右翻转歧义。

共同物理口径由输入端 `478×17 µm = 8126 µm` 决定，在8 µm相位SLM上量化为
`1016×1016` 像素（8128 µm，物理宽度误差仅 `+2 µm`）。相位中心为可配置的
`(980,590)`，所以有效边界为 `[472,82,1488,1098]`。2×2阵列的逻辑中心为：

```text
(726,336)   (1234,336)
(726,844)   (1234,844)
```

这些坐标采用“像素边界坐标”；对应像素索引中心应各减 `0.5`。导出BMP已经执行旧
光路使用的纵向翻转，CSV同时保存逻辑坐标和实际BMP坐标，不能再次手工翻转。

这里的 `478` 和 `508` 属于两套不同的原生像素单位，并不矛盾。478像素有效区分成
2×2后，每个方向的一半是239个17 µm像素；映射到8 µm相位SLM后，1016像素有效区
的一半是508像素：

```text
完整有效宽度：478×17 = 8126 µm；1016×8 = 8128 µm
相邻焦点间距：239×17 = 4063 µm； 508×8 = 4064 µm
```

所以四个透镜中心相对光轴的物理位置只相差0.5 µm，相邻中心间距只相差1 µm。
使用532 nm、8 µm采样和5 cm角谱传播进行数值复核后，四个最强焦点均落在理论中心，
误差为离散偶数网格坐标约定导致的±0.5个相位像素。

实验流程应是：数值全0振幅提供均匀照明（前提是当前振幅SLM极性中0确实为透光）→
播放菲涅尔阵列→沿z轴移动CCD寻找对应5/10/15 cm图案中最清晰的焦面→提取四焦点
中心。四焦点是四个半区的中心，不是有效区边界，因此轴对齐情况下要向外延伸半个
焦点间距才能得到完整ROI；有旋转/剪切时应将已知四中心拟合为仿射/单应映射，再映射
有效区四角。

建议依次播放：

1. 全0振幅 + `n1_1x1_uniform` 的5/10/15 cm版本，选择实际焦面；
2. 同一距离的 `n4_2x2_flip_coded`，通过唯一弱焦点确定四点对应及翻转；
3. `n4_2x2_uniform`，用四个等强焦点拟合CCD ROI；
4. `n9_3x3_uniform`，检查边缘畸变和非线性误差。

无明显旋转/透视时，若四焦点左右、上下中心间距分别为 `dx`、`dy`，ROI可由焦点
外推半个间距：`left=x_left-dx/2`、`right=x_right+dx/2`、
`top=y_top-dy/2`、`bottom=y_bottom+dy/2`。存在旋转、剪切或畸变时，应先用已知
的四个相位中心与实测CCD中心拟合仿射/单应映射，再映射有效区四角；不能只用轴对齐
外推。四个完全相同的焦点本身具有对称性，所以只播放 `uniform` 不能判断翻转。
