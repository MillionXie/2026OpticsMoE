# Commands

> [!IMPORTANT]
> 当前 17 µm 振幅 SLM / 8 µm 相位 SLM / 10 cm 新光路只使用
> [V3_CALIBRATION_COMMANDS.md](V3_CALIBRATION_COMMANDS.md) 中的 square-aperture Fresnel v3
> 和 `00_k1_ready_to_play`。本文件后面的 `fresnel_roi_vertex_array_..._v2` 只用于复现
> 已封存的旧 formal ZIP，当前对齐、找焦和 ROI 标定均不要播放 v2 文件。

从仓库根目录执行：

```bash
python -m experiments.hardware_sdk.generators.slm_patterns \
  --config experiments/hardware_sdk/generators/slm_patterns/configs/slm_956.yaml
```

输出位于：

```text
experiments/hardware_sdk/generators/slm_patterns/generated/slm956_calibration/
```

## 17 µm / 8 µm：正常极性、大块标定和相位倍率扫描

从仓库根目录执行：

```powershell
python -m experiments.hardware_sdk.generators.dual_slm_registration_sweep --config experiments/hardware_sdk/generators/slm_patterns/configs/dual_slm_17um_8um_normal_scale_sweep.yaml
```

输出使用全新目录，不会覆盖旧反相产物：

```text
generated/dual_slm_17um_8um_normal_large_blocks_k0p1/
├── 01_checker_c64_normal/
│   ├── amplitude_bmp/                 # 255=白/透光，0=黑/遮光
│   ├── phase_bmp_scale_sweep/         # 相位规则不变，41个k值
│   └── preview/
├── 02_large_blocks_c48_normal/
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
振幅命令不再反相，播放软件中也不得再次反相。

旧配置 `dual_slm_17um_8um_inverted_scale_sweep.yaml` 只用于复现极性修正前的历史文件，
当前实验不要使用。

## 532 nm 菲涅尔阵列修正版：焦点直接位于 ROI 顶点

> **历史封存 v2——当前新光路勿用。** 本节仅保留给旧 formal ZIP 的可追溯复现。
> 当前实验请返回顶部并按 `V3_CALIBRATION_COMMANDS.md` 生成/播放 full square-aperture v3。

从仓库根目录执行：

```powershell
python -m experiments.hardware_sdk.generators.fresnel_roi_vertex_array --config experiments/hardware_sdk/generators/slm_patterns/configs/fresnel_roi_vertex_array_17um_8um.yaml
```

输出到全新目录，不覆盖任何旧标定文件：

```text
experiments/hardware_sdk/generators/slm_patterns/generated/fresnel_roi_vertex_array_532nm_17um_8um_v2/
├── amplitude_bmp/
│   ├── amplitude_focus_full_white_1024x1024.bmp
│   └── amplitude_roi478_white_black_1024x1024.bmp
├── phase_bmp/                      # n1/n4/n9 × 5/10/15 cm，共9张
├── preview/
├── fresnel_focus_targets.csv
├── numerical_focus_validation.csv
├── numerical_focus_validation.json
├── fresnel_roi_vertex_manifest.json
└── README.md
```

精确几何关系：

```text
振幅有效宽度             = 478 × 17 µm = 8126 µm
相位SLM上的精确宽度       = 8126 / 8 = 1015.75 pixel
精确物理边界（edge坐标）  = [472.125,82.125] → [1487.875,1097.875]
量化承载边界（半开区间）   = [472,82,1488,1098)，1016×1016
```

四点图的四个焦点直接落在精确物理边界四角，不再外推；九点图落在四角、四个边中点和
中心。生成器不是只修改manifest：每个分区的二次相位中心直接使用目标ROI点，角点对应
向内quarter-lens，边中点对应向内half-lens。几何cell完整覆盖1016×1016且无重叠，
但为避免相位采样混叠，只有安全圆内写透镜相位：

```text
r_Nyquist = lambda*z/(2*p^2)
r_safe    = 0.92*r_Nyquist
5/10/15 cm的r_safe约为191.2/382.4/573.6个相位像素
```

安全圆外明确写0平相位；相位SLM无法把这部分变暗，因此四点/九点必须配合中央478白窗
黑底振幅图，不能把平相位区域误称为光学暗孔径。

BMP像素索引 `(x,y)` 的中心使用连续edge坐标 `(x+0.5,y+0.5)`。精确物理边界可能位于
两个像素中心之间，实验时应拟合光斑质心；最亮像素允许存在不超过0.5像素的采样误差。
相位BMP已经执行既有纵向翻转，播放端禁止再次翻转。manifest和CSV同时给出逻辑坐标与
实际导出BMP坐标。

推荐流程：

1. 播放全白振幅和 `n1` 的5/10/15 cm版本，沿z轴寻找实际焦面；
2. 改播中央478白窗黑底振幅和对应距离的 `n4_exact_roi_vertices`，直接测四个ROI顶点；
3. 播放同距离 `n9_exact_roi_vertices_edge_midpoints_center` 检查旋转、剪切和非线性畸变；
4. 查看 `numerical_focus_validation.json`。九张量化相位图均需同时通过：≤0.75相位像素
   位置误差、焦点一一对应、目标峰/全局背景中位数≥100、最弱目标峰/目标区外最强伪峰≥50。

四点/九点几何本身对称，不能仅凭等强光点判断翻转身份；应采用已知的BMP纵翻约定，
必要时另播非对称棋盘/光栅图确认方向。

## 历史532 nm菲涅尔分区中心阵列（留档，不能直接当ROI顶点）

从仓库根目录执行：

```powershell
python -m experiments.hardware_sdk.generators.fresnel_phase_array --config experiments/hardware_sdk/generators/slm_patterns/configs/fresnel_phase_array_17um_8um.yaml
```

输出目录：

```text
experiments/hardware_sdk/generators/slm_patterns/generated/fresnel_phase_array_532nm_17um_8um_normal_polarity/
├── amplitude_bmp/       # 1张全255、1024×1024、8-bit均匀照明BMP
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
使用完整1920×1200相位画布、全场均匀振幅、532 nm、8 µm采样和5 cm角谱传播
进行数值复核后，即使有效透镜区外保持平相位，四个最强焦点仍全部落在理论中心；峰值
约为背景中位数的 `3.1×10^4` 倍，位置误差为偶数网格坐标约定导致的±0.5个相位像素。

实验流程应是：全255振幅提供均匀照明（当前合同为255=白/透光，0=黑/遮光）→
播放菲涅尔阵列→沿z轴移动CCD寻找对应5/10/15 cm图案中最清晰的焦面→提取四焦点
中心。四焦点是四个半区的中心，不是有效区边界，因此轴对齐情况下要向外延伸半个
焦点间距才能得到完整ROI；有旋转/剪切时应将已知四中心拟合为仿射/单应映射，再映射
有效区四角。

建议依次播放：

1. 全255振幅 `amplitude_uniform_white_1024x1024.bmp` + `n1_1x1_uniform` 的5/10/15 cm版本，选择实际焦面；
2. 同一距离的 `n4_2x2_flip_coded`，通过唯一弱焦点确定四点对应及翻转；
3. `n4_2x2_uniform`，用四个等强焦点拟合CCD ROI；
4. `n9_3x3_uniform`，检查边缘畸变和非线性误差。

无明显旋转/透视时，若四焦点左右、上下中心间距分别为 `dx`、`dy`，ROI可由焦点
外推半个间距：`left=x_left-dx/2`、`right=x_right+dx/2`、
`top=y_top-dy/2`、`bottom=y_bottom+dy/2`。存在旋转、剪切或畸变时，应先用已知
的四个相位中心与实测CCD中心拟合仿射/单应映射，再映射有效区四角；不能只用轴对齐
外推。四个完全相同的焦点本身具有对称性，所以只播放 `uniform` 不能判断翻转。
