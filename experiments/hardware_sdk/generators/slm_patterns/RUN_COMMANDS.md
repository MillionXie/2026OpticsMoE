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
