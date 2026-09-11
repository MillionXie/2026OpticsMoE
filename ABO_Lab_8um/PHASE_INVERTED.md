# 本实验室相位 LUT 反向（2026-09-12）

用户已确认本实验室相位 SLM 的灰度映射方向相反：0→255、255→0。因此新增 `phase_slm.gray_encoding=inverted_255_minus_g`，最终 BMP 做精确 `255-g`，不修改训练权重，不附带空间翻转，不改振幅或 CCD。

在本工程根目录：

```powershell
python prepare_phase_inverted.py
```

生成 `LAB.phase_inverted.json` 和 `generated\phase_inverted\P`、`cal`、`dual`。缺少正式相位 NPY 时，可先用 `--calibration-only`。

已在本地生成六层相位和配套标定。旧的 `generated\P`、`generated\cal`、原始相位示例、旧 session 均不覆盖。新配置的零相位 `P_ZERO.bmp` 是全 255。

后续新实验在所有 `run.py` 命令添加 `--config LAB.phase_inverted.json`，并用新 session。修改已加载的旧 session 参数会触发硬件身份不一致，这是为了不混用数据。旧流程默认仍为 normal，师姐正常 LUT 不应套用这个反向设置。

旧光路电脑当前不可达，未自动替换其正在使用的图案。将本地新增的相位反向目录和对应配置同步后再开始新实验，不能把旧记录的 hash 手工改掉。
