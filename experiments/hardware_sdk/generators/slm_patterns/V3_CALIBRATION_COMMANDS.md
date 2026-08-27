# 17 µm / 8 µm 标定图：当前命令

所有命令从仓库根目录运行：

```powershell
Set-Location E:\code\guest\2026OpticsMoE
conda activate xml
```

## k=1 双 SLM 棋盘格与不规则大块

```powershell
python -m experiments.hardware_sdk.generators.dual_slm_registration_sweep `
  --config experiments\hardware_sdk\generators\slm_patterns\configs\dual_slm_17um_8um_normal_scale_sweep.yaml
```

无需倍率扫描时，只使用：

```text
experiments/hardware_sdk/generators/slm_patterns/generated/
  dual_slm_17um_8um_normal_large_blocks_k0p1/00_k1_ready_to_play/
```

其中三组分别是规则棋盘格、不规则大块 X 光栅、不规则大块 Y 光栅。每组只播放
同一子目录里的 `amplitude_1024x1024.bmp` 与 `phase_1920x1200.bmp`，禁止跨目录配对。
振幅极性为 `255=白/透光，0=黑/遮光`。

先用非对称 `large_blocks_c48_x/y`（或已知方向的 quadrant 图、数字 `3`）确定 CCD 相对
模型坐标唯一且固定的旋转/镜像关系。规则棋盘格可能存在对称歧义，不能单独用来判断方向。

## 532 nm、10 cm 普通菲涅尔 v3

```powershell
python -m experiments.hardware_sdk.generators.fresnel_square_aperture_array `
  --config experiments\hardware_sdk\generators\slm_patterns\configs\fresnel_square_aperture_array_17um_8um.yaml
```

输出目录：

```text
experiments/hardware_sdk/generators/slm_patterns/generated/
  fresnel_square_aperture_array_532nm_17um_8um_v3/
```

它生成 `n1/n4/n9 × a48/a64/a96/a128` 共 12 对匹配振幅/相位 BMP，并附理想 CCD、
数值指标和 SHA manifest。建议先用 `a64`；焦斑太小换 `a48`，光太弱换 `a96`。

相位只含普通二次菲涅尔相位，没有画十字或外圈。十字来自方形 pupil 的 sinc 旁瓣。
相位纵向翻转围绕配置的 `center_y=590` 执行，导出中心仍为 `(980,590)`，不是围绕
1200 行面板的几何中心翻转。播放端不得再次翻转、缩放或重新居中。

推荐标定顺序固定为：`n1` 找焦面 → 用前述非对称图确定方向 → `n4` 定位四顶点并按已知
方向赋逻辑 TL/TR/BR/BL → `n9` 用未参与拟合的边中点和中心做独立几何验证。n4 的四个
焦点完全对称，n4 图本身不能判断哪个焦点是 TL/TR/BR/BL；n9 也只验证几何误差，不用于
事后选择一个更好看的翻转方向。

## 一次性校验并生成实验室工具包

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.lab_validation_package `
  --check-only

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.lab_validation_package `
  --overwrite
```

打包器会拒绝旧中心语义、错误尺寸、错误极性、缺失配对或 SHA 不一致的图案。
