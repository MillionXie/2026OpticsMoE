# 光路标定与仿真一致性实验室工具包

本工具包与封存的 warmstart5 formal ZIP 完全独立。它不带 checkpoint、数据集或正式
81% 结果，也不会改写旧 ZIP 的校验和；用途只有三类：双 SLM/CCD 标定、短时亮度响应
标定，以及同一振幅和相位输入下的仿真—实测 CCD 一致性评估。

## 1. 安装位置与环境

把 ZIP **直接解压覆盖到仓库根目录**：

```text
E:\code\guest\2026OpticsMoE
```

所有命令均从这个目录运行。数据和 session 也放在相应项目的 `experiments/...` 下，
不要再使用 `E:\code\guest\20260826`、磁盘根目录或旧 bundle 的 `payload` 目录。

如果使用已有 conda 环境：

```powershell
conda activate xml
python -m pip install -r experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\requirements-lab.txt
```

提示符应只有 `(xml)`。如果前面还有 `(.venv_capture)`，先运行 `deactivate`，再运行
`conda activate xml`。采集/评估端不需要 Torch、Transformers 或 Qwen；
`agreement_export` 只在服务器模型环境中运行。

工具包独立保证的是 Windows 硬件控制/采集，以及不加载模型的 `agreement_evaluate` 和
`agreement_report`。包内的 `agreement_export.py` 与配置只作为版本一致的服务器端源码
参考；它依赖完整仓库中的 robust/grocery/Qwen 工程、checkpoint、模型 cache 和 Torch
环境，**仅解压本 ZIP 不能独立执行 export**，也不要继续向 ZIP 复制整套模型依赖。

解压后从仓库根目录做一次逐文件 SHA 校验：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.lab_validation_package `
  --verify-tree .
```

## 2. 先确认方向：只做一次 canonicalization

角谱仿真保持输入坐标方向，而实际 4F 中继可能产生 180° 旋转。若输入数字 `3` 在 CCD
上表现为上下和左右同时翻转，实测 CCD 必须在进入电子 readout/微调前变回模型坐标；
不能指望下一层再“自动抵消”。四层逐层实验的每个 CCD 边界都遵循同一规则。

推荐用四顶点透视变换一次完成几何校正和方向统一：四个源点必须按其**逻辑身份**标记为
`top_left, top_right, bottom_right, bottom_left`，而不是按相机图像中的位置排序。这样
homography 已经吸收旋转/镜像，变换后必须保持以下四项为 `false`：

注意，n4 的四个焦点几何上完全对称，**只看 n4 不能知道哪个点是逻辑 TL/TR/BR/BL**。
必须先播放 k=1 的非对称 `large_blocks_c48_x/y`（也可以用已知方向的 quadrant 图或数字
`3`）确定整套光路唯一、固定的旋转/镜像关系，再依据这个方向关系给 n4 四个焦点赋逻辑
标签。规则棋盘格和 n4 本身都不能单独消除方向歧义。

```yaml
orientation:
  flip_vertical_after_warp: false
  flip_horizontal_after_warp: false
  downstream_loader_flip_vertical: false
  downstream_loader_flip_horizontal: false
```

也就是说：使用 canonical homography 后，Qwen 配置中的 CCD 横/纵 flip 也应关闭；否则
会发生二次翻转。只有暂时继续使用旧的轴对齐 ROI 时，才保留原配置中的
`flip_vertical=true, flip_horizontal=true`。

复制模板、填入满传感器坐标中的四顶点，生成不可变合同：

```powershell
Copy-Item experiments\hardware_sdk\configs\detector_homography_478.example.yaml `
  experiments\hardware_sdk\artifacts\calibration\detector_homography_478.lab.yaml

python -m experiments.hardware_sdk.workflows.detector_homography fit `
  --config experiments\hardware_sdk\artifacts\calibration\detector_homography_478.lab.yaml `
  --output experiments\hardware_sdk\artifacts\calibration\detector_homography_478.contract.json
```

然后把合同路径和文件 SHA 固定进
`experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml` 的
`camera.detector_geometry`。采集顺序固定为：原始 uint16 设备 ROI → 一次 homography 到
478×478 正方形 → 固定输入范围量化为 uint8。禁止逐帧 min-max、逐帧最佳平移、gamma
拟合或没有真实暗场时的“背景扣除”。

## 3. k=1 双 SLM 对齐即播套装

目录：

```text
experiments\hardware_sdk\generators\slm_patterns\generated\dual_slm_k1_ready_to_play
```

它只有三组，不含倍率 sweep：

- `01_checker_c64`：规则棋盘格；
- `02_large_blocks_c48_x`：不规则大块，单一 X 光栅；
- `03_large_blocks_c48_y`：同一不规则大块，单一 Y 光栅。

每次必须播放同一子目录的 `amplitude_1024x1024.bmp` 和
`phase_1920x1200.bmp`。振幅固定为 `255=开/白，0=关/黑`；相位已经按旧约定纵向翻转且
中心为 `(980,590)`，播放端不得再翻转。这里的纵向翻转是围绕配置的光轴 `y=590`
反射有效内容，不是对 1200 行面板做整图 `flipud`；所以相位中心在导出 BMP 中仍是
`(980,590)`，不会错误地移动到 `(980,610)`。

## 4. 普通菲涅尔 v3：找焦面与 ROI

目录：

```text
experiments\hardware_sdk\generators\slm_patterns\generated\fresnel_square_aperture_array_532nm_17um_8um_v3
```

这里仍是 532 nm、10 cm 的普通二次菲涅尔相位，没有画十字，也不是 CGH。十字状焦斑来自
完整方形 pupil 的 sinc 旁瓣。每一个 `pair_id` 的振幅和相位必须配对播放；全零振幅不会
得到合同中的焦点。`n1/n4/n9` 分别对应中心、完整 478 光场四顶点、四顶点+四边中点+中心。
每个透镜都是完整、独立、未裁切的方形 pupil，不再把透镜挤在边缘后截成四分之一。
本工具包固定校验 center-preserving v3 manifest SHA-256：
`07914c0d92d4d3f772a423a987c0c71e8d65d3a7d2a6f345ac1f07079230a568`。

建议顺序：

1. `n1_a64px_z10cm` 找焦面；
2. 用 k=1 非对称 large-block/quadrant 或已知数字 `3` 确定固定方向；
3. 播放 `n4_a64px_z10cm`，按第 2 步方向给四个对称焦点赋 TL/TR/BR/BL，再拟合 homography；
4. `n9_a64px_z10cm` 只用未参与拟合的五个点做独立几何误差验证，不重新决定方向；
5. 焦点太小可换 `a48px`，信号太弱可换 `a96px`，`a128px` 最亮但焦点最紧。

小 pupil 才会得到更宽的主瓣/十字，增大 pupil 反而使焦点更小。`ideal_ccd_linear/`、
`ideal_ccd_log/` 和 `numerical_metrics.*` 是预先生成的理想 CCD 与数值检查，不能把 log
预览直接当线性强度定量比较。

## 5. 32 灰度 × 3 帧亮度标定

配置固定使用一个曝光、32 个覆盖 0–255 的灰度点、每点 3 帧；共 96 帧。0 是闭态漏光，
不是凭空获得的背景帧。关闭自动曝光并保持增益不变：

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration generate `
  --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml
```

`generate` 完成后，必须在相位 SLM 上固定加载它刚生成的
`experiments\hardware_sdk\artifacts\calibration\masks\phase\phase_zero.bmp`，并在全部
32×3 帧采集期间保持该相位图、曝光时间和增益不变。确认加载后再运行：

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration exposure `
  --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml
```

检查 `exposure_summary.json`、`slm_response.csv`、响应曲线和 0/128/255 共用显示范围的
预览。若出现饱和，调整固定曝光后整套重拍，不要逐灰度自动曝光。

## 6. 仿真—实测 CCD 多指标一致性实验

证据分为三组：人为设计的方向/频率/灰度探针、每类固定选择的 held-out 模型输入、固定
样本重复拍摄。主参考是 `transport_quantized`（包含实际 8-bit 振幅/相位导出量化）；
`ideal_model_fp32` 用来区分数字传输误差。不能只看 retrieval 精度，也不能只看 PCC。

单层 quick session 必须在“完整服务器仓库 + 模型环境”中导出。下面命令不能在只解压了
本工具包的实验室目录中运行；实验室电脑接收的是服务器已经完整导出的 session：

```bash
PROJECT=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5
CKPT=$PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt
AGREE=$PROJECT/hardware_sessions/agreement_run1

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m \
  experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_export \
  --config $PROJECT/configs/release/agreement_quick_language_global.yaml \
  --checkpoint $CKPT \
  --session-dir $AGREE \
  --stages language_global \
  --upstream-source simulation
```

把整个 `agreement_run1` 传到实验室电脑相同的 `experiments/.../hardware_sessions/`。对每个
stage 先重建 1024×1024 振幅 BMP，再人工加载该 stage 唯一的 phase BMP，然后采集：

```powershell
$STAGE = "experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\hardware_sessions\agreement_run1\04_language_global"

python -m experiments.hardware_sdk.workflows.reconstruct_slm `
  --stage-dir $STAGE --payload amplitude --hardware-profile meadowlark_17um

python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml `
  --stage-dir $STAGE `
  --file-manifest "$STAGE\amplitude_to_play\reconstruction_manifest.csv" `
  --validate-only

python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml `
  --stage-dir $STAGE `
  --file-manifest "$STAGE\amplitude_to_play\reconstruction_manifest.csv" `
  --clear-output
```

实验室端严格配对并评估、绘图：

```powershell
$SESSION = "experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\hardware_sessions\agreement_run1"

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_evaluate `
  --session-dir $SESSION

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_report `
  --evaluation-dir "$SESSION\agreement_evaluation" `
  --require-arial
```

输出同时包含 PCC、仿真信号域 PCC、SSIM、shape-NRMSE、统一亮度标定后的能量比、饱和率、
质心误差和仿真支持区外能量，并分别报告线性域和网络输入域。网络输入域对仿真与实测执行
完全相同的 `非负截断 → 单帧均值归一化 → 相对强度截断 → log1p → 478→224
AdaptiveAvgPool`。方向候选只作 calibration 诊断；若最佳候选不是 identity，应修复
session 固定 homography 后重拍，不能逐帧选择最佳翻转来提高指标。

## 7. 工具包本身如何生成

在服务器或完整仓库根目录运行：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.lab_validation_package `
  --check-only

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.lab_validation_package `
  --overwrite
```

输出位于项目的 `validation_bundles/`，包含 ZIP、`.zip.sha256` 和 `.zip.json`。打包器只选择
x64 振幅 SLM/TUCam 运行库，排除 x86、PDF、厂商示例图、checkpoint、数据集、CCD session
和旧 formal ZIP。
