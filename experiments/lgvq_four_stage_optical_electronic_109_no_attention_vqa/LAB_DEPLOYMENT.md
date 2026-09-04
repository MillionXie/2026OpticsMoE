# LGVQ 四帧光电模型：实验室部署顺序

本文是实验人员唯一需要顺序执行的部署文档。所有命令均在解压后的工程根目录、
PowerShell 中运行。不要跳过编号，不要把不同阶段的相位或 CCD 目录交叉使用。

## 0. 先明确“4 层”和“6 次曝光”

网络有 4 个光电融合层，但两个 Top-2 router 也要先走一次光路，因此完整样本需要
按固定顺序完成 6 次物理传播：

| 次序 | 光学 pass | 作用 | 四帧状态 |
|---:|---|---|---|
| 1 | `stage1_router` | CCD 四个区域产生每帧 4 个专家权重，取 Top-2 | 4 帧并行，4 个 232×232 lane |
| 2 | `stage1_expert` | 每帧所选 2/4 专家传播 | 4 帧并行 |
| 3 | `stage2_global` | 每帧全局相位传播 | 4 帧并行 |
| 4 | `stage3_router` | 四帧已由固定桥接合为一条 68-token 序列，再选 Top-2 | 单光场，但同时包含四帧信息 |
| 5 | `stage3_expert` | 序列级所选 2/4 专家传播 | 单光场 |
| 6 | `stage4_global` | 序列级全局相位传播 | 单光场 |

“四帧并行”不是连续播放四张振幅 BMP：在 pass 1–3 中，每个样本的一张
1024×1024 BMP 已同时包含四个帧 lane。

## 1. 环境与文件完整性

```powershell
Set-Location <解压目录>
conda activate xml
python VERIFY_BUNDLE.py
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果需要安装依赖，先按显卡驱动安装合适的 PyTorch，再执行：

```powershell
pip install -r experiments\lgvq_four_stage_optical_electronic_109_no_attention_vqa\requirements-lab.txt
```

正式初始权重为：

```text
experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/deployment/checkpoints/best_observed_test_checkpoint.pt
SHA256 = d357fe51b888ecace74c050096febebc09c08abc21d6b42f533ecc3cf1f4de55
epoch = 75
```

## 2. 一次性硬件标定

只编辑 `experiments\lab_lgvq\LAB_CONFIG.yaml`：选择真实存在且经当前光路验证的 LUT、
填写曝光、相位 SLM 的 `center_xy`/导出翻转关系与四个逻辑角点。新实验台还未有四点时，
先保持四项全为 `null`。六条 `export-pass` 命令都会自动读取该文件，不要逐条手写不同中心。

### 2.1 双 SLM 共面/倍率检查

文件在：

```text
experiments/lab_lgvq/calib/dual_slm_k1/00_k1_ready_to_play
```

三个子目录均是严格配对的 `amplitude_1024x1024.bmp` 与
`phase_1920x1200.bmp`。振幅合同为 `255=亮/透光、0=暗/关闭`；相位 BMP 已包含
`LAB_CONFIG.yaml` 中声明的导出翻转（默认仅纵向翻转），播放软件不得再次翻转。先用
`k=1` 对准；只有确认 4F 存在倍率误差时才
重新生成倍率扫描，不能用缩放去掩盖旋转或错位。

正式相位 BMP 使用 `floor(mod(phase,2π)/(2π)×256)` 的 uint8 编码。这要求 8 μm
相位 SLM 及播放软件已在 532 nm 下完成“灰度→相位”的器件响应校正；本包不包含另一块
相位 SLM 可通用的响应 LUT。若师姐的相位器件响应不同，应先单独标定器件，不能修改网络
相位图来补偿未知的灰度曲线。

相位检查图位于项目的 `deployment/hardware_assets/visualization`：

- `phase_masks_logical_overview`：六个 478×478 逻辑相位面；
- `phase_expert_tiles_stage1` / `phase_router_and_stage3_tiles`：专家与 router 细节；
- `phase_masks_physical_1920x1200`：真实 BMP 全面板、1016×1016 映射框与相位中心；
- `router_detector_rois`：统一使用一基编号 E1–E4 的 CCD 路由区域。

### 2.2 10 cm 焦面与 CCD 四点

菲涅尔文件在 `experiments\lab_lgvq\calib\fresnel_10cm`。振幅始终加载
`A_WHITE.bmp`（全 255），相位依次为 `P1_POINT.bmp`、`P4_POINT.bmp`、
`P9_POINT.bmp`。它们使用老师 MATLAB 方窗二次相位原理；P4 四个焦点对应
478×17 μm 有效场的四个物理顶点。

先生成 bootstrap 配置：

```powershell
python -m experiments.lab_lgvq.prepare_lab
```

用 P4 捕获一张全传感器/设备 ROI 图（程序只播放清单中的 `A_WHITE.bmp`）：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_lgvq\generated\bootstrap_hardware.yaml `
  --input-dir experiments\lab_lgvq\calib\fresnel_10cm `
  --output-dir experiments\lab_lgvq\work\fresnel_p4 `
  --log-dir experiments\lab_lgvq\work\fresnel_p4_log `
  --file-manifest experiments\lab_lgvq\calib\fresnel_10cm\amplitude_manifest.csv `
  --phase-mask experiments\lab_lgvq\calib\fresnel_10cm\P4_POINT.bmp `
  --clear-output
```

把四个焦点的 2048×2048 全传感器坐标，按光场逻辑身份填入
`LAB_CONFIG.yaml` 的 `top_left/top_right/bottom_right/bottom_left`；角点本身不要求是
4 的倍数。然后再次运行：

```powershell
python -m experiments.lab_lgvq.prepare_lab
```

程序会自动外扩并量化合法 TUCam 硬件 ROI，生成并锁定 478×478 单应性合同。正式
`ccd_captured/*.png` 是：设备 ROI 原始强度 → 单应性双线性变换 → 固定位深保存。
没有背景扣除、逐帧 min-max、自动翻转或网络外 log。

### 2.3 曝光/LUT

压缩包默认使用实际随包提供的原厂文件
`slm7930_at532-70c-pixel-2.lut`，因此第一次 `prepare_lab` 不会因缺少 LUT 而失败。
原厂 LUT 只作为新光路的安全起点，不代表已完成线性化。

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration exposure `
  --config experiments\lab_lgvq\generated\formal_hardware.yaml
```

若同一块振幅 SLM 的当前光路已经有通过 128 点验证的 `field_amplitude` LUT，复制该
LUT 到包内 `LUT Files` 并在 `LAB_CONFIG.yaml` 只填写文件名与 SHA256。若没有，先用
原厂 LUT 完成四点，再运行：

```powershell
python -m experiments.hardware_sdk.workflows.amplitude_lut_calibration all `
  --config experiments\lab_lgvq\generated\formal_hardware.yaml
```

只有报告 `recommended_for_use=true` 后才能切换到新 LUT；切换后重新运行
`prepare_lab` 和正常 32 点曝光检查。LUT 标定曝光和正式曝光原则上应一致，换光强或
曝光后要重新验证。

通过后，把 `LAB_CONFIG.yaml` 的 `amplitude_lut_filename` 改为
`slm7930_at532-70c-pixel-2_linearized-amplitude.lut`，并把报告中的 SHA256 填入
`amplitude_lut_expected_sha256`。不要把另一台实验台标定的线性化 LUT 直接当成当前光路的结果。

## 3. 正式部署变量

```powershell
$Project = 'experiments\lgvq_four_stage_optical_electronic_109_no_attention_vqa'
$Module  = 'experiments.lgvq_four_stage_optical_electronic_109_no_attention_vqa.hardware_bridge'
$Config  = "$Project\configs\deployment\lab_hardware_finetune.yaml"
$Ckpt0   = "$Project\deployment\checkpoints\best_observed_test_checkpoint.pt"
$Session = 'experiments\lab_lgvq\sessions\formal01'
$HW      = 'experiments\lab_lgvq\generated\formal_hardware.yaml'
```

先做 64 train + 32 test 小批量演练时，删掉下列命令的 `--all-data`。正式全量微调必须
保留 `--all-data`；同一 `$Session` 一旦生成，后续不允许改变样本清单。

## 4. Stage 1：光 router + 光专家

### 4.1 导出并采集 router

```powershell
python -m $Module export-pass --config $Config --checkpoint $Ckpt0 `
  --session-dir $Session --optical-pass stage1_router --all-data --device cuda `
  --no-reconstruct-amplitude

python -m experiments.hardware_sdk.workflows.acquire_folder --config $HW `
  --stage-dir "$Session\01_stage1_router" --clear-output

python -m $Module validate-capture --config $Config --checkpoint $Ckpt0 `
  --session-dir $Session --optical-pass stage1_router
```

采集程序会将 `compact_amplitude/*.png` 自动 1:1 重建为 1024×1024 BMP，并提示手动
加载唯一的 1920×1200 router 相位。router CCD 的每帧四区能量经标准化 softmax，
再取 Top-2；这一步已经取代电子 router。

### 4.2 用实测 router 决策导出并采集专家

```powershell
python -m $Module export-pass --config $Config --checkpoint $Ckpt0 `
  --session-dir $Session --optical-pass stage1_expert --all-data --device cuda `
  --no-reconstruct-amplitude

python -m experiments.hardware_sdk.workflows.acquire_folder --config $HW `
  --stage-dir "$Session\02_stage1_expert" --clear-output

python -m $Module validate-capture --config $Config --checkpoint $Ckpt0 `
  --session-dir $Session --optical-pass stage1_expert
```

### 4.3 微调 Stage 1 下游

```powershell
python -m $Module finetune --config $Config --checkpoint $Ckpt0 `
  --session-dir $Session --stage stage1 --epochs 100 --batch-size 16 `
  --test-interval 5 --device cuda

$Ckpt1 = "$Session\checkpoints\after_stage1_best_test.pt"
```

## 5. Stage 2：四帧全局光路

```powershell
python -m $Module export-pass --config $Config --checkpoint $Ckpt1 `
  --session-dir $Session --optical-pass stage2_global --all-data --device cuda `
  --no-reconstruct-amplitude

python -m experiments.hardware_sdk.workflows.acquire_folder --config $HW `
  --stage-dir "$Session\03_stage2_global" --clear-output

python -m $Module validate-capture --config $Config --checkpoint $Ckpt1 `
  --session-dir $Session --optical-pass stage2_global

python -m $Module finetune --config $Config --checkpoint $Ckpt1 `
  --session-dir $Session --stage stage2 --epochs 100 --batch-size 16 `
  --test-interval 5 --device cuda

$Ckpt2 = "$Session\checkpoints\after_stage2_best_test.pt"
```

## 6. Stage 3：序列光 router + 光专家

```powershell
python -m $Module export-pass --config $Config --checkpoint $Ckpt2 `
  --session-dir $Session --optical-pass stage3_router --all-data --device cuda `
  --no-reconstruct-amplitude

python -m experiments.hardware_sdk.workflows.acquire_folder --config $HW `
  --stage-dir "$Session\04_stage3_router" --clear-output

python -m $Module validate-capture --config $Config --checkpoint $Ckpt2 `
  --session-dir $Session --optical-pass stage3_router

python -m $Module export-pass --config $Config --checkpoint $Ckpt2 `
  --session-dir $Session --optical-pass stage3_expert --all-data --device cuda `
  --no-reconstruct-amplitude

python -m experiments.hardware_sdk.workflows.acquire_folder --config $HW `
  --stage-dir "$Session\05_stage3_expert" --clear-output

python -m $Module validate-capture --config $Config --checkpoint $Ckpt2 `
  --session-dir $Session --optical-pass stage3_expert

python -m $Module finetune --config $Config --checkpoint $Ckpt2 `
  --session-dir $Session --stage stage3 --epochs 100 --batch-size 16 `
  --test-interval 5 --device cuda

$Ckpt3 = "$Session\checkpoints\after_stage3_best_test.pt"
```

## 7. Stage 4：序列全局光路与最终结果

```powershell
python -m $Module export-pass --config $Config --checkpoint $Ckpt3 `
  --session-dir $Session --optical-pass stage4_global --all-data --device cuda `
  --no-reconstruct-amplitude

python -m experiments.hardware_sdk.workflows.acquire_folder --config $HW `
  --stage-dir "$Session\06_stage4_global" --clear-output

python -m $Module validate-capture --config $Config --checkpoint $Ckpt3 `
  --session-dir $Session --optical-pass stage4_global

python -m $Module finetune --config $Config --checkpoint $Ckpt3 `
  --session-dir $Session --stage stage4 --epochs 100 --batch-size 16 `
  --test-interval 5 --device cuda

$Ckpt4 = "$Session\checkpoints\after_stage4_best_test.pt"

python -m $Module evaluate --config $Config --checkpoint $Ckpt4 `
  --session-dir $Session --stage stage4 --batch-size 16 --device cuda
```

最终结果在 `$Session\final_evaluation`。本项目遵照当前实验要求，每隔 5 epoch 在 test
上评估并以 Spatial/Temporal 平均 SRCC 选权重；因此报告会明确写
`test_used_for_selection=true`，不能把该 test 再描述为未参与选模的独立盲测。

## 8. 每个光学 pass 的仿真—实测一致性

每条 `export-pass` 命令默认保存前 8 个样本的浮点理论 CCD；如需更稳定的统计，可在
导出命令末尾增加 `--theoretical-count 32`。对应 pass 完成采集和
`validate-capture` 后，运行以下命令：

```powershell
$AgreementModule = 'experiments.lgvq_four_stage_optical_electronic_109_no_attention_vqa.agreement_evaluate'
$AgreementPasses = @(
  @{ Dir = '01_stage1_router'; Pass = 'stage1_router' },
  @{ Dir = '02_stage1_expert'; Pass = 'stage1_expert' },
  @{ Dir = '03_stage2_global'; Pass = 'stage2_global' },
  @{ Dir = '04_stage3_router'; Pass = 'stage3_router' },
  @{ Dir = '05_stage3_expert'; Pass = 'stage3_expert' },
  @{ Dir = '06_stage4_global'; Pass = 'stage4_global' }
)
foreach ($Item in $AgreementPasses) {
  python -m $AgreementModule `
    --stage-dir "$Session\$($Item.Dir)" `
    --optical-pass $Item.Pass
}
```

每个 pass 输出到自身的 `agreement` 目录：

- `agreement_per_sample.csv`：逐样本 PCC、SSIM、gain-aligned NMAE；
- `agreement_summary.json`：汇总统计、输入 manifest 哈希与明确的预处理合同；
- `agreement_examples.png/.pdf`：Arial 7 pt 的低/中/高 PCC 代表样本对照图。

理论与实测只共同执行“非负截断 + 各自单帧均值归一化”。评估拒绝经过背景扣除、
逐帧 min-max 或未做单应性 canonical 化的采集，不搜索翻转、旋转、平移或尺度；图中的
99.5% 显示上限只影响可视化，不参与指标。该评估是诊断报告，不会修改 CCD 文件，也
不会改变微调输入。

## 9. 不可破坏的规则

- 后续 pass 必须由上一阶段微调后的 checkpoint 重新导出；不能把六张初始相位一次
  拍完后再统一微调。
- 每个 pass 的振幅 BMP、相位 BMP、CCD PNG、单应性合同和来源 checkpoint 都按
  SHA256 绑定。`validate-capture` 失败时禁止微调。
- 禁止增加 Transformer、attention、Qwen/VL backbone、电子 router 或可部署外部模型。
- 禁止 CCD 背景扣除、逐帧 min-max、搜索最佳翻转/平移。网络内部只使用训练时相同
  的非负截断、单帧均值归一化、相对强度截断和 `log1p`。
- 初始导出的六张相位仅用于仿真检查和第一步起点；已经实测封存的上游相位在后续
  微调中冻结，未采集的下游相位仍可训练。
