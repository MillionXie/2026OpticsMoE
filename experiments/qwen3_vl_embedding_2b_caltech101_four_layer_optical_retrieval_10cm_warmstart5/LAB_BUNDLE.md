# Warmstart5 实验室独立包

这个包对应唯一固定的 Stage-B EMA checkpoint：光电融合系数硬下限为 5%，
初始化值为 5.5%，固定仿真测试 Top-1 为 81.00%。打包器会同时核对 checkpoint、
四张 phase、quick210 transport 和离线末层 cache/state 的 SHA-256；任一来源不一致
都会拒绝生成 ZIP。
封存 checkpoint SHA-256 为
`6a27f54d8c869cce46150583383a127b0ba47b3d34503f5753aa23974ac1e55d`；正式 checker/
grating 与 Fresnel v2 也被固定 SHA 绑定，不能替换成其他“自洽但非本次确认”的文件。

这里的 5% 是残差融合系数下限，而不是光能量占比：

```text
alpha = 0.05 + 0.95 * sigmoid(raw_gate)
output = electronic + alpha * optical_delta
```

## 0. 正式双 SLM 与 ROI 顶点标定（必须先完成）

先成对播放下面两张图，把振幅 SLM 与相位 SLM 调到接近像素级重合：

```text
payload/calibration/dual_slm_checker_grating/amplitude_checker_255open_c64_1024x1024.bmp
payload/calibration/dual_slm_checker_grating/phase_grating_xy_in_255open_cells_c64_p8_1920x1200.bmp
```

振幅采用 255=开、0=关，不能反相；相位光栅只出现在 255 白格内，并已执行纵向
翻转。两张图必须作为一对使用，不能从旧目录换入同名的 primary/complement 图。
随后再使用正式菲涅尔目录：

```text
payload/calibration/fresnel_roi_vertex_array_532nm_17um_8um_v2/
```

该目录包含全白振幅、中央 478×478 白窗振幅，以及 n1/n4/n9 在 5、10、15 cm
的相位 BMP。Qwen 的名义传播距离是 10 cm：先用 n1 和全白振幅寻找焦面，再用
同一距离的 n4/n9 和中央白窗确定实际相机 ROI。

固定物理关系是：

```text
478 × 17 µm = 8126 µm = 1015.75 × 8 µm
phase centre = (980,590)
logical ROI vertices =
  (472.125,82.125)      (1487.875,82.125)
  (472.125,1097.875)    (1487.875,1097.875)
```

- n1 的一个焦点位于共同中心，用于找焦；
- n4 的四个焦点直接位于完整 ROI 四个物理顶点，横纵间距均为 1015.75，相机
  ROI 不需要再外推半个间距；
- n9 是四顶点、四边中点和中心组成的全 ROI 3×3 网格；
- 相位 BMP 已按现有硬件约定执行纵向翻转，未横向翻转，播放时不得二次翻转；
- `amplitude_roi478_white_black_1024x1024.bmp` 的中央 478×478 为 255，
  外围为 0，没有 resize 或插值。

旧 `fresnel_phase_array_532nm_17um_8um_normal_polarity` 中的 n4 是间距 508
的象限中心图，不是 ROI 顶点，不能作为正式标定。本项目打包器只把 v2 目录列为
`formal_roi_vertex_calibration`；厂商 SDK 自带示例即使随运行库存在，也绝不作为
本项目标定证据。包内 manifest、焦点 CSV 和数值传播报告用于复核坐标和 SHA。
菲涅尔相位采用 0.92 Nyquist 安全圆孔径，以抑制 5/10 cm 下的欠采样伪峰；
安全圆外是平坦相位而不是暗区，因此 n4/n9 必须配合 478 白窗振幅使用。完整操作
见 `payload/calibration/README.md`。

## 1. 两套相互独立的实验室环境

仅播放和采集时不需要 Torch、Transformers、Qwen 或模型权重参与计算：

实验室必须使用 64-bit Windows、64-bit Python 3.10/3.11，并先安装与相机、
Meadowlark PCIe 板卡匹配的 x64 驱动。Python、厂商 DLL 和驱动位数必须一致。
配置中的振幅 LUT 必须按 SLM 实际温度选择
`slm7930_at532_30C.lut` 或 `slm7930_at532_70C.lut`；不要只因文件存在就默认 30C。
正式 ZIP 只保留采集必需的 x64 DLL、`TUCam.py` 和 30/70 °C LUT；x86 库、厂商
示例图、旧菲涅尔图、PDF、IDE 文件和 WFC 示例均不入包。仓库 vendor 目录里的示例
即使存在，也不能作为本项目标定或性能证据。

```powershell
python -m venv .venv_capture
.\.venv_capture\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\requirements-lab.txt
```

如果实验室电脑还要直接微调第四层后面的轻量电子 tail，另建环境：

```powershell
python -m venv .venv_offline
.\.venv_offline\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\requirements-offline-finetune.txt
```

第二套环境只加载约 255,811 个可训练参数，不加载 Qwen、Transformers、原始
Caltech101 图片或光学传播器。弱 CPU 也能运行；若机器已有适配的 CUDA PyTorch，
可使用 `--device cuda` 加速。

## 2. ZIP 内容与明确排除项

核心目录如下：

```text
README_LAB_AND_SERVER.md
bundle_manifest.json
payload/
├── checkpoint/                       # 仅一个固定 Stage-B EMA checkpoint
├── four_phase_export/                # 四层 compact phase + 1920×1200 phase BMP
├── calibration/                      # 正式 v2 n1/n4/n9 ROI 顶点标定
└── quick210/
    ├── manifest.csv                  # 100 train + 10 gallery + 100 test
    └── 04_language_global/
        ├── compact_amplitude/        # 210 张 478×478 uint8 PNG
        ├── compact_phase/
        ├── phase_to_play/            # 固定 language_global phase BMP
        └── offline_downstream/
            ├── cache.pt              # 冻结边界：Language Block-2 输入 [L,192]
            ├── downstream_state.pt   # 255,811 参数的初始轻量 tail
            └── contract.json
experiments/hardware_sdk/             # 重建、播放、采集及精简后的 x64 vendor runtime
reference/training_evidence/           # Stage A/B 小型日志与固定测试 JSON
reference/qwen_project_source/         # 服务器训练代码/配置快照，仅用于追溯
```

ZIP 不包含 Caltech101 原图、Hugging Face/Qwen cache、teacher cache、CCD、
`amplitude_to_play` 全尺寸 BMP 或额外 checkpoint。`offline_downstream/cache.pt`
是唯一刻意保留的轻量 latent cache，用于在实验室电脑跳过大模型前向。

## 3. 第四层快速采集

从解压后的 ZIP 根目录执行。先把逻辑振幅重建为 1024×1024、17 μm、1:1 播放
BMP；该大文件只在实验室本地生成，不需回传：

```powershell
$STAGE = "payload\quick210\04_language_global"

python -m experiments.hardware_sdk.workflows.reconstruct_slm `
  --stage-dir $STAGE `
  --payload amplitude `
  --hardware-profile meadowlark_17um
```

重建清单固定为
`$STAGE\amplitude_to_play\reconstruction_manifest.csv`；预检和正式采集都必须绑定该
allowlist，目录中的其他或遗留 BMP 不会被播放；清单存在 `output_sha256` 时还会逐张
核对重建振幅，防止同名文件漂移。

手动向相位 SLM 加载：

```text
payload\quick210\04_language_global\phase_to_play\language_global.bmp
```

`--stage-dir` 会强制读取同目录 `phase_to_play\reconstruction_manifest.csv`，并核对
`output_sha256`；缺失清单或相位内容不匹配都会在打开硬件前终止。

首次使用新相机时，先在
`experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml` 中把
`camera.device_roi_xywh` 填成实验标定得到的 `[left, top, width, height]`；四个值都
必须是 4 的倍数，所以设备 ROI 尺寸应使用 472、476、480 等合法值，不能直接填
478。模板保留 `null` 是为了阻止未经标定的 ROI 被误当作正式数据。
checker/n1/n4 尚未得到 ROI 时，使用
`experiments\hardware_sdk\configs\tucam_meadowlark_calibration_windows.yaml` 保留完整
相机几何；该 bootstrap 配置只用于少量标定帧，不能用于正式 Qwen 采集。

然后先做静态校验，再正式播放 210 张振幅并采集。`--validate-only` 只检查运行库、
文件集合、相位 SHA 和 ROI 配置，不打开 SLM/相机，也不代表硬件连接、曝光或成像
已经验证；仍须实际试拍：

```powershell
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

相机先采集满足 4 像素对齐的设备 ROI；采集工作流仅在落盘前执行一次固定几何
resize，保存为与 manifest key 同名的 210 张 `478×478`、8-bit、PIL mode `L`
PNG。落盘后禁止再次 resize，也不做背景扣除、拉伸或翻转。相位 BMP 已纵向翻转，
播放端不再翻；CCD 文件按相机原方向落盘，模型加载后再按正式配置执行纵向+横向
翻转。

## 4. 可选：实验室离线微调第四层 tail

采集后先严格检查文件集合、尺寸、hash、split、cache 和 state：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.offline_quick_finetune `
  --session-dir payload\quick210 `
  --device cpu `
  --validate-only
```

校验通过后微调 10 epoch：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.offline_quick_finetune `
  --session-dir payload\quick210 `
  --device auto `
  --epochs 10
```

可训练范围严格为：Language Block 2、CCD `224→192` adapter、融合 gate、融合后
LayerNorm 和 64 维 retrieval head，共 255,811 参数。它执行与服务器第四层相同的
CCD 合同和下游电子结构，但不更新前三层、Qwen、phase 或 router。checkpoint 仅按
train loss 选择，固定 gallery/test 不参与选轮。

输出位于：

```text
payload/quick210/04_language_global/offline_results/
├── best_offline_tail_state.pt
├── train_log.csv
├── metrics.json                       # train-loss best tail 的汇总结果
├── pre_finetune_metrics.json          # 微调前固定 gallery/test 结果
├── post_finetune_metrics.json         # 恢复 best tail 后的固定结果
├── predictions.csv                    # 每个 test query 相邻两行：pre/post
└── ccd_inventory.json
```

`predictions.csv` 逐 query 保存 manifest 的 `sample_id` 和 `key`、真实/预测类别、
Top-1 正误、真实 prototype 相对最强错误 prototype 的 similarity margin，以及真实
类别 rank。pre/post 评估只在训练前和 train-loss best tail 恢复后执行；固定
gallery/test 从不用于选 epoch。这样结果绘图可以直接生成逐样本正确性变化和 margin
变化，而不从 aggregate 指标反推。

该结果只能表述为“前三层仿真 + 第四层实测 + 第四层下游电子微调”，不能表述为
四层全部实测。

## 5. 服务器负责的工作

完整 Qwen 训练、理论振幅导出、四层逐层采集后的全模型微调仍在服务器完成。实验室
电脑只需回传 `ccd_captured`，不需要回传重建后的 1024×1024 振幅 BMP。

服务器 quick210 导出命令：

```bash
PROJECT=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5
MODULE=experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5
CKPT=$PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt
SESSION=$PROJECT/hardware_sessions/quick_language_global_run1

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m $MODULE.hardware_bridge \
  --config $PROJECT/configs/release/quick_last_stage_10x10.yaml \
  --checkpoint $CKPT \
  --session-dir $SESSION \
  --stage language_global \
  --phase export \
  --upstream-source simulation
```

正式打包命令：

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.lab_package \
  --checkpoint $CKPT \
  --phase-export-dir $PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test/hardware_phase_export \
  --quick-session-dir $SESSION \
  --stage-a-run-dir $PROJECT/runs/caltech101_warmstart5_stage1_optical_calibration \
  --stage-b-run-dir $PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test \
  --fresnel-calibration-dir experiments/hardware_sdk/generators/slm_patterns/generated/fresnel_roi_vertex_array_532nm_17um_8um_v2 \
  --dual-slm-calibration-dir experiments/hardware_sdk/generators/slm_patterns/generated/dual_slm_17um_8um_alignment_normal_polarity/recommended_checker_grating_pair \
  --fixed-simulation-report-dir $PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test/fixed_simulation_report \
  --output $PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test/qwen_caltech101_10cm_warmstart5_quick210_lab_bundle.zip \
  --overwrite
```

不要在正式移交时使用 `--omit-vendor-sdk`；默认只打入运行必需项，不会复制整套厂商
示例。打包完成后保留终端报告中的 ZIP
SHA-256 和同名 `.zip.json` sidecar；解压后的 `bundle_manifest.json` 可逐文件核对。

## 6. 四层逐层实验边界

完整四层顺序固定为：

```text
vision_expert → vision_global → language_expert → language_global
```

每完成一层采集，就在服务器按该层 CCD 微调其下游，再导出下一层输入。这个 quick210
包只预生成“前三层仿真、第四层实测”的最快路径；不能拿第四层 quick payload 代替
四层逐层 session。四张正式 phase BMP 已随包保存，便于核对版本和相位 SLM 加载。

服务器每导出一个正式 stage 后，实验室对该 stage 固定执行：

```powershell
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload amplitude --hardware-profile meadowlark_17um
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --file-manifest "$STAGE\amplitude_to_play\reconstruction_manifest.csv" --validate-only
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --file-manifest "$STAGE\amplitude_to_play\reconstruction_manifest.csv" --limit 3 --clear-output
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --file-manifest "$STAGE\amplitude_to_play\reconstruction_manifest.csv" --clear-output
```

## 7. 结果汇总与论文图

包内 `result_report.py` 只读取真实存在的固定仿真、CCD 和微调指标；缺少硬件数据时
明确标记 unavailable，不会补造数值。解压根目录运行：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.result_report `
  --root . `
  --output-dir result_report `
  --require-arial
```

输出包含机器可读 JSON/CSV、Arial 优先 7 pt 图、SVG/PDF、600 dpi PNG/TIFF、
文件 SHA 和 QA 报告。正式包还可在 `reference/fixed_simulation_report/` 预置真实
81% 固定仿真基线图；后续采集后再次运行即可加入 quick210 或四层实测结果。详细
合同见 `reference/qwen_project_source/RESULT_PLOTTING.md`。
