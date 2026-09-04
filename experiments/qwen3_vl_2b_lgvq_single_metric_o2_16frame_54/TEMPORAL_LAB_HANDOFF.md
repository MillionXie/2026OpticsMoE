# LGVQ Temporal-16 实验交接（先读本文）

这份交接只对应 **Temporal quality**，不是 Spatial，也不是旧四帧模型。正式仿真权重为
`deployment/best_observed_test_checkpoint.pt`，正式相位文件为
`deployment/hardware_masks/phase_slm_1920x1200/*.bmp`。不要混用旧工程的 checkpoint 或
phase BMP。

## 1. 已锁定的模型合同

- 输入：每个视频 16 帧；每帧由冻结的 Qwen3-VL patch embedding + position embedding
  得到 49 个 1024 维视觉 token。没有 Vision Transformer、attention 或 Qwen merger。
- 文本：固定 Temporal prompt 经 Qwen tokenizer 和冻结的 2048 维 text embedding；文本
  参与视觉调制并在视觉阶段后与 16 个 frame token 拼接。
- 光路：532 nm、17 um 逻辑像素、10 cm、518 仿真画布、478 有效区域。
- 前三次传播把 16 帧同时排成 4x4；每帧 lane 为 114x114，每个 MoE 专家为 54x54，
  每帧 2x2 共 4 个专家。后面三次传播是包含 16 帧和 prompt 的序列光场，专家为
  109x109、2x2 共 4 个。
- router：光学能量 router，Top-2。完整推理有四个光电融合层，但需要六次物理传播：
  `vision_router -> vision_expert -> vision_global -> language_router ->
  language_expert -> language_global`。
- 融合：每个融合点先分别做 RMS 尺度对齐，再用
  `(1-alpha)*electronic + alpha*optical`；四个最终 alpha 约为
  0.5635、0.5634、0.5703、0.5701。
- 仿真含 20% nominal coherent unmodulated optical-power 分量；训练时在 20% 到 35%
  随机。它是场叠加，不等于 CCD 每像素固定加 20% 灰度。

## 2. 已验证结果

最佳 checkpoint 来自 epoch 35。没有 validation；每 5 epoch 在 test 上评估并按最高
Temporal SRCC 选权重，这是本实验约定的选择口径。

| 同一 checkpoint | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| 正常光电 | 0.840591 | 0.633062 | 0.859658 | 7.2170 | 5.6309 |
| 屏蔽全部光路 | 0.689916 | 0.484531 | 0.610969 | 14.9725 | 11.7094 |
| 光开减光关 | +0.150675 | +0.148532 | +0.248689 | -7.7555 | -6.0786 |

“屏蔽光路”不是另训一个电模型，而是同一权重在推理时绕过光支路，因此用于测量该
checkpoint 对光的实际依赖。

## 3. 解压后先检查

在工程根目录的 PowerShell 中：

```powershell
conda activate xml
python VERIFY_BUNDLE.py
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

正式文件：

```text
deployment/best_observed_test_checkpoint.pt
deployment/hardware_masks/phase_slm_1920x1200/
deployment/hardware_masks/amplitude_layout_1024x1024/
deployment/evidence/
deployment/figures/
```

`phase_slm_1920x1200` 是可播放的 uint8 BMP，中心为 (980,590)，已经做了纵向翻转，
播放软件不能再次翻转。`amplitude_layout_1024x1024` 是布局检查模板，不是某个样本的
振幅数据；正式样本振幅必须由前一电子/CCD 状态逐 pass 生成。

## 4. 硬件标定

硬件代码与 Windows SDK 位于 `experiments/hardware_sdk`，唯一实验配置入口为
`experiments/lab_lgvq/LAB_CONFIG.yaml`。顺序：

1. 双 SLM：使用 `experiments/lab_lgvq/calib/dual_slm_k1`，先从 k=1 对齐。
2. 10 cm 与 CCD ROI：振幅加载 `A_WHITE.bmp`，相位依次加载
   `P1_POINT.bmp`、`P4_POINT.bmp`、`P9_POINT.bmp`；P4 用于测四个逻辑角点。
3. 在 `LAB_CONFIG.yaml` 填 LUT、曝光、相位中心/翻转、四个逻辑角点。
4. 运行：

```powershell
python -m experiments.lab_lgvq.prepare_lab
python -m experiments.hardware_sdk.workflows.roi_calibration exposure `
  --config experiments\lab_lgvq\generated\formal_hardware.yaml
```

正式保存的 CCD 是硬件 ROI 原始强度经单应性矫正到 478x478；不做背景扣除、逐帧
min-max 或额外 log。网络内部对仿真 CCD 与实测 CCD 使用同一套非负截断、单帧均值
归一化、相对强度截断和 `log1p`。

## 5. 六次实验的不可更改顺序

每个 pass 都执行：生成该 pass 的样本相关 1024x1024 振幅 BMP；手动加载本包同名
1920x1200 相位 BMP；用 `acquire_folder` 采集为 canonical 478x478 CCD；校验文件数、
命名和饱和率；再推进到下一 pass。

| # | pass | 相位 BMP | 输入布局 |
|---:|---|---|---|
| 1 | vision_router | `vision_router.bmp` | 16 个中心输入，4x4 并行 |
| 2 | vision_expert | `vision_expert.bmp` | 16x4 个 54x54 专家位，按实测 Top-2 加权 |
| 3 | vision_global | `vision_global.bmp` | 16 个中心输入，4x4 并行 |
| 4 | language_router | `language_router.bmp` | 中心 109x109 序列场 |
| 5 | language_expert | `language_expert.bmp` | 2x2 个 109x109 专家位，按实测 Top-2 加权 |
| 6 | language_global | `language_global.bmp` | 中心 109x109 序列场 |

振幅 SLM 合同为 1024x1024、17 um、`255=亮/透光`。相位 SLM 合同为
1920x1200、8 um、532 nm；相位逻辑 478x478 以物理坐标最近邻映射成 1016x1016，
中心放在 (980,590)。CCD 必须先按本实验台四点合同矫正到 478x478，再交给读出网络。

## 6. 代码边界与微调

本包包含两套清楚隔离的代码：

- `experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54`：本次 Temporal-16 的
  **精确模型、训练、评估、mask 导出代码**，与正式 checkpoint 完全匹配。
- `experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa`：此前已在实验室
  验证过的 **六 pass 导出/采集/逐层微调参考实现**。它的旧四帧 checkpoint 不在本包，
  也绝不能用它直接加载本次 Temporal-16 checkpoint；它用于让接手 AI 按
  `AI_TEMPORAL16_PORTING_CONTRACT.md` 移植设备循环、session 清单、CCD 校验和 best-test
  选模逻辑。

本次 checkpoint 只有约 80 MB，但 2250 train + 558 test 的 16 帧冻结 Qwen 前端缓存
约 4.3 GB。为了让包可快速交付，默认未复制这个缓存；`deployment/cache_metadata` 保留
其身份与形状合同。需要本地全量微调时，从训练服务器另取
`qwen3vl_front_16f_49x1024_quality14.pt`，并保持 manifest/sample 顺序完全一致。

微调选择约定：每 5 epoch 计算一次 test 指标，保存最高 test SRCC，而不是最低训练
loss。硬件微调必须逐段进行；完成前缀 pass 的实测替换后，只微调其下游电子读出，
不能反向更新已经加载到物理 SLM 的相位。

## 7. 给接手 AI 的第一条指令

请先阅读本文、`AI_TEMPORAL16_PORTING_CONTRACT.md`、正式 config、
`hardware_mask_export_report.json` 和旧硬件桥的 `hardware_contract.py`，然后建立一个
Temporal-16 专用硬件桥。不得改变 16 帧 4x4/54 几何、Top-2 光路由、六 pass 顺序、
CCD 478x478 合同、RMS 融合或 20% DC 合同；不得把布局模板当成样本振幅。
