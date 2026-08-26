# 第四层 quick210 离线微调

本文只描述“前三层仿真、第四层 `language_global` 实测”的快速路径。离线程序不加载 Qwen、Transformers、Caltech101 原图、相位传播器或 router；实验室电脑只需 PyTorch、NumPy 和 Pillow。

## 1. 服务器导出

```bash
PROJECT=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust
CKPT=$PROJECT/runs/caltech101_four_layer_moe4_joint_17um_10cm_robust/ema_best_train_loss_checkpoint.pt
SESSION=$PROJECT/hardware_sessions/quick_language_global_10cm_robust_run1

CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge \
  --config $PROJECT/configs/release/caltech101_four_layer_optical_quick_last_stage_10x10.yaml \
  --checkpoint $CKPT \
  --session-dir $SESSION \
  --stage language_global \
  --phase export \
  --upstream-source simulation
```

除理论振幅和相位外，这一步新增以下小文件：

```text
quick_language_global_10cm_robust_run1/
├── manifest.csv
└── 04_language_global/
    ├── compact_amplitude/
    ├── phase_to_play/
    ├── ccd_captured/
    └── offline_downstream/
        ├── cache.pt                 # 一个 packed float32 Block-2 输入缓存
        ├── downstream_state.pt      # 255,811 参数的轻量 tail
        └── contract.json            # hash、split、CCD 和模型合同
```

`cache.pt` 保存的是第四层之前的冻结边界，即 Language Block 1 融合结果，也就是 Block 2 输入。它没有混入仿真第四层 CCD。quick210 必须严格为10类，每类10张 train、1张 gallery、10张 test，总计210张。

## 2. 实验室采集

实验室播放、采集命令不变：

```powershell
$STAGE = "quick_language_global_10cm_robust_run1\04_language_global"

python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml `
  --stage-dir $STAGE `
  --clear-output
```

最终必须得到恰好210张：

```text
04_language_global/ccd_captured/<manifest-key>.png
```

每张必须是原始运输方向的 `478×478`、8-bit、PIL mode `L` 灰度 PNG。离线程序会按训练合同执行上下和左右翻转。程序禁止背景扣除、resize、956降采样和自动灰度转换；文件 stem 缺失、多余或重复都会直接报错。

## 3. 安装离线微调环境

CPU 弱机也能运行：

```powershell
python -m venv .venv_offline
.\.venv_offline\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pillow
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

如已有适配本机 CUDA 的 PyTorch，不要再安装 CPU wheel。该路径无需安装 `transformers`。

先只校验全部文件、hash、split 和模型 state：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.offline_quick_finetune `
  --session-dir quick_language_global_10cm_robust_run1 `
  --device cpu `
  --validate-only
```

校验通过后执行10轮微调：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.offline_quick_finetune `
  --session-dir quick_language_global_10cm_robust_run1 `
  --device auto `
  --epochs 10
```

也可以显式指定 `--device cpu`、`--device cuda` 或 `--device cuda:0`。如果 CCD 不在 stage 默认目录，可增加 `--ccd-dir D:\path\to\ccd_captured`。

## 4. 实际训练范围

离线 tail 与服务器 `language_global` 微调范围相同：

```text
cached Block-2 input [L,192]
  -> causal Depthwise Conv1D + residual MLP
measured CCD [478,478]
  -> clamp>=0
  -> single-frame mean normalization
  -> relative clip 12
  -> log1p
  -> AdaptiveAvgPool 478->224
  -> per-token LayerNorm + ReLU
  -> Linear 224->192
electronic + bounded optical gate * optical delta
  -> LayerNorm
  -> token mean/max [384]
  -> LayerNorm + Linear 384->64 + L2 normalization
```

可训练参数精确为255,811：电子 Block 2为186,818，CCD输出适配器43,200，融合后的 LayerNorm 384，光融合门控1，64维检索头25,408。门控仍使用 `0.10 + 0.90×sigmoid(raw)`，因此光参与比例不会低于10%。

训练采用10类×3张的 PK batch，每轮3个 optimizer step，loss 为 supervised contrastive（温度0.07）加 episodic prototype（温度0.15）。仅按 train loss 选择最佳轮次，最后再使用固定10张 gallery 和100张 test 计算指标；test 不参与 checkpoint 选择。电子 Block 2保持 eval mode，因此 dropout 关闭，但参数仍有梯度，这与原服务器硬件微调路径一致。

## 5. 输出与移交

默认输出：

```text
04_language_global/offline_results/
├── best_offline_tail_state.pt
├── train_log.csv
├── metrics.json
└── ccd_inventory.json
```

`ccd_inventory.json` 记录210张原始 PNG 的逐文件 SHA-256；`metrics.json` 同时记录整个 CCD 集合指纹、源 checkpoint hash、contract hash、最佳轮次、gate 和固定测试指标。`best_offline_tail_state.pt` 仅包含轻量 tail tensor，不包含 Qwen，也不会覆盖正式服务器 checkpoint。

这个结果只能表述为“前三层仿真 + 第四层实测 + 第四层下游电子微调”，不能表述为四层全部实测或正式全量 Caltech101 指标。
