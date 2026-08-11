# Grocery10 MoE4 发布版命令

以下命令均从仓库根目录 `2026OpticsMoE` 运行。物理相机和 SLM 的播放采集命令见
`experiments/hardware_sdk/COMMANDS.md`；本文件只负责模型训练、BMP 生成、CCD 电处理和微调。

## 1. 一键复现训练

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.reproduce_release
```

该命令依次执行：

1. Grocery31 预训练 26 epoch；
2. 缓存 Grocery10 Teacher embeddings；
3. 从 epoch-26 EMA 继续训练 14 epoch；
4. 对绝对 epoch-40 EMA 做完整评测；
5. 保存检索可视化。

已有完整 epoch-26/40 checkpoint 时会跳过训练但重新评测。若发现不完整目录则直接报错，
不会静默覆盖。先查看命令而不执行：

```bash
python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.reproduce_release --dry-run
```

## 2. 单独评测现有最佳权重

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/release/stage2_grocery10_finetune.yaml --phase evaluate --checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/release_moe4_grocery10_epoch40/ema_last_checkpoint.pt
```

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/release/stage2_grocery10_finetune.yaml --phase visualize
```

## 3. 生成完整四平面硬件 session

生成 gallery、全部 train 和全部 test 的固定 manifest、逐层 amplitude BMP、四张 phase BMP
以及理论 CCD：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/release/hardware_moe4.yaml --phase prepare --artifact-profile full
```

默认输出：

```text
experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_sessions/release_moe4_epoch40
```

只导出 gallery + test、且完全不做硬件微调时，可以覆盖选择模式和输出目录：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/release/hardware_moe4.yaml --phase prepare --artifact-profile full --selection-mode test_only --output-dir experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_sessions/release_moe4_epoch40_test_only
```

## 4. 四层纯推理

每次用独立 `hardware_sdk` 播放当前层 `amplitude_to_play/`，把同名 CCD 文件上传至该层
`ccd_captured/`，然后执行：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/release/hardware_moe4.yaml --phase process_vision_expert
```

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/release/hardware_moe4.yaml --phase process_vision_global
```

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/release/hardware_moe4.yaml --phase process_language_expert
```

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/release/hardware_moe4.yaml --phase process_language_global
```

最终结果：

```text
hardware_sessions/release_moe4_epoch40/05_retrieval/metrics.json
hardware_sessions/release_moe4_epoch40/05_retrieval/retrieval_results.csv
hardware_sessions/release_moe4_epoch40/05_retrieval/confusion_matrix.csv
```

## 5. 每层真实 CCD 后微调下游

第一层：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/release/hardware_moe4.yaml --capture-stage vision_expert
```

第二、三、四层依次把上一阶段输出的 `checkpoints/best_train_loss.pt` 作为 `--checkpoint`：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/release/hardware_moe4.yaml --checkpoint PREVIOUS_BEST_TRAIN_LOSS_PT --capture-stage vision_global
```

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/release/hardware_moe4.yaml --checkpoint PREVIOUS_BEST_TRAIN_LOSS_PT --capture-stage language_expert
```

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/release/hardware_moe4.yaml --checkpoint PREVIOUS_BEST_TRAIN_LOSS_PT --capture-stage language_global
```

默认只用 gallery + train 实测图适配，test 保持独立。只有明确进行 transductive 校准时，才把
`adaptation.include_test_split` 改为 `true`。

## 6. 关键硬件参数在哪里改

只修改 `configs/release/hardware_moe4.yaml`：

- `slm.amplitude_encoding`：振幅 BMP 百分位和 gamma；
- `slm.phase_transform.flip_vertical`：相位 BMP 上下翻转；
- `capture.flip_vertical/flip_horizontal`：真实 CCD 坐标方向；
- `capture.registration_mode`：CCD 到 956×956 的注册方式；
- `capture.binning_factor`：956→478 的 2×2 平均合并；
- `adaptation.*`：硬件域微调参数。

曝光、相机 ROI、SLM SDK 路径和播放延迟只在 `experiments/hardware_sdk/configs/` 修改，
不再复制到本模型配置中。
