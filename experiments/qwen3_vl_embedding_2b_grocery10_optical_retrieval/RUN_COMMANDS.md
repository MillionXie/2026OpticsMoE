# Grocery10 训练与逐层实物实验命令

所有服务器命令均从仓库根目录 `2026OpticsMoE/` 执行。当前实物版本固定使用：

```text
model config: grocery10_moe4_latest.yaml
hardware config: grocery10_moe4_latest_hardware.yaml
结构: Vision expert → Vision global → Language expert → Language global
物理 CCD ROI: 956×956 → 2×2 mean binning → 478×478
```

## 1. 从零训练当前 MoE4

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest.yaml \
  --phase all
```

## 2. 创建一次完整硬件 session

只在实验开始时运行一次。开始采集后不要再次运行 `prepare`，否则配置中的清理选项会重建 session。

```bash
export CFG=experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest_hardware.yaml
export SESSION=experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_sessions/moe4_transfer_001
export GPU=2  # 按服务器空闲情况修改

CUDA_VISIBLE_DEVICES=$GPU python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config "$CFG" --phase prepare --artifact-profile full --output-dir "$SESSION"
```

初始输入与四张共享相位 mask：

```text
$SESSION/01_vision_expert/amplitude_to_play/*.bmp
$SESSION/00_masks/01_vision_expert/*.bmp
$SESSION/00_masks/02_vision_global/*.bmp
$SESSION/00_masks/03_language_expert/*.bmp
$SESSION/00_masks/04_language_global/*.bmp
```

相位 BMP 已在导出前上下翻转。实验室电脑从仓库根目录执行同一条命令（新 TUCam CCD）：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\hardware_sdk\configs\acquisition\tucam_windows.json `
  --clear-output
```

将生成的同名无损 PNG 上传到对应的 `$SESSION/<layer>/ccd_captured/`。

## 3. 普通逐层处理（不微调）

### Layer 1：Vision expert

上传 CCD 到：

```text
$SESSION/01_vision_expert/ccd_captured/
```

```bash
CUDA_VISIBLE_DEVICES=$GPU python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config "$CFG" --output-dir "$SESSION" --phase process_vision_expert
```

下一层振幅：`$SESSION/02_vision_global/amplitude_to_play/`

### Layer 2：Vision global

上传 CCD 到 `$SESSION/02_vision_global/ccd_captured/`，然后运行：

```bash
CUDA_VISIBLE_DEVICES=$GPU python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config "$CFG" --output-dir "$SESSION" --phase process_vision_global
```

下一层振幅：`$SESSION/03_language_expert/amplitude_to_play/`

### Layer 3：Language expert

上传 CCD 到 `$SESSION/03_language_expert/ccd_captured/`，然后运行：

```bash
CUDA_VISIBLE_DEVICES=$GPU python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config "$CFG" --output-dir "$SESSION" --phase process_language_expert
```

下一层振幅：`$SESSION/04_language_global/amplitude_to_play/`

### Layer 4：Language global

上传 CCD 到 `$SESSION/04_language_global/ccd_captured/`，然后运行：

```bash
CUDA_VISIBLE_DEVICES=$GPU python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config "$CFG" --output-dir "$SESSION" --phase process_language_global
```

最终结果：

```text
$SESSION/05_retrieval/metrics.json
$SESSION/05_retrieval/retrieval_results.csv
$SESSION/05_retrieval/confusion_matrix.csv
```

## 4. 推荐：每层实测后微调剩余网络

规则是：实测 CCD 所在平面及其上游全部冻结，只训练它后面的光学和电子参数。每次命令会：

1. 读取本层所有同名 CCD 和 SKU 标签；
2. 用 Teacher embedding KD + SKU supervised contrastive loss 微调；
3. 保存本层 `best_train_loss.pt` 与 `last.pt`；
4. 导出所有被更新的下游相位 BMP；
5. 自动生成下一层振幅 BMP；
6. 下一层从本层 best checkpoint 继续，形成明确 checkpoint 链。

### 捕获 Layer 1 后

```bash
CUDA_VISIBLE_DEVICES=$GPU python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune \
  --config "$CFG" --output-dir "$SESSION" --capture-stage vision_expert
```

```bash
export CKPT1=$SESSION/06_hardware_finetune/after_01_vision_expert__ccd_native/checkpoints/best_train_loss.pt
```

加载更新后的 `$SESSION/00_masks/02_vision_global/*.bmp`，播放更新后的 `$SESSION/02_vision_global/amplitude_to_play/*.bmp`。

### 捕获 Layer 2 后

```bash
CUDA_VISIBLE_DEVICES=$GPU python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune \
  --config "$CFG" --output-dir "$SESSION" --checkpoint "$CKPT1" --capture-stage vision_global
export CKPT2=$SESSION/06_hardware_finetune/after_02_vision_global__ccd_native/checkpoints/best_train_loss.pt
```

加载更新后的 `$SESSION/00_masks/03_language_expert/*.bmp`，播放 `$SESSION/03_language_expert/amplitude_to_play/*.bmp`。

### 捕获 Layer 3 后

```bash
CUDA_VISIBLE_DEVICES=$GPU python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune \
  --config "$CFG" --output-dir "$SESSION" --checkpoint "$CKPT2" --capture-stage language_expert
export CKPT3=$SESSION/06_hardware_finetune/after_03_language_expert__ccd_native/checkpoints/best_train_loss.pt
```

加载更新后的 `$SESSION/00_masks/04_language_global/*.bmp`，播放 `$SESSION/04_language_global/amplitude_to_play/*.bmp`。

### 捕获 Layer 4 后

```bash
CUDA_VISIBLE_DEVICES=$GPU python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune \
  --config "$CFG" --output-dir "$SESSION" --checkpoint "$CKPT3" --capture-stage language_global
```

这里没有后续光学层，因此只训练 final detector normalization 和 64D retrieval readout，并直接写最终指标。

### 只捕获 Layer 4 的快捷方式

无需补齐前三层 CCD。把最终 CCD 按 manifest 同名放入：

```text
$SESSION/04_language_global/ccd_captured/
```

直接执行：

```bash
CUDA_VISIBLE_DEVICES=$GPU python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune \
  --config "$CFG" --output-dir "$SESSION" --capture-stage language_global
```

该模式以前三层仿真路径提供样本上下文，但最终 embedding 的 detector 输入严格替换为实测 Layer-4 CCD；只有最终电子 readout 可训练。

如果 20 个 epoch 已经训练完成、只在最后生成指标时中断，不要重跑训练。直接加载本阶段已经保存的 best checkpoint 收尾：

```bash
CUDA_VISIBLE_DEVICES=$GPU python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune \
  --config "$CFG" --output-dir "$SESSION" --checkpoint "$CKPT3" \
  --capture-stage language_global --finalize-only
```

## 5. 输出目录约定

```text
$SESSION/
├── 00_manifest/                         # 固定样本与播放顺序
├── 00_masks/                            # 当前 checkpoint 对应、实验应加载的相位 BMP
├── 01_vision_expert/{amplitude_to_play,ccd_captured,registered_ccd}/
├── 02_vision_global/{amplitude_to_play,ccd_captured,registered_ccd}/
├── 03_language_expert/{amplitude_to_play,ccd_captured,registered_ccd}/
├── 04_language_global/{amplitude_to_play,ccd_captured,registered_ccd}/
├── 05_retrieval/                        # 最终检索指标
└── 06_hardware_finetune/
    └── after_XX_<stage>__ccd_<transform>/
        ├── checkpoints/{best_train_loss.pt,last.pt}
        ├── metrics/{history.csv,summary.json}
        ├── exported_downstream_masks/   # 本次适配后的 mask 快照
        └── next_amplitude_bmp/          # 本次适配后的下一层振幅快照
```

始终以 `00_masks/` 和下一层标准 `amplitude_to_play/` 作为下一次实验输入；`06_hardware_finetune/` 是不可覆盖的追溯副本。

## 6. ROI 与最近邻注册

推荐先用棋盘格确定 `capture.roi_xywh=[x,y,width,height]`。裁剪后若 CCD 尺寸仍不是 956×956，当前配置会执行：

```text
captured CCD → ROI crop → optional vertical flip → optional horizontal flip → nearest resize to 956×956 → 2×2 mean binning → 478×478
```

每张图的原始尺寸和目标尺寸记录在对应层 `registered_ccd/<sample>.json`。最近邻只解决像素数差异，不能纠正旋转、透视和 ROI 位置错误。

`capture.flip_vertical` 和 `capture.flip_horizontal` 只作用于实测 CCD，不作用于
仿真张量。ROI 的 `[x,y,width,height]` 始终按相机原始画面填写；程序裁完 ROI
后依次执行垂直、水平变换。当前重建光路的实测配置两者均为 `true`。

微调目录会按坐标变换自动区分，避免混用 checkpoint：

```text
__ccd_native   # 不翻转
__ccd_vflip    # 仅上下翻转
__ccd_hflip    # 仅左右翻转
__ccd_vhflip   # 上下和左右都翻转（当前重建光路）
```

### 最终 CCD 的全 SKU 原型检索强化

重新搭建并确认 CCD 方向后，使用下列配置继续训练最终电子读出：

```bash
CUDA_VISIBLE_DEVICES=3 python -m \
  experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_high_accuracy_hardware.yaml \
  --capture-stage language_global
```

这一版不会覆盖原来的 36% checkpoint，输出到：

```text
hardware_sessions/moe4_transfer_001/06_hardware_finetune/
after_04_language_global__ccd_vhflip__prototype_allsku_v1/
```

与旧微调每轮随机使用 `20/110` 张图不同，新版每个 epoch 覆盖全部 100 张
实测 query；每个 batch 都包含 10 个 SKU 的实测 gallery prototype，并直接优化
与部署一致的 cosine gallery retrieval。日志新增 `prototype_loss`、`train_top1`、
`train_top3` 与 `train_mrr`，checkpoint 首先按实测训练 Top-1、同 Top-1 时按总损失选择。

注意：这 100 张 query 同时参与硬件域适配，因此这里的 `train_top1` 是校准集
重代入精度，不是独立测试精度。要报告无泄漏泛化结果，应重新采集未参与微调的
商品视角作为独立 query。

## 7. 测试

```bash
pytest experiments/hardware_sdk/tests \
  experiments/hardware_sdk/generators/slm_patterns/tests \
  experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/tests -q
```
## 推荐正式方案：Grocery31 预训练 → 目标 10 SKU 微调

第一阶段使用相同 2×2 / MoE4 实物结构学习 31 种包装商品，第二阶段切回目标 10 SKU 并降低学习率、开启强增强和 EMA。两个阶段不共享测试图像参与反向传播。

```bash
EXP=experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval
PRE_CFG=$EXP/configs/optimization/grocery31_moe4_pretrain.yaml
FT_CFG=$EXP/configs/optimization/grocery10_moe4_from31_strong_ema.yaml
PRE_OUT=$EXP/runs/qwen3_vl_embedding_2b_grocery31_moe4_pretrain

CUDA_VISIBLE_DEVICES=3 python -m \
  experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval \
  --config "$PRE_CFG" --phase all

CUDA_VISIBLE_DEVICES=3 python -m \
  experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval \
  --config "$FT_CFG" --phase all \
  --resume-checkpoint "$PRE_OUT/ema_best_train_loss_checkpoint.pt"
```

最终 10-SKU 结果位于：

```text
$EXP/runs/qwen3_vl_embedding_2b_grocery10_moe4_from31_strong_ema/
```

当前服务器参考结果还额外保留了固定第 40 轮的 EMA：

```text
$EXP/runs/qwen3_vl_embedding_2b_grocery10_moe4_from31_epoch40_replay/ema_last_checkpoint.pt
```

为这组新权重建立全新的硬件 session（旧 CCD 是旧 mask 的输出，不能混用）：

```bash
CUDA_VISIBLE_DEVICES=3 python -m \
  experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config "$EXP/configs/grocery10_moe4_from31_hardware.yaml" \
  --phase prepare --artifact-profile full
```
