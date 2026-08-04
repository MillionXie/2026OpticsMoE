# Grocery10 训练与逐层实物实验

服务器命令均从仓库根目录 `2026OpticsMoE/` 执行。

## 保留的训练版本

| 配置 | 结构 | 用途 |
|---|---|---|
| `grocery10_moe16_best.yaml` | 4×4、16 experts、Top-4 | 历史最佳版本 |
| `grocery10_moe4_latest.yaml` | 2×2、4 experts、Top-2、2×2 CCD integration | 当前实物鲁棒版本 |

MoE4 从零训练：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest.yaml \
  --phase all
```

## 一、服务器建立完整实物 session

逐层人工传输必须使用 `full`，因为服务器需要保存原始张量和中间电子结果。先约定唯一 session；后面四条命令都必须使用同一个路径：

```bash
SESSION=experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_sessions/moe4_transfer_001

CUDA_VISIBLE_DEVICES=5 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest_hardware.yaml \
  --phase prepare \
  --artifact-profile full \
  --output-dir "$SESSION"
```

`prepare` 会生成四张本轮固定相位 mask：

```text
$SESSION/00_masks/01_vision_expert/*.bmp
$SESSION/00_masks/02_vision_global/*.bmp
$SESSION/00_masks/03_language_expert/*.bmp
$SESSION/00_masks/04_language_global/*.bmp
```

相位 BMP 已按配置在导出前上下翻转。四张 mask 可以一次下载到实验室电脑，逐层人工加载。

第一层振幅位于：

```text
$SESSION/01_vision_expert/amplitude_to_play/*.bmp
```

## 二、实验室电脑的固定操作

每一层都只做以下动作：

1. 清空 `hardware_sdk/workspace/amplitude_to_play/` 和 `ccd_captured/`；
2. 从服务器下载本层 `amplitude_to_play/*.bmp` 到前者；
3. 人工加载本层相位 mask；
4. 运行 `python acquire_folder.py --config configs\acquisition_windows.json --clear-output`；
5. 把 `hardware_sdk/workspace/ccd_captured/*.npy` 上传到服务器本层 `ccd_captured/`。

输入 BMP 与 CCD `.npy` 必须同 stem。禁止 JPEG、重命名、缩放或重复平方光强。实验室电脑的完整命令见 [共享 Hardware SDK](../hardware_sdk/RUN_COMMANDS.md)。

## 三、四层数据交接表

| 层 | 人工加载的相位 mask | 实验室振幅来源 | CCD 上传位置 | 服务器处理 phase | 处理后下一层振幅 |
|---|---|---|---|---|---|
| 1 | `00_masks/01_vision_expert/*.bmp` | `01_vision_expert/amplitude_to_play/` | `01_vision_expert/ccd_captured/` | `process_vision_expert` | `02_vision_global/amplitude_to_play/` |
| 2 | `00_masks/02_vision_global/*.bmp` | `02_vision_global/amplitude_to_play/` | `02_vision_global/ccd_captured/` | `process_vision_global` | `03_language_expert/amplitude_to_play/` |
| 3 | `00_masks/03_language_expert/*.bmp` | `03_language_expert/amplitude_to_play/` | `03_language_expert/ccd_captured/` | `process_language_expert` | `04_language_global/amplitude_to_play/` |
| 4 | `00_masks/04_language_global/*.bmp` | `04_language_global/amplitude_to_play/` | `04_language_global/ccd_captured/` | `process_language_global` | 最终 embedding 与检索指标 |

## 四、每一层的服务器电子处理命令

### Layer 1：Vision expert CCD → Vision global 振幅

先把实验室 `.npy` 放入：

```text
$SESSION/01_vision_expert/ccd_captured/
```

运行：

```bash
CUDA_VISIBLE_DEVICES=5 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest_hardware.yaml \
  --phase process_vision_expert --output-dir "$SESSION"
```

下载下一层：

```text
$SESSION/02_vision_global/amplitude_to_play/*.bmp
```

### Layer 2：Vision global CCD → Language expert 振幅

上传到：

```text
$SESSION/02_vision_global/ccd_captured/
```

运行：

```bash
CUDA_VISIBLE_DEVICES=5 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest_hardware.yaml \
  --phase process_vision_global --output-dir "$SESSION"
```

下载下一层：

```text
$SESSION/03_language_expert/amplitude_to_play/*.bmp
```

### Layer 3：Language expert CCD → Language global 振幅

上传到：

```text
$SESSION/03_language_expert/ccd_captured/
```

运行：

```bash
CUDA_VISIBLE_DEVICES=5 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest_hardware.yaml \
  --phase process_language_expert --output-dir "$SESSION"
```

下载下一层：

```text
$SESSION/04_language_global/amplitude_to_play/*.bmp
```

### Layer 4：Language global CCD → 最终检索结果

上传到：

```text
$SESSION/04_language_global/ccd_captured/
```

运行：

```bash
CUDA_VISIBLE_DEVICES=5 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest_hardware.yaml \
  --phase process_language_global --output-dir "$SESSION"
```

最终结果：

```text
$SESSION/05_retrieval/metrics.json
$SESSION/05_retrieval/retrieval_results.csv
$SESSION/05_retrieval/confusion_matrix.csv
```

每层 `electronic_output/*.json` 还包含实测与仿真的 MSE、MAE、relative L2 和 cosine，用来定位误差最先出现在哪一层。

## 五、ROI 约定

当前 MoE4 仿真 active size 为 `478×478`，物理相机按 `2×2` 像素合并，因此送入服务器电子处理前必须得到精确 `956×956` ROI。两种方式二选一：

- 上传全传感器 `.npy`，在硬件 YAML 的 `capture.roi_xywh` 设置 `[x,y,956,956]`；
- 相机直接输出 `956×956`，硬件 YAML 保持 `capture.roi_xywh: null`。

服务器随后只做严格的 `2×2 mean binning`，不会插值缩放。ROI 棋盘格标定命令见 Hardware SDK 文档。

## 六、仅导出精简 BMP/理论图（不用于逐层回放）

下面命令默认生成精简目录，只用于查看或拿 BMP，不含服务器电子 bridge 所需原始数据：

```bash
CUDA_VISIBLE_DEVICES=5 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest_hardware.yaml \
  --phase prepare
```

输出只有 `phase_bmp/`、`amplitude_bmp/`、`theoretical_ccd/` 和必要 manifest。要做四层实物交接，必须使用前述 `--artifact-profile full --output-dir "$SESSION"`。

## 测试

```bash
pytest experiments/hardware_sdk/tests \
  experiments/hardware_sdk/slm_calibration_bmp_generator/tests \
  experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/tests -q
```
