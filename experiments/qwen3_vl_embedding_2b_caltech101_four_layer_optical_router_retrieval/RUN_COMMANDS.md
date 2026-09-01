# Router 实验命令

所有命令都从仓库根目录执行。本文是命令记录，不是 `.sh` 文件。

工程模块：

```text
experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval
```

## 0. 运行前确认

```text
cd /DATA/DATA1/guest3/2026OpticsMoE
conda activate xml
```

确认 GPU 0、1、3、4、5 当前空闲：

```text
nvidia-smi
```

本实验固定使用下面五份 release 配置：

```text
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/electronic_legacy_topk2_anchor.yaml
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/electronic_power_topk1.yaml
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/electronic_power_topk2.yaml
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/electronic_power_topk4.yaml
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/optical_power_topk2.yaml
```

含义：

| 配置 | Router | k | 幅度/功率口径 | 用途 |
|---|---|---:|---|---|
| `electronic_legacy_topk2_anchor` | 电子 | 2 | 原始 amplitude weight | 复现旧模型语义 |
| `electronic_power_topk1` | 电子 | 1 | power_l2 + STE | 正式 k 消融 |
| `electronic_power_topk2` | 电子 | 2 | power_l2 + STE | 正式 k 消融及光/电基线 |
| `electronic_power_topk4` | 电子 | 4 | power_l2 + STE | 正式 k 消融 |
| `optical_power_topk2` | 光学 | 2 | power_l2 + STE | 与电子 Top-2 公平比较 |

不要把 legacy anchor 与三组 power_l2 当作同一口径的 k 排名。

## 1. 静态检查和测试

```text
python -m pytest experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/tests -q
```

如果本工程测试尚未通过，不要启动正式训练。

## 2. 数据准备

只需执行一次。数据缓存仍使用工程配置指向的正式 Caltech101 cache，不应重复下载或
在每个 run 目录复制数据。

```text
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/electronic_legacy_topk2_anchor.yaml \
  --phase prepare_data
```

`prepare_data` 完成后先检查输出的类别、train/gallery/test 样本数和 cache 路径，再启动
四个并行训练任务。

## 3. 先物化不训练的 Legacy Top-2 锚点

该锚点必须保持 optimizer step 为 0，不能再训练 12 epoch 后仍称为“原 81% 模型”。

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/electronic_legacy_topk2_anchor.yaml \
  --phase materialize_initialization
```

输出 checkpoint：

```text
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/runs/electronic_legacy_topk2_anchor/converted_warmstart5_initialization_checkpoint.pt
```

## 4. 正式电子 Router：Top-1/2/4 并行训练

下面按用户指定的空闲 GPU 0、1、3、4 分配。每条命令应在独立终端中运行。

### GPU 1：power_l2 + STE，Top-1

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/electronic_power_topk1.yaml \
  --phase train
```

### GPU 3：power_l2 + STE，Top-2

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/electronic_power_topk2.yaml \
  --phase train
```

### GPU 4：power_l2 + STE，Top-4

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/electronic_power_topk4.yaml \
  --phase train
```

训练日志必须至少确认：

- 实际 `router_implementation` 为 electronic；
- 实际 k 分别是 2、1、2、4；
- power 三组启用了 `power_l2` 和 STE，legacy 组没有被误改；
- Qwen 仍冻结；
- 每组 output directory 不相同；
- Router entropy、四专家 load、margin 和 phase 梯度均为有限值。

## 5. 电子 Router：显式 sealed-test evaluate

训练阶段不逐 epoch 查看 test。四组完成后才分别执行：

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/electronic_legacy_topk2_anchor.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/runs/electronic_legacy_topk2_anchor/converted_warmstart5_initialization_checkpoint.pt
```

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/electronic_power_topk1.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/runs/electronic_power_topk1/ema_best_train_loss_checkpoint.pt
```

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/electronic_power_topk2.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/runs/electronic_power_topk2/ema_best_train_loss_checkpoint.pt
```

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/electronic_power_topk4.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/runs/electronic_power_topk4/ema_best_train_loss_checkpoint.pt
```

比较 k=1/2/4 时只使用三份 `electronic_power_topk*` 结果。Legacy anchor 单独报告为
“原 warmstart5 口径复现”。

## 6. 光学 Router Top-2 训练

它可以与三组电子消融并行；下例使用当前空闲的 GPU 5：

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/optical_power_topk2.yaml \
  --phase train
```

光 Router 必须从与电子 power Top-2 相同的 warmstart5 起点开始，并保留完全相同的
power_l2、STE、数据划分、训练步数和下游损失。训练开始时检查日志：

- `router_implementation=phase_only_detector_energy_topk`；
- Vision/Language 各有一张 224×224 Router phase；
- CCD Router 输出为 478×478；
- detector intervals 为 `[162,221)` 和 `[257,316)`；
- 四区 `capture_fraction`、Router entropy、load、margin 有记录；
- 启动前单元测试证明 Router phase 梯度非零且有限；训练后 snapshot/preview 证明其相对
  确定性初值发生了变化；
- feature optics 仍为 Vision Expert/Global、Language Expert/Global 四层。

训练结束后显式执行一次测试：

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/optical_power_topk2.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/runs/optical_power_topk2/ema_best_train_loss_checkpoint.pt
```

公平的主对照是：

```text
electronic_power_topk2  vs  optical_power_topk2
```

不要用 `electronic_legacy_topk2_anchor` 作为光 Router 的唯一主对照，因为两者的功率
归一化和梯度口径不同。

### 6.1 导出两张 Router 相位 BMP

该命令不加载 Qwen，只从光 Router checkpoint 严格读取两张相位，因此导出较快：

```text
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.export_router_masks \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/optical_power_topk2.yaml \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/runs/optical_power_topk2/ema_best_train_loss_checkpoint.pt \
  --output-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/hardware/router_masks
```

可播放文件在 `hardware/router_masks/phase_to_play/`，并沿用 17 μm 逻辑像素到
8 μm 相位 SLM 的最近物理坐标映射、中心 `(980,590)` 和原 vertical flip。

### 6.2 从已校正方向的 Router CCD 读取四个权重

输入必须已经由现有 homography 处理成 canonical 478×478；此工具不会再次翻转：

```text
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.score_router_ccd \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/optical_power_topk2.yaml \
  --input-dir router_ccd_canonical \
  --output-dir router_scores
```

输出 `routing.csv`、`routing_report.json` 和带四个计分框的首帧预览。程序不扣背景，
只将四个区域能量做相同的样本内标准化后执行 softmax/top-k。

### 6.3 把实测权重重新装载成原 2×2 Expert 振幅

`central_router_inputs` 中必须是和 Router 曝光逐文件对应的 224×224 灰度振幅；输出
恢复为原 MoE4 的 478×478 布局，并重建为 Meadowlark 1024×1024 BMP：

```text
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.build_routed_amplitudes \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/optical_power_topk2.yaml \
  --input-dir central_router_inputs \
  --routing-csv router_scores/routing.csv \
  --output-dir routed_expert_amplitudes
```

最终播放 `routed_expert_amplitudes/amplitude_to_play/*.bmp`。manifest 会逐样本记录四个
振幅权重和 `sum(weight²)`，正式 power_l2 运行应接近 1。

### 6.4 审计 Router 相位是否真正移动

```text
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.audit_router_checkpoint \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/optical_power_topk2.yaml \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/runs/optical_power_topk2/ema_best_train_loss_checkpoint.pt \
  --output experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/runs/optical_power_topk2/router_phase_audit.json
```

它从配置重新构建确定性的四束初值，再以相位圆周差计算 Vision/Language 两张 mask 的
RMS 位移、最大位移及移动超过 0.01 rad 的像素比例，并记录 checkpoint SHA。

## 7. 结果检查顺序

先检查训练协议，再看准确率：

1. config、checkpoint、初始化 SHA 和 output directory 是否唯一；
2. train 阶段是否没有使用 sealed test 选择 checkpoint；
3. power 三组是否满足每个样本 `Σ amplitude_scale²≈1`；
4. 每个样本激活专家数是否严格等于 k；
5. k=1 的 Router 是否存在非零 STE 梯度；
6. 四专家是否都曾被激活，是否出现单专家塌缩；
7. 光 Router 四区捕获率和 top-k margin 是否足够；
8. 最后比较 Top-1、Top-3、MRR 和扰动稳定性。

建议最终表格至少包含：

| run | router | k | weighting | Top-1 | Top-3 | MRR | entropy | min load | route stability | capture fraction |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|

## 8. 六次硬件采集的操作顺序

光 Router 仿真通过后，硬件执行顺序必须固定为：

```text
01_vision_router
02_vision_expert
03_vision_global
04_language_router
05_language_expert
06_language_global
```

其中：

- `vision_router` 和 `language_router` 播放居中的 224×224 输入，不是 2×2 mosaic；
- Router CCD 经现有 homography/方向合同变成 canonical 478×478；
- 读取四区后立即生成 routing weights 和下一阶段的 2×2 amplitude；
- Vision Global 复用 Vision Router 的 route；
- Language Global 复用 Language Router 的 route；
- 四个 feature CCD 阶段仍沿用原 readout 和逐层微调逻辑。

当前已实现 Router phase 导出和 canonical Router CCD 四区评分；完整六阶段自动 bridge
仍需在仿真结果确认后，把“中心振幅导出→实测路由权重→2×2条件振幅生成”接入原四阶段
逐层微调。不要直接套用 warmstart5 的四阶段 `hardware_bridge`：它不知道两次 Router
CCD，也无法根据实测四区结果生成条件 amplitude。

## 9. 常见错误

- 不要将 detector 区间写成包含右端点；`[162,221)` 的最后一个像素是 220。
- 不要把四个 detector 小区误认为新的 CCD crop；正式 ROI 仍是 478×478。
- 不要对已经 canonical 的 CCD 再次翻转。
- 不要用 k=1 的普通 hard top-k 训练并声称 Router 得到了任务梯度。
- 不要将 `Σw=1` 误认为总光功率相等；振幅播放时功率与 `Σa²` 成正比。
- 不要逐 epoch 选择最高 test；本工程使用显式 `--phase evaluate` 的 sealed-test 协议。
