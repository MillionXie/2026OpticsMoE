# SPAQ Qwen3-VL-2B MoE16-224 单专家层基线

本目录是一份可独立运行、适合组内共享的常规 baseline。它复制自
`qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224`，
但不会修改或依赖该实验的 student checkpoint。

核心变化只有一项：Vision 和 Language 的每个光学 stack 都从多专家阶段缩减为：

```text
1 个 expert phase plane + 1 个 global phase plane
```

因此，每个 stack 只有两个可训练相位平面。16 个专家仍然位于同一个物理
expert plane 上，并不是串联的 16 层。

## 实验任务

数据集为 SPAQ，支持四个相互独立的单属性回归任务：

- `MOS`
- `Brightness`
- `Colorfulness`
- `Contrast`

每次运行只训练一个任务。输入图像保持 RGB，由原始 Qwen processor 和 chat
template 处理；标签统一除以 100 后训练，报告时恢复到 0–100 分制。

Teacher 是冻结的完整电子 `Qwen/Qwen3-VL-2B-Instruct` 加一个
`LayerNorm(2048) -> Linear(2048, 1)` 回归头。Student 保留冻结的 Qwen patch
embedding、token embedding、vision merger、单个 DeepStack 辅助注入和 final RMSNorm，
同时用独立的 Vision Optical MoE 与 Language Optical MoE 替换完整 Transformer
stack。

## 两平面光路

Vision 和 Language 使用相同的物理拓扑，但参数完全独立：

```text
token hidden
-> Linear(D, 224)
-> LayerNorm(224)
-> Softplus
-> zero-pad token rows to [224, 224]
-> electronic Top-4 router
-> weighted copies on the amplitude SLM
-> 16-expert phase plane
-> 10 cm propagation
-> square-law detection
-> per-expert non-affine LayerNorm
-> ReLU
-> reapply the original routing weights; zero unselected experts
-> amplitude reload, co-planar with global phase
-> global phase plane
-> 10 cm propagation
-> CCD crop [986, 986]
-> adaptive-average pool [224, 224]
-> non-affine per-token LayerNorm
-> ReLU
-> Linear(224, D)
-> fixed identity residual
```

几何保持不变：

- propagation canvas：`1026 × 1026`
- active/global/CCD ROI：`986 × 986`
- expert layout：`4 × 4`
- each expert：`224 × 224`
- expert gap：`30 px`
- outer zero padding：每边 `20 px`
- top-k：`4`
- wavelength：`532 nm`
- pixel pitch：`8 µm`
- expert-to-OEO/global plane：`10 cm`
- global phase-to-CCD：`10 cm`

这里的 ideal 4f relay 不单独做一次数值传播；它表示 amplitude SLM 的图案被理想
复制到共面的 phase SLM。第一层传播后的 CCD/OEO 输出重新加载到 global phase
所在平面。

## Qwen DeepStack 的处理

Qwen3-VL 原生有三个 vision DeepStack provider，但单专家层只产生一个
pre-global optical hidden。Student 因此只保留第一个 native provider：
pre-global hidden 经冻结的 `deepstack_merger_list[0]` 后，在 Language slot 0
之后注入一次；另外两个 auxiliary provider 不生成、不注入，也不会用同一 hidden
重复填充。Global phase 后的输出仍作为 main/final vision embedding。

Language slot 0 为 bypass，随后 Qwen 加入唯一 auxiliary visual feature。唯一
Language optical stage 放在 decoder layer index 1，因此它同时处理 main visual
embedding、一个 auxiliary visual embedding 与文本。其余 decoder layer bypass，
最后仍经过冻结的 Qwen RMSNorm。

详见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 固定的 baseline 开关

正式配置明确且由 `Settings.validate()` 保护：

- native attention prelude：关闭
- SAM：关闭
- electronic/phase weight decay：均为 `0`
- phase dropout：关闭
- ranking loss：关闭
- Norm-in-Norm loss：关闭
- 历史 student checkpoint 初始化：关闭
- Transformer-style residual：保留，固定为 `Y = X + OpticalMoE(X)`
- formal student batch size：`8`
- formal inference batch size：`8`

Smoke 配置为了显存和执行速度使用 batch size 1；它不改变模型结构、loss 或光路。

## Loss

正式配置使用：

```text
L =
    1.0 * normalized vision hidden MSE（1个auxiliary target + final target）
  + 1.0 * normalized answer hidden MSE
  + 0.5 * SmoothL1(student prediction, teacher prediction)
  + 1.0 * SmoothL1(student prediction, SPAQ label)
  + 0.03 * router balance loss
```

`router_importance_weight=0`。Teacher 与 student head 结构相同，但 student head
从随机初始化开始，不继承 teacher head。

## 参数量

主配置（Vision + Language 均为 optical）：

| 部分 | Vision | Language |
|---|---:|---:|
| 16 个 expert masks（1 plane） | 802,816 | 802,816 |
| global phase | 972,196 | 972,196 |
| input/norm/output adapters | 460,448 | 920,224 |
| electronic router | 3,152 | 3,152 |
| stack total | 2,238,612 | 2,698,388 |

回归 head 为 6,145 参数，student 总可训练参数为 **4,943,145**。冻结的 Qwen
参数不计入这一数字。运行时的真实统计会写入 `model.json`。

## 配置

正式配置：

- `configs/spaq_mos.json`
- `configs/spaq_brightness.json`
- `configs/spaq_colorfulness.json`
- `configs/spaq_contrast.json`

Smoke 配置提供对应的 `*_smoke.json`。另有
`spaq_mos_vision_electronic_language.json`，仅用于定位 Language optical 误差，
不代表主 baseline。

## 快速开始

在仓库根目录执行：

```bash
python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/configs/spaq_mos_smoke.json --phase all
```

正式 MOS：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/configs/spaq_mos.json --phase all
```

所有命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)，配置字段见
[CONFIGURATION.md](CONFIGURATION.md)。

## 输出与选择规则

输出包含 resolved config、dataset/model/environment report、teacher/processor
cache、teacher/student checkpoints、每 epoch history、预测 CSV、MAE/SRCC/PLCC、
训练曲线、散点图、相位 mask 和中间光场。

本 baseline 沿用该 SPAQ 系列的约定：90/10 固定 train/test split，每个 epoch
在 test split 评估，并按 test SRCC 保存 `best`。因此 `best` 指标存在
selection bias；用于正式论文时应明确披露，或另建 train/validation/test 实验。
