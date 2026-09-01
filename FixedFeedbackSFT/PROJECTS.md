# FA 项目目录与实验谱系

本页是 `FixedFeedbackSFT/projects` 中九个工程的权威索引。项目目录名暂不做语义重命名，以避免同时改变物理布局、模块身份和 checkpoint provenance。

## 一张图看懂主线

```text
早期可行性
  V1：CIFAR-100-C 20-stage 分类
  V2：CIFAR-100 → CIFAR-10 对比学习迁移
        │
        ▼
性能与光学依赖优化
  CIFAR 高性能骨干：A00–A13、部署扰动、早期通用预训练探索
        │
        ▼
静态图像 token backbone
  P08 ──> P09 ──┬──> P10
                 └──> P11 ──┬──> P12：三任务 fixed-feedback
                             └──> P13：渐进扩深
```

## 九个物理工程

| 编号/角色 | 物理目录 | 核心问题 | 当前状态与代表结果 |
|---|---|---|---|
| V1 | [`d2nn_cifar100c10_fixed_feedback_20stage400`](projects/d2nn_cifar100c10_fixed_feedback_20stage400/) | 小漂移下，预训练 optical adjoint 能否作为固定 connector | 20-stage、400×400 的早期完整分类验证；保留为历史基线 |
| V2 | [`d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400`](projects/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/) | fixed feedback 能否跨真实数据集迁移 | CIFAR-100→CIFAR-10 三 seed：epoch-30 BP `31.00±0.52%`、FA-pretrained `31.02±0.52%`、FA-random `28.19±2.24%`、NoFT `27.56%`；同时暴露绝对性能和 optical bypass 问题 |
| 性能线 | [`d2nn_cifar10_high_performance_optical_backbone`](projects/d2nn_cifar10_high_performance_optical_backbone/) | 先提高 BP 性能与光学依赖，再回到四组 FA | A13 四 seed CIFAR-10 Top-1 `72.34±0.14%`，optical-off `13.70%`，归一化光学依赖 `94.07±1.19%`；另含部署偏移和 P03–P07 规划 |
| P08 | [`qwen3_vl_patch_stem_8stage_optical_imagenet_backbone`](projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/) | 冻结 Qwen Patch/Position Stem 是否能提供可训练 token 输入 | 建立静态 stem、`1024→224` adapter、三 latent banks 和 8-stage ImageNet 骨架；正式训练在 epoch 9 后停止，用于后继架构对照，不是最终 backbone |
| P09 | [`qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone`](projects/qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone/) | 每层加入轻量电子空间/channel mixer 能否稳定通用预训练 | width-96 Slim Spatial Token Mixer；90 epoch ImageNet Top-1/Top-5 `49.812/74.224%` |
| P10 | [`qwen3_vl_patch_stem_8stage_dual_scale_optical_imagenet_backbone`](projects/qwen3_vl_patch_stem_8stage_dual_scale_optical_imagenet_backbone/) | 光学传播加入局部/全局双尺度归纳偏置是否有效 | 与 P09 同预算，90 epoch Top-1/Top-5 `50.888/74.956%` |
| P11 | [`qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone`](projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/) | 光学能否交替承担 token-axis 与 feature-axis mixing | 当前 source backbone；best epoch 88 Top-1/Top-5 `51.348/75.552%`，相对 P09 `+1.536 pp` Top-1 |
| P12 | [`qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa`](projects/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/) | P11 的 source operator 能否在分类、分割、姿态中复用为固定反馈 | 3 tasks × 4 methods × 3 seeds 的正式迁移已完成，并补 No-ImageNet body、phase-only、梯度和 P/E/H 机制面板 |
| P13 | [`qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone`](projects/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/) | 能否保持电子参数近似恒定，把光学主体扩至 16/32/64/100 stages | 64/100-stage 全深度 CUDA 反传审计已完成；8→16 ImageNet growth 已完整收口，best epoch 19 Top-1/Top-5 `51.428/75.752%`，相对同 run 的 8-stage 起点仅 `+0.082/+0.192 pp` |

表中结果用于定位项目，不替代正式 source data。P09/P10/P11 只有一个独立 ImageNet pretraining seed，不能据此写统计显著性；P13 虽已完成终态审计，但尚缺同预算 8-stage continuation，不能把 `+0.082 pp` 归因为有效深度收益。

## P08–P13 的架构继承关系

### P08：公共输入与部署接口

P08 只从 Qwen3-VL checkpoint 一次性提取 patch convolution 和 position table。在线静态图像前向不加载 2B 模型，不包含 Transformer、attention、语言模型、teacher loss 或 hidden-state cache：

```text
224×224 RGB
→ frozen Patch/Position Stem
→ 196×1024 tokens
→ trainable 1024→224 adapter
→ three latent optical banks
→ 8 OEO stages
→ disposable ImageNet readout
```

三路 bank 不是 RGB 三波段；RGB 已在冻结 patch convolution 中融合。

### P09：电子残差基线

P09 保留 P08 的光学相位预算，把旧低分辨 CNN bypass 换成每 stage 一个 width-96 mixer。mixer 在 14×14 token 网格上先做 gated 3×3 depthwise spatial update，再做独立 gated `96→192→96` channel MLP，并投影回 224。它是轻量电子残差，不是电子 Transformer。

### P10 与 P11：只改变光学 mixing

- P10 交替使用 5 mm 与 50 mm 传播，形成局部/全局双尺度光学 receptive field。
- P11 交替做 token-axis 和 feature-axis 单轴传播，形成 `[token-axis→feature-axis]×4` 的语义轴光学 mixer。

二者都继承 P09 的 width-96 电子 mixer 和训练协议，因此 P09/P10/P11 的 90-epoch结果是受控架构筛选，而不是三套完全不同预算的模型。

### P12：只比较四个反馈组

P12 不再增加 ImageNet 架构。每个任务只保留 NoFT/head-only、BP-current、FA-source、FA-random 四组，统一 50-epoch head-only + 50-epoch adaptation：

| 任务主指标 | NoFT | BP | FA-source | FA-random |
|---|---:|---:|---:|---:|
| Caltech-101 Top-1 | `76.961±1.930%` | `79.128±0.847%` | `79.464±1.500%` | `78.832±0.680%` |
| ISIC2016 mIoU | `81.350±0.376%` | `84.021±0.146%` | `83.992±0.152%` | `84.278±0.119%` |
| LSP PCK@0.2 torso | `50.333±0.412%` | `71.171±0.343%` | `71.205±0.079%` | `70.750±0.206%` |

这些是 `n=3` 的描述性均值±样本 SD。它们支持 FA-source “达到 BP 性能水平”，但未预定义等效界值，不能写成统计等效或稳定优于 BP。

### P13：规模工程与语义性能必须分开

P13 用 `y=x+alpha(Stage(x)-x)` 从 P11 渐进扩深。8 个 P11 anchor 保留 mixer；新增 stage 使用无参数 identity electronic skip，只增加 phase 和融合标量。因此 64/100 stage 分别有约 `9.63M/15.05M` phase 参数，而电子 body 仍约 `0.965M`。

当前已证明的是：迁移等价、全深度 feedback connector、64/100-stage CUDA 梯度覆盖，以及 16-stage ImageNet 训练可完整收口。16 层 best 只比同 run 的 8-stage 起点高 `0.082 pp` Top-1；仍需 8-stage matched continuation 和新增层 drop/reset 才能判断深度是否带来可归因的语义增益。

## 仍保留在根 `experiments/` 的共享依赖

并非所有被 FA 工程 import 的代码都属于 FA 主线。以下共享数据/损失/训练基础设施继续保留在根 `experiments/`，避免复制或把其他课题一起搬入：

- `optical_mlp_mixer_moe9_imagenet1k_clip_distill`：ImageNet dataset/settings 等公共基础设施；
- `qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval`：Caltech 数据准备；
- `qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation`：ISIC dataset/metrics；
- `qwen3_vl_embedding_2b_lsp_pose_optical_moe16`：LSP dataset/loss/metrics；
- `qwen3_vl_embedding_2b_fss1000_vision_optical_saliency`：稠密任务 objective。

这些模块仍通过 `experiments.*` 引用，是 P08/P12/P13 迁移验收的组成部分。

## 结果与代码应该去哪里找

- 每个工程的结构定义：其 `ARCHITECTURE.md` 或 `README.md`；
- 每次优化动作：其 `OPTIMIZATION_LOG.md`；
- P09/P10/P11 受控比较：[`reports/P09_P10_P11_IMAGENET_BACKBONE_COMPARISON_2026-08-29.md`](reports/P09_P10_P11_IMAGENET_BACKBONE_COMPARISON_2026-08-29.md)；
- 论文主线、图表和创新点：[`reports/teacher_report_2026-09-01/README.md`](reports/teacher_report_2026-09-01/README.md)；
- 服务器产物真实位置：[`RUN_REGISTRY.md`](RUN_REGISTRY.md)；
- 新旧布局兼容规则：[`MIGRATION.md`](MIGRATION.md)。
