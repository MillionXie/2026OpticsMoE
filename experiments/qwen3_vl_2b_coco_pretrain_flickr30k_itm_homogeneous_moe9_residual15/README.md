# COCO generic distillation → Flickr30k fine-tuning (MoE9 residual15)

这是一个独立的两阶段 Qwen3-VL-2B 光学 MoE 实验，不修改原 Flickr30k 工程。

第一阶段使用通用图文数据蒸馏。默认从 COCO 的 train/restval 图像中为每张图稳定选择一条人工 caption，完整冻结的电子 `Qwen/Qwen3-VL-2B-Instruct` 直接作为 teacher，不先在 COCO 上微调。teacher 缓存三处原生 DeepStack vision tap、最终 vision hidden 和最终 language answer hidden；student 联合蒸馏 vision 与 language optical MoE，不使用 Flickr 标签、二分类 head、logit KD 或 BCE。

第二阶段加载通用阶段的最终 optical checkpoint，再在 Flickr30k 固定正负 pair manifest 上微调。此时使用完整电子 Qwen teacher 的 hidden、raw binary logit、真实标签 BCE 和 router regularization。teacher 与 student 使用相同的 `LayerNorm(2048) -> Linear(2048,1)` 原始-logit head。

## 为什么默认 COCO，而不是 ImageNet

ImageNet 很适合只预训练 vision surrogate，但没有自然语言 caption，无法覆盖视觉 merger、DeepStack 注入和 language surrogate 的输入分布。这个工程的目标是 vision + language 联合替换，因此默认使用图像-caption 数据。COCO 规模适中、caption 质量稳定，适合先验证流程；需要扩大规模时，可在确认缓存和收敛正常后再接 CC3M 或经过清洗的 LAION 子集。不要把 ImageNet 分类标签模板化后当作完整的图文通用蒸馏替代品。

## 无 attention 的 Transformer 对齐

默认不复制任何原生 attention 参数：

```text
Vision:   Y_v = X_v + VisionOpticalMoE(FrozenVisionNorm(X_v))
Language: Y_l = X_l + LanguageOpticalMoE(FrozenLanguageNorm(X_l))
```

残差 identity 系数固定为 1，不增加可训练 alpha/beta，也不在残差相加后增加激活。配置位置为：

```json
"native_pre_attention_enabled": false,
"native_pre_norm_enabled": true,
"residual_enabled": true
```

因此，“不加 attention、保留 residual”在本工程中只需保持上述配置；代码已经提供 frozen norm-only prelude。若同时把 `native_pre_norm_enabled` 设为 false，则退化成 `Y = X + OpticalMoE(X)`。

## 15 层如何组织

Qwen-facing 逻辑仍保持 5 个 optical stages，但每个逻辑 stage 内连续执行 3 个物理相位传播/OEO 层：

```text
logical stage 1 = physical layers 1..3
logical stage 2 = physical layers 4..6
logical stage 3 = physical layers 7..9
logical stage 4 = physical layers 10..12
logical stage 5 = physical layers 13..15
```

每个物理层仍有 9 个独立 `120×120` expert phase masks。vision 的三个 DeepStack tap 仍位于逻辑 stages `[1,3,4]`，language 的三次原生 DeepStack image injection 仍发生在前三个逻辑 stage 之间。15 层不会被错误解释为 15 个 Qwen decoder 替换层。

## 通用蒸馏 loss

```text
L_generic = λ_v * mean(normalized vision tap MSE)
          + λ_a * normalized answer hidden MSE
          + λ_b * router balance
          + λ_i * router importance
```

这里的 answer hidden 只是通用 prompt 最后有效 token 的 teacher representation，不是任务分类答案。第一版联合训练两套 surrogate，因为 language optical stack 的输入包含 vision merger 和 DeepStack injection；分开训练容易产生接口分布偏移。

## Flickr30k 微调 loss

```text
L_finetune = λ_v * vision hidden MSE
           + λ_a * answer hidden MSE
           + λ_logit * SmoothL1(student raw logit, teacher raw logit)
           + λ_cls * BCEWithLogits(student raw logit, label)
           + router regularization
```

## SAM

SAM 是可选项。`configs/coco_pretrain_flickr30k_sam.json` 对通用阶段和 Flickr 微调都启用标准 SAM (`rho=0.05`)。每个 batch 需要两次完整 student forward/backward，运行时间和显存压力都会明显增加，因此主配置默认关闭。SAM 与随机 phase dropout 不能同时开启，避免两次 SAM forward 使用不同网络。

## 数据来源

默认通用数据源是 `HuggingFaceM4/COCO`，选择内部 `train` 和 `restval`。也支持官方本地 COCO 2017：

```json
"source": "local_coco2017",
"data_root": "../../../data/coco",
"annotations_file": "annotations/captions_train2017.json",
"images_dir": "train2017"
```

Flickr30k 继续使用 `nlphuji/flickr30k` 和持久化 pair manifest。两阶段各自拥有独立 processor/teacher cache identity；dataset、revision、manifest digest、prompt 或 pixel budget 改变时会拒绝旧 cache。

## 主要输出

```text
runs/.../
  generic_pretrain/
    dataset.json
    manifests/train.jsonl
    processor_cache/
    teacher_cache/
    checkpoints/vision_moe_final.pt
    checkpoints/language_moe_final.pt
    metrics/training_history.csv
  pair_manifests/
  processor_cache/
  teacher_cache/
  checkpoints/
  metrics/
  model.json
  config_resolved.json
```

完整架构见 `ARCHITECTURE.md`，分阶段命令见 `RUN_COMMANDS.md`。
