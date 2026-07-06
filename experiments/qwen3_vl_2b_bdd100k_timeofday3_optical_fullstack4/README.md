# Qwen3-VL-2B BDD100K TimeOfDay-3 Optical Fullstack4

本实验由已经在 CIFAR-10 上跑通的 `qwen3_vl_2b_cifar10_optical_fullstack4` 独立适配而来。Teacher 是完整电子 Qwen3-VL-2B + MLP；Student 同时将完整 vision transformer stack 和完整 language decoder stack 分别压缩为四次连续 optical conversions。

保留 Qwen processor、chat template、tokenizer、vision patch embedding/merger、token embedding、final language norm、answer-position hidden extraction 和三分类 MLP。四次 optical conversion 之间没有电子 Linear adapter，不存在 electronic residual bypass，并直接传递 detected intensity，不执行 `sqrt(intensity)`。

## 数据与类别

任务使用 BDD100K TimeOfDay-3：

```text
daytime
night
dawn_dusk
```

原始标签 `dawn/dusk` 规范化为 `dawn_dusk`。BDD100K train 作为训练来源，val 作为最终 test。主配置的 `train_limit_per_class=null`，因此数据集和 teacher cache 覆盖完整 BDD100K train split。Student 每个 epoch 使用 `train_samples_per_class_per_epoch=5000`：每类轮换抽取最多 5000 张，不永久删除其他样本，多个 epoch 后覆盖完整类别数据。test 默认使用完整 BDD100K val split。

Student sampler 会在每个 epoch 重新打乱，并在每个 batch 中尽量交错 `daytime/night/dawn_dusk`。为兼顾随机混类和 teacher cache I/O，采样器在每个类别内保留 shard 局部性，`TeacherCacheStore` 使用可配置的多 shard LRU。旧的单 shard 顺序采样会让按 ImageFolder 类别排序的数据产生大量同类 batch，现已移除。

注意：从永久 5000/class 改成完整数据后，旧 teacher cache 的样本数和 metadata 不匹配。新版主配置使用独立输出目录 `qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_full_epoch5000`，保留旧结果，并重新运行 `teacher_precompute`、`teacher_train` 和 `teacher_logits`，不会静默复用旧 cache。

数据准备优先复用仓库已有 BDD100K `_raw`，不会重复保存图片。Teacher cache 只保存完整 vision stack output、answer hidden、标签和必要边界，不保存 stack input 或中间 block 输出。Student 训练不在线运行 teacher。

训练过程每个 epoch 实时写 history/latest、validation predictions 和 best/last optical/MLP checkpoints。
