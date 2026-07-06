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

原始标签 `dawn/dusk` 规范化为 `dawn_dusk`。BDD100K train 作为训练来源，val 作为最终 test。默认 `train_limit_per_class=5000`，因此 teacher precompute、teacher MLP 和 student 训练最多使用 15000 个训练来源样本；其中再按类别分层切分 10% validation，约为 13500 train + 1500 validation。test 默认使用完整 BDD100K val split。

数据准备优先复用仓库已有 BDD100K `_raw`，不会重复保存图片。Teacher cache 只保存完整 vision stack output、answer hidden、标签和必要边界，不保存 stack input 或中间 block 输出。Student 训练不在线运行 teacher。

训练过程每个 epoch 实时写 history/latest、validation predictions 和 best/last optical/MLP checkpoints。

