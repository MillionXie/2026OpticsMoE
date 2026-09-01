# 8 层光学 foundation backbone 训练路线

更新日期：2026-09-02

## 1. 当前结论

当前优先研究对象是 P11 八层 token/channel 轴向光学 backbone。原始 ImageNet-1K
训练已经完整跑满 90 epochs，最佳 checkpoint 位于 epoch 88：

- ImageNet-1K Top-1：51.348%
- ImageNet-1K Top-5：75.552%
- epoch 80 Top-1：51.198%
- epoch 90 Top-1：51.224%
- epoch 80--88 的提升：0.150 percentage point
- epoch 81--90 Top-1 均值：51.2434%，标准差：0.0757 percentage point
- epoch 81--90 线性斜率：+0.0150 percentage point/epoch
- 最后 5 epochs 斜率：-0.0096 percentage point/epoch

因此可以说它在**原 90-epoch cosine schedule 下进入平台期**，但不能说八层模型已经达到
容量上限。epoch 90 时各组学习率仅为峰值的 5%，八层相位梯度仍全部 finite/non-zero，
所以更合理的解释是原训练日程已经耗尽。

原始 P11 及完成训练的 P13 十六层 checkpoint 必须作为只读基线保存。任何新训练都使用
独立 output directory，不覆盖原 checkpoint、optimizer、history 或 result.json。

## 2. 主训练路线

### Stage A：资产冻结与可复现性

为八层和十六层各建立独立资产目录，保存：

1. Top-1 最佳 full training checkpoint；
2. 去除临时任务头的 backbone checkpoint；
3. 源文件及副本 SHA-256；
4. 模型层数、参数量、光学参数占比、stem SHA-256；
5. ImageNet 最佳指标、epoch、配置 digest；
6. feature contract 与严格加载结果。

资产副本统一放在 `FixedFeedbackSFT/runs/_assets/`，不在 `2026OpticsMoE`
工程外建立目录。

服务器使用 linked worktree 时，不复制 312 GiB ImageNet 数据；通过
`FixedFeedbackSFT/commands/05_register_imagenet1k_cache.sh` 把当前工作树的
`data/imagenet1k` 安全登记到既有数据目录。脚本只会移除失败加载产生的空目录，遇到任何
非空目标都会拒绝覆盖。

### Stage B：八层监督式 continued pretraining

从 P11 epoch-88 最佳 checkpoint 启动新的 optimizer 和 cosine schedule。第一轮采用两个
短代理学习率实验，随后将较稳定且验证指标更好的配方扩展到正式长训练。

主要训练机制：

- EMA student weights，并分别保存 student-best、EMA-best 与 last；
- soft-target BCE，配合 Mixup/CutMix；
- phase 不使用 weight decay；
- norm、bias 和 gate 不使用 weight decay；
- 电子矩阵使用 AdamW weight decay；
- 逐层学习率衰减，后层更新更快；
- phase 学习率保持在可见更新量级，不采用过小的 `1e-5` 级学习率；
- warmup + cosine restart；
- checkpoint 保存 optimizer、scheduler、AMP scaler 和 RNG 状态；
- 继续强制检查光学参数占比、光学门控下限和每层相位梯度。

这是一套受 DeiT-III/ConvNeXt 启发、针对光学相位参数做过调整的训练配方，不宣称严格
复现 DeiT-III 或 ConvNeXt。

### Stage C：teacher-free masked optical pretraining

监督长训练稳定后，增加 Optical-MAE/FCMAE-inspired 分支：

- 冻结一次性提取的 Qwen Patch/Position Stem；
- 对 14×14 token 网格施加 block mask；
- 八层光学 backbone 在完整 224×224 光场上传播；
- 临时浅 decoder 仅重建 masked RGB patches 或 stop-gradient stem tokens；
- 预训练结束后删除 decoder，只导出光学 backbone；
- 先做 200-epoch proxy，再根据 frozen linear probe 决定是否扩展到 800 epochs。

因为当前光学传播仍计算完整稠密网格，论文中只能称为
“MAE/FCMAE-inspired dense optical masked autoencoding”，不能称为严格 MAE/FCMAE。

当前不优先实现完整 DINOv2/iBOT：它们需要 EMA teacher、多裁剪和额外投影头，光学仿真
成本接近翻倍。若 teacher-free masked pretraining 失败，再将 DINO/iBOT-lite 作为第二路线。

## 3. 数据路线

ImageNet-1K 暂时不替换，它继续承担统一监督训练、linear probe 和 full fine-tune 评估。
MAE、iBOT 与 FCMAE 都证明 ImageNet-1K 本身可以作为有效的自监督预训练数据。

扩展上游数据时按以下顺序：

1. **ImageNet-21K/22K**：首选。静态图像、训练接口与当前 ImageNet 管线一致，并有成熟的
   `IN-22K pretrain -> IN-1K finetune` 配方；服务器当前尚未发现该数据集。
2. **ImageNet-1K + COCO2017 + ABO**：服务器已有，可用于无标签 masked pretraining；
   必须先做去重、域采样权重和数据 manifest，不能直接混在一起。
3. **Places365 / Objects365**：需要场景或检测分布时再加入。
4. **LAION/DataComp 类网页数据**：下载、许可、清洗与重复图像问题较重，暂不优先。

DINOv2 的 LVD-142M 不是公开可直接复现的数据集，因此本项目不能声称按 DINOv2 数据规模
训练。

## 4. 正规 backbone 的验收矩阵

ImageNet Top-1 只是其中一个指标。八层 backbone 的正式版本至少需要同时通过：

1. ImageNet-1K k-NN；
2. 冻结 backbone linear probe；
3. ImageNet-1K full fine-tune；
4. 一个分类迁移任务；
5. 一个稠密任务（分割、关键点或深度）；
6. 一个分布外或鲁棒性任务；
7. 光学 phase-only 或固定电子 scaffold 的适配实验；
8. BP、FA-source、FA-random 和 NoFT 四组统一协议。

只有在多个输出形式上均获得可迁移特征，才称为通用 backbone；只在 ImageNet 分类头上训练
并不能单独证明这一点。

## 5. 训练决策门

- 短代理实验首先检查：无 NaN/OOM、八层 phase gradients 全部 finite/non-zero、光学门控
  不低于 0.5、EMA 与 raw 权重都能严格恢复。
- 若 5-epoch proxy 的最佳验证 Top-1 比 51.348% 至少提高 0.10 pp，或呈稳定上升趋势，
  则进入正式 100-epoch continuation。
- 正式续训只有达到 Top-1 51.648%（相对原最佳至少 +0.30 pp）、Top-5 不低于
  75.552% 且 validation loss 不恶化，才记为突破原平台。
- 若两个 phase LR 都导致显著退化，保留原资产并改做更低电子 LR/无 Mixup 收尾，而不覆盖
  原 P11。
- 正式续训达到新最佳后，重新执行 P12 的冻结 linear probe 和三类下游迁移；不能只比较
  ImageNet Top-1。
- 若监督 continuation 仍无收益，再进入 teacher-free masked optical pretraining；不是先
  盲目更换数据集。

## 6. 一手参考

- DeiT-III official recipe: https://github.com/facebookresearch/deit/blob/main/README_revenge.md
- ConvNeXt training: https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md
- MAE pretraining: https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md
- ConvNeXt-V2/FCMAE: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/TRAINING.md
- iBOT: https://github.com/bytedance/ibot
- DINOv2: https://github.com/facebookresearch/dinov2
- ImageNet-21K processing: https://github.com/Alibaba-MIIL/ImageNet21K
