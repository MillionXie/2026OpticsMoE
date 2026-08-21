# P06/P07：ImageNet 通用光学 backbone 与跨任务迁移计划

更新日期：2026-08-21

## 1. 中心目标

本阶段不再把 CIFAR-10 分类准确率当作最终目标，而是训练一个可以导出全局描述和空间特征的通用光学视觉 backbone，再在不同输出形式的下游任务中检验：部署前最后一次训练得到的光学算子，能否作为固定反馈连接接近当前 BP，并稳定优于随机反馈。

主线假设是：大规模 ImageNet 语义监督与教师表征蒸馏可以把当前 P05 的“高性能且抗错位的 CIFAR 骨干”变成可迁移光学算子；下游任务只更换小型电子头，正式反馈比较仍只有 NoFT、BP-current、FA-pretrained、FA-random 四组。

## 2. 已确认的服务器资产与已有经验

- ImageNet-1K Hugging Face Arrow cache 完整：1,281,167 train、50,000 validation，1,000 类。
- 冻结 CLIP ViT-B/16 teacher cache 完整：train 为每图 4 个确定性增强视图，共 5,124,668 个 512 维 fp16 embedding；validation 为每图 1 个视图，共 50,000 个 embedding。训练/验证 completion mask 全部为 true，可直接复用。
- 下游数据已存在：Caltech-101 为 101 个目标类别；KADID-10k 有 10,125 个带 DMOS 的失真图像、81 个参考图；ISIC2016 有 900 对训练图像/掩模和 379 对官方测试图像/掩模。
- 当前磁盘可用约 1 TB。不得复制第二份 ImageNet；P06 直接读取现有 Arrow 和 CLIP memmap。

仓库里已有独立的 792×792 OpticalMixerMoE9 ImageNet 实验，但它不是本阶段主线。该模型每样本有 49 条大画幅传播路径，实际跑到 epoch 8 时 validation Top-1 仅 6.59%，单 epoch 约 3--5.4 小时。曲线仍在上升，但投入剩余 90 余 epoch 的成本和风险过高；保留其结果作为工程经验，不继续把它当通用 backbone 候选。

## 3. 要训练的 backbone 到底是什么

保持已验证的 P05/A13 物理主体不变：RGB 三通道、128×128 光学画幅、8 个光学 OEO stage、每层受限低分辨率电子残差、逐层光学 gate 不低于 0.5。初始化使用 P05 epoch 18 的 best checkpoint，而不是随机相位或表现较差的重型 MoE。

新增的是无参数特征导出协议，而不是增加一条电子分类捷径：

1. 导出 stage 2/4/6/8 的完整 `[B,3,128,128]` detector/残差融合图，供 dense task 使用；
2. 对四个 stage 分别做 4×4 adaptive average pool 和 max pool，拼接成 384 维全局描述；
3. ImageNet 预训练时增加 `LayerNorm(384) -> Linear(384,512)` CLIP projector 和 `Linear(512,1000)` classifier；它们合计约 0.71M 参数，属于预训练/读出头；
4. 原有 residual electronic processing 仍为 312,336 参数，不扩大；下游任务丢弃 ImageNet classifier，并换成任务头。

`model.forward_features()` 已作为无新增参数的 backbone 接口加入，现有 checkpoint key 和原分类前向保持兼容。

特别注意输入数值：现有 CLIP cache 对应的图像张量经过 CLIP mean/std 标准化；进入光学振幅编码前必须先反标准化回 `[0,1]` 强度，再执行平方根。直接对 CLIP-normalized 张量 `clamp_min(0).sqrt()` 会丢失大量像素信息，P06 smoke 必须用数值测试阻止这个错误。

## 4. ImageNet 预训练目标

学生使用与 cache 完全相同的增强视图，优化：

`L = 0.5 * CE_ImageNet + 1.0 * (1 - cosine(student, CLIP)) + 0.5 * KL(student_text_logits || CLIP_text_logits)`

- CE 保证 1,000 类判别能力，防止只对齐一个平滑教师空间却分类很弱；
- cosine 把 512 维学生描述拉到具有迁移性的 CLIP 图像空间；
- KL 使用已经缓存的 ImageNet text prototypes，保留教师类间结构；
- raw phase 不使用 weight decay；电子 projector/classifier 使用 `1e-4` weight decay；
- ImageNet 语义预训练使用精确 BP，因为它的终点就是后续 FA-pretrained 要冻结的 source operator。

该选择参考了 CLIP 的跨数据集迁移目标、DeiT 的教师蒸馏经验，以及 DINOv2 对“通用图像级和像素级特征”的评价原则。第一版不同时叠加 MAE 重建，避免在无法确认基本语义训练是否成立前增加另一个大解码器；若 ImageNet 分类提高而 dense transfer 明显失败，再把轻量 masked reconstruction 作为第二阶段，而不是本轮一起调参。

## 5. 确保 backbone 真正训练起来的三级门槛

### P06-E0：工程 smoke

- 64--256 个训练样本，一次 head-only 更新和一次全 backbone 更新；
- 检查 8 层 phase gradient 全部有限且非零；
- 检查 DDP 单卡/双卡前向、loss reduce、resume 后 batch/视图序列一致；
- 检查反标准化后的光学输入范围、teacher cache sample/view/label 对齐；
- 检查 `forward()` 与 `head(forward_features().final)` 数值一致；
- 不满足任一项，不进入真实筛选。

### P06-S：100k 分层屏幕

- 每类固定抽取 100 张训练图，共 100,000 张；validation 每类固定 10 张，共 10,000 张；不改变原始 cache 索引。
- 先冻结 backbone 训练 projector/classifier 1 epoch，再精确 BP 联合训练 5 epochs。
- 建议物理 GPU 3/4/5 三卡 DDP，有效 batch 96；phase LR `1e-4`，电子头 LR `5e-4`，2,000 step warm-up 后 cosine。
- 进入 full run 的预注册门槛：ImageNet-1K validation Top-1 至少 10%；student/CLIP cosine 至少 0.70；八层 phase gradient 无死亡层；最小 optical gate 不低于 0.5；phase-random 或 optical-off 后 Top-1 相对完整模型至少下降 30%；冻结 Caltech-101 probe 相对 P05 source 至少提高 3 pp。
- 指标不通过时先查输入、梯度、特征有效秩和电子旁路依赖，不直接延长到完整 ImageNet。

### P06-F：完整 ImageNet-1K

- 通过 P06-S 后，读取全部 1,281,167 张训练图，30 epochs；每图每 epoch 只使用四个 cache view 中的一个并轮换，保证每个原始样本每 epoch 覆盖一次。
- 延续 P06-S best checkpoint；不从零重跑。
- DDP 目标为 GPU 3/4/5，有效 batch 96；按 P05 的速度估计，紧凑模型单卡完整 ImageNet 一 epoch 约 35--45 分钟，三卡目标约 12--18 分钟，30 epochs 约 6--10 小时，必须以首轮真实计时修正。
- 每 epoch 保存 last、validation-best 和训练历史；每 5 epoch 保存可回退 checkpoint。
- validation-best 以 ImageNet Top-1 选择，但 checkpoint 必须同时满足 gate 和光学破坏消融，避免选择电子头绕开光学的模型。
- 完成语义预训练后另做 5-epoch P06-R 错位刷新；P06-F 与 P06-R 分开保存，判断鲁棒刷新是否保留迁移能力。

## 6. P07 下游任务：三种输出形态足够

### 任务 A：Caltech-101 分类/检索

- 排除 `BACKGROUND_Google`；每类 25 train、5 validation、其余 test，三个固定 split seeds。
- 冻结 backbone 的 384 维描述训练线性/单隐层小头；报告 Top-1、balanced accuracy、retrieval mAP 和 Recall@1。
- 这是最便宜的语义迁移和数据效率入口，同时做 1/5/10/25-shot 曲线。

### 任务 B：KADID-10k 图像质量判别

- 按 81 个 reference image 切分，严禁同一参考图的不同失真落到不同 split；否则存在内容泄漏。
- 使用 384 维全局描述和小型回归头预测 DMOS；报告 SRCC、PLCC，同时报告 distortion-type 分层结果。
- 该任务检验表征是否保留与类别无关的低层感知信息，而不是再做一次物体分类。

### 任务 C：ISIC2016 病灶分割

- 900 个官方训练对中固定 20% 作 validation，379 个官方 test 只在协议冻结后评估。
- 读取 stage 2/4/6/8 空间图，使用不超过 1M 参数的轻量 FPN/decoder；报告 Dice 和 IoU。
- 这是 dense perception 检查。如果全局分类迁移成功而分割失败，说明当前 3-channel stage map 或空间监督不足，届时才考虑 masked reconstruction/空间教师，而不是提前堆复杂结构。

## 7. 每个下游任务仍只有四组

每个任务先冻结同一个 ImageNet backbone，只训练一个共同电子任务头并保存 SHA-256；四组从该完全相同 checkpoint 开始：

| 方法 | backbone 更新 | 光学跨层反向连接 |
|---|---|---|
| NoFT | 不更新，只报告共同 head-warmup endpoint | 无 |
| BP-current | 更新 | 当前精确算子 |
| FA-pretrained | 更新 | 固定为 ImageNet 预训练结束时算子 |
| FA-random | 更新 | seed-matched 随机算子 |

电子 residual 和任务头在后三组中均使用普通 BP。正式表只比较这四组；P05 source 与 ImageNet source 的 frozen probe 差异作为“预训练是否有用”的 source 诊断，不增加第五种反馈方法。

## 8. 论文可展示结果

主结果不是一堆架构消融，而是一条因果链：

1. ImageNet 预训练曲线、Top-1/Top-5、CLIP cosine 和光学依赖；
2. 三类下游任务的 frozen-transfer 与四组微调表；
3. 数据效率曲线，比较相同四组在少样本到全数据下的恢复；
4. BP/FA-pretrained/FA-random 分层 gradient cosine 与最终性能的关联；
5. 理想部署与固定错位部署各一条迁移曲线，验证 P05 鲁棒性是否传到新任务；
6. 参数量、电子 MAC、每层 optical gate 和实际训练时间。

只有完成 ImageNet source 的多任务迁移后，才进入大模型路线：把 384 维全局描述或 stage pyramid 转成视觉 token，接到冻结语言模型/多模态模型；届时光学 backbone 的预训练算子才有合理依据作为固定反馈，而不是把 CIFAR 分类器直接包装成大模型视觉塔。

## 9. 当前实现和命令编号

- `model.py`：新增无参数 `forward_features()`，导出最终特征和全部 stage maps。
- `general_backbone_assets.py`：严格审计 ImageNet/CLIP cache、Caltech-101、KADID-10k、ISIC2016。
- command 50：运行资产审计并写入 `runs/p06_general_backbone/assets.json`。
- 后续预留：command 51 为 P06 smoke，52 为 100k 三卡 screen，53 为 full ImageNet，54--56 为三个下游任务；只有对应代码和 smoke 通过后才创建，避免留下不能运行的假命令。

参考：

- CLIP: https://arxiv.org/abs/2103.00020
- DeiT distillation: https://arxiv.org/abs/2012.12877
- DINOv2 general features: https://arxiv.org/abs/2304.07193
- MAE（仅作为 dense 失败后的候选）: https://arxiv.org/abs/2111.06377
