# P12：P11 通用骨干的下游固定反馈迁移

P12 不是再增加一种 ImageNet 架构，而是检验已经完成预训练的 P11
骨干在不同输出形态上的迁移能力，以及预训练光学算子作为固定反向连接
（fixed feedback）时是否仍然有效。

本实验只保留四个论文主表组：

```text
Task | NoFT | BP-current | FA-pretrained | FA-random
```

任何读出头选择、学习率筛选、P09/P10 架构对比或破坏性推理消融都只作为
诊断，不能增加第五个反馈方法组。

## 1. 核心问题与源骨干

主问题是：在相同下游前向传播、相同初始化和相同优化预算下，固定的
ImageNet 预训练光学算子能否替代每一步都更新的精确反向算子，并且比
匹配维度的随机光学反馈稳定地更好？

唯一源骨干是 P11 的 ImageNet-1K 90-epoch 训练中按验证 Top-1 选出的
epoch-88 `backbone.pt`：

```text
experiments/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/
  runs/p11_imagenet1k_pretrain_bs96_90e/checkpoints/backbone.pt
```

源实验的已验证结果为 ImageNet-1K Top-1/Top-5 `51.348%/75.552%`。这是
P11 的预训练结果，不是 P12 的下游结果。P12 每次启动都应记录源文件
SHA-256、架构签名、八层源相位和冻结 stem 的 SHA-256，并拒绝 P09/P10
或不完整 checkpoint。

## 2. 共享前向结构

```text
224x224 RGB
  -> 冻结 Qwen Patch/Position Stem（196 个 1024-D token；无 Transformer）
  -> 可训练 1024->224 adapter
  -> 8 个 P11 OEO stage：[token-axis, channel-axis] x 4
  -> 仅使用最后一层特征的临时任务头
```

P11 中的三路光学 bank、八个物理相位面、宽度 96 的轻量电子残差 mixer、
受约束光学门和 token 排列全部从源骨干严格载入。每层电子旁路在三路
bank 间共享权重，具体为 `224 -> LN -> 96`，先做带 sigmoid 门的
3x3 depthwise 空间 token 混合，再做带独立 sigmoid 门的
`96 -> 192 -> 96` channel MLP，最后投影回 224，并用上限 0.25 的残差
尺度合并。token-axis 光学传播前，
196 个有效 token 从 Qwen 2x2 block-major 顺序还原为真实二维 row-major；
传播后再还原为 Qwen 顺序。channel-axis 层不使用这组 token 置换。

可复用骨干具有 `1,204,224` 个光学相位参数、`231,648` 个 adapter 参数
和 `733,472` 个八层 mixer/融合门参数，即可训练电子骨干共 `965,120`。
临时任务头不计入骨干时，光学参数占比为 `55.511%`，并在运行时强制
不少于 `50%`。约 `0.988M` 参数的冻结 Qwen stem 不属于可训练参数，但其
推理计算必须单独披露，不能把它隐藏为“无电子前端”。每个任务的临时头
精确参数量仍以该 run 的 `model_report.json` 为准。

光学门下界 `0.5` 是数值融合系数约束，不等同于实际光功率、能耗占比或
硬件计算占比。任务头是必要的电子读出接口，因此同时报告“排除任务头的
可复用骨干光学占比”和“包含临时头的整体占比”，但论文的 backbone
约束使用前者。

## 3. 三个任务和读出头

| 任务 | 数据与输出 | 临时电子头 | 主指标 | 次指标 |
|---|---|---|---|---|
| Caltech-101 | 101 类分类，并以 val 为 gallery、test 为不相交 query 做检索评估 | 三 bank 凸组合，token LN，mean/max 拼接为 448-D，`LN -> Linear(448,256) -> GELU -> Dropout -> Linear(256,101)`；归一化 256-D hidden 用于检索 | Top-1 | balanced accuracy、跨 split Recall@1、mAP |
| ISIC 2016 | 224x224 二值病灶分割 | 最后一层 224x14x14 网格，宽度 64 的无 attention 深度可分离渐进解码器，输出 224x224；头参数严格小于 1M | mean IoU | Dice、sensitivity、specificity |
| LSP | 14 个关键点热图定位 | 同一解码器族，输出 14x56x56 热图；头参数严格小于 1M | PCK@0.2 torso | PCKh@0.5、per-joint PCK |

稠密任务有意只连接最终 stage。若直接从第 2/4/6 层建立 U-Net 式旁路，
早期光学层会获得更短的精确梯度路径，八层固定反馈比较就不再干净。
只有在 BP-current 稠密任务本身无法越过性能门槛时，才允许把共享多尺度头
作为后续独立架构消融，而且四组必须一起更换。

正式 split 由数据 manifest 固定并做泄漏检查：

- Caltech-101 排除 `BACKGROUND_Google`，每类固定 25 train、5 val，
  其余 test；完整数据通常为 2525/505/5647。
- ISIC 2016 官方 900 对 train/val 按 seed 固定为 720/180，官方 379 对
  test 在完成验证选择前封存。
- LSP 使用验证过的 HR-LSPET 与 LSP 训练池，按来源分层做 90/10
  train/val，并保留官方最后 1000 张 LSP test；完整记录为
  9385/1043/1000。

每个 run 的实际样本列表、数量和 SHA-256 以生成的 manifest 与
`dataset_summary.json` 为最终依据。

## 4. 四组的准确含义

| 方法 | 下游前向 | 更新内容 | 传向前一光学 stage 的连接 |
|---|---|---|---|
| `noft` / NoFT | 源 P11 | 只训练任务头；骨干完全冻结 | 无 |
| `bp` / BP-current | 当前 P11 | adapter、8 层相位、电子残差和任务头 | 当前算子的精确导数 |
| `fa_pretrained` / FA-pretrained | 与 BP 完全相同的当前 P11 | 与 BP 完全相同 | 冻结的 ImageNet 源相位/算子 |
| `fa_random` / FA-random | 与 BP 完全相同的当前 P11 | 与 BP 完全相同 | seed 匹配且全程冻结的随机相位/算子 |

FA 两组不是冻结前向：每个光学层在前向及其本层相位梯度中始终使用当前
可训练相位，只有传播到前一 stage 的误差连接被替换。adapter、电子残差
和任务头始终使用普通 BP。FA-random 也不是随机前向基线。
FA-random 的反馈 seed 固定为 `task seed + 8,000,003`，只用于构造运行时
反馈相位，不修改共同前向初始化。

## 5. 统一的 50 + 50 训练协议

三个正式 seed 均为 `2026, 2027, 2028`。对每个 task/seed：

1. 从同一个 P11 源 checkpoint 开始，冻结完整骨干，只训练任务头恰好
   50 epochs；在 50 个 epoch 内按验证主指标选取 `common_start.pt`。
   该端点同时就是 NoFT 结果。
2. BP-current、FA-pretrained、FA-random 从字节一致的
   `common_start.pt` 启动，各自再训练恰好 50 epochs。
3. 验证集只选择报告 checkpoint，不提前停止。test 只在选择完成后评估。

因此 NoFT 本身训练 50 epochs；每个适配端点代表 `50 epoch head-only +
50 epoch adaptation` 的完整流水线。这里的“全部 50 epoch”不能被解释成
三个适配组总共共享 50 epoch。

共享优化设置：

- AdamW；phase LR `3e-3`、adapter/residual LR `2e-4`、head LR `1e-3`；
- phase weight decay `0`，电子参数 weight decay `5e-4`；
- 1 epoch warm-up 后 cosine decay，最低 LR 比例 `0.05`；
- AMP；phase/electronic gradient clip 分别为 `2.0/5.0`；
- batch size：Caltech train/eval `96/192`，ISIC 与 LSP `32/64`；
- 三个适配方法在同一任务内共享 split、增强、batch、scheduler 步数和
  所有超参数，禁止对某个 FA 方法单独调参。

损失函数：Caltech 为 label smoothing `0.1` 的交叉熵；ISIC 为
`1.0*BCE + 1.0*Dice + 0.75*soft-IoU + 0.25*boundary`；LSP 为可见
关键点 masked heatmap MSE 加 `0.1*masked coordinate loss`。

## 6. 公平性与解释边界

- 每个适配组必须记录并校验完全相同的 common-start SHA、源 checkpoint
  SHA、数据 manifest SHA 和 config digest。
- 初始化时四组普通前向输出应数值一致。BP-current 与 FA-pretrained 在
  尚未偏离源相位时的逐层相位梯度 cosine 应接近 1；FA-random 应明显
  分离。该检查失败时不解释正式结果。
- Qwen stem 在所有组中冻结且 buffer 不得变化；所有光学门保持不低于
  0.5；每层 phase、adapter、残差和任务头应具有有限、非零的预期梯度。
- phase-random、optical-off、electronic-skip-off 是训练后破坏性依赖
  诊断，不是重新训练的独立“光/电性能”，不能据此拆分性能贡献。
- 若 BP-current 没有相对 NoFT 产生真实适配增益，该任务对 fixed
  feedback 的 recovery 比率不具诊断性。

高者更优指标的 BP 增益恢复率定义为：

```text
recovery(method) = (M_method - M_NoFT) / (M_BP-current - M_NoFT)
```

预注册的支持条件为：BP 确有下游增益；FA-pretrained 恢复至少 80% BP
增益；FA-pretrained 在配对 seed 上优于 FA-random；梯度几何与性能排序
一致。只在 Caltech 成立只能说明语义迁移，不能宣称通用 backbone。

## 7. 五 GPU 执行与产物

正式队列默认只使用物理 GPU `1,2,3,4,5`，每卡同时最多一个 P12 run。
队列在每次派发前以 `nvidia-smi` 的 compute-app 记录重新检查所有权；低
瞬时利用率不等于空闲，检测到他人进程的设备不得使用。NoFT 完成并生成
合法 common start 之前，相同 task/seed 的三个适配作业不能启动。

所有 run 位于：

```text
runs/p12_downstream_fa_50e/<task>/<method>/seed_<seed>/
```

每个完整 run 至少应包含 resolved config、launch 信息、数据摘要和
manifest、模型/反馈报告、完整 50 行 epoch history、`last.pt`、验证最优
`best.pt`、梯度诊断、最终 test 与破坏性消融以及
`result.json(status=complete)`。NoFT 还应生成：

```text
runs/p12_downstream_fa_50e/<task>/common/seed_<seed>/common_start.pt
```

没有进程不表示训练完成；只有 `result.json` 明确为 `complete`、本 run
记录 50 个完成 epoch 且 checkpoint/哈希校验通过才算完成。

服务器启动、监控、单任务恢复与汇总命令见本实验的
`commands/P12_DOWNSTREAM_FA_50E_COMMANDS.md`。运行器默认 resume；不要在
已有正式目录上使用 `--no-resume`。

## 8. 当前结果状态

本 README 记录的是已锁定方案和实现接口，不填充尚未完成的下游成绩。
正式数值只能由 `result.json` 和跨 seed 汇总器写入
`OPTIMIZATION_LOG.md` 的结果模板。P11 的 ImageNet 源成绩不得抄成 P12
的 NoFT 或迁移成绩。

## 9. 无 ImageNet 光学骨干预训练辅助控制

为了把“P11 ImageNet 预训练收益”与“下游训练器本身的收益”分开，新增
Scratch-P11-body 辅助控制。它保留完全相同的冻结 Qwen patch/position stem、
P11 架构、任务头和 `50 epoch head-only + 50 epoch adaptation` 协议，但不会读取
任何 P11 ImageNet backbone checkpoint；adapter、8 个相位面、8 个电子 mixer
及门控均由指定 seed 一次性全新初始化。准确名称是“无 ImageNet 光学 body
预训练”，而不是全模型从零训练，因为 Qwen stem 仍来自同一个冻结 artifact。

该控制复用原有四个 method key。此时 `fa_pretrained` 只为保持队列和表格接口
一致，其论文标签必须写成 `FA-source-init`：固定反馈来自随机 source 的初始
相位，而不是 ImageNet 预训练相位。默认先跑 3 tasks × seed 2026 × 4 groups，
通过后可原样扩展到 downstream seeds `2026,2027,2028`。所有产物使用独立
`p12_scratch_*` 目录，不能与主表的预训练 P12 结果混合。

由于 source 文件 SHA 只有序列化后才存在，仓库不提交伪 SHA 配置。
`commands/p12_scratch_downstream_50e.sh prepare` 先生成带 semantic tensor digest、
`source_regime`、`init_seed`、stem SHA 和 model report 的 source，再把真实文件
SHA 写入隔离目录中的正式 resolved config。完整命令与解释边界见
`commands/P12_SCRATCH_DOWNSTREAM_50E_COMMANDS.md`。
