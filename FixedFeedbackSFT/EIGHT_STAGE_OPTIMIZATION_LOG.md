# 8 层光学 backbone 优化复盘日志

更新日期：2026-09-02

本文件只记录已经执行的动作和可核验结果。研究动机、后续数据路线与决策门见
`EIGHT_STAGE_FOUNDATION_BACKBONE_ROADMAP.md`。

## 1. 原始平台审计

- 对象：P11 8-stage token/channel 轴向光学 backbone。
- 原训练：ImageNet-1K，90 epochs，global batch 192，seed 2026。
- 最佳：epoch 88，Top-1 51.348%，Top-5 75.552%。
- epoch 81--90 Top-1 均值 51.2434%，标准差 0.0757 percentage point；最后 5 epochs
  斜率为 -0.0096 percentage point/epoch。
- epoch 90 的学习率已经降至峰值的 5%，但 8/8 层相位梯度仍 finite/non-zero。

结论：原 90-epoch cosine schedule 已到平台，不能据此断言 8 层容量达到上限。

## 2. 不可变资产保存

2026-09-02 在服务器仓库内部创建：

- `FixedFeedbackSFT/runs/_assets/8stage`
  - content identity：`f2f14d795036785c05ce070d7cb6cdcbbbebbed8f3317f2801d1bab462727146`
  - manifest SHA-256：`318011b8f0d3f3765a8c9fff410167e4ae71fdc1fead51f8cd4ed5dd5edc2dd4`
  - P11 backbone SHA-256：`c3ad0b780dfbb3e5f8e1f7b7850c06fcb5c6d977e106f351b4602fcaadf210d2`
  - P11 epoch-88 training checkpoint SHA-256：
    `a30d5c06b61a635bb3dc379aeaca4c371c1d27e6b862c5ffd4977ce738b33034`
- `FixedFeedbackSFT/runs/_assets/16stage`
  - content identity：`14fcd0759ae96a78389c980463b28ac00691e4491b56e54bfea5f81c91b5f920`
  - manifest SHA-256：`071dc809c7f1e39f566952bc9158419d80349ad256b3bb0d022747b6d000f9dc`
  - P13 backbone SHA-256：`80b9b7b0f4415fd789bf46312dc23ccaf3600b5c4df9885c2972ce465dc9129d`
  - P13 best full-depth checkpoint SHA-256：
    `97ae579e9b33a2fc5825debde2fbad8d4fbddcbf91df5e39584036eb125ed6ec`

冻结工具会重建模型并 strict-load，核验层数、config digest、stem digest 与所有 checkpoint
tensor；同内容重复执行为 no-op，不同内容或损坏目标会拒绝覆盖。

## 3. 本轮实现

Git 提交：

- `fc23350e`：资产冻结工具、large-scale continuation trainer、proxy/formal 配置与命令；
- `8ce72c8d`：linked worktree 的 ImageNet cache 安全登记命令。

训练配方新增：

- 从冻结的 P11 epoch-88 strict-load，不复用旧 optimizer；
- AdamW、layer-wise LR decay、phase/no-norm/no-gate weight decay exemption；
- Mixup 0.8、CutMix 1.0、mean-reduced soft-target BCE；
- raw/EMA 双验证与独立 best checkpoint；
- 参数零增量 stochastic depth；
- DDP、`no_sync` accumulation、尾 accumulation window 正确缩放；
- rank-wise RNG、dataset/index/stem/初始 phase SHA 身份锁和严格 resume；
- 启动前检查 GPU 索引、UUID、显存和利用率，避免共享服务器撞卡。

## 4. 验证记录

服务器环境：

- `bash -n`：通过；
- 相关 pytest：`16 passed in 3.42s`；
- 真实 GPU smoke：通过；
- smoke 验证了前向、反向、8 层相位梯度健康门、EMA、raw/EMA checkpoint 和 backbone
  export；
- backbone trainable 参数中的光学占比：55.51097%；
- 最小光学门控：0.5000776。

第一次 smoke 在模型构建前失败，原因是 linked worktree 将相对 ImageNet cache 解析到一个
新空目录。没有发生训练或 checkpoint 覆盖。随后增加
`FixedFeedbackSFT/commands/05_register_imagenet1k_cache.sh`，只删除本次失败产生的两个空
目录，并把工作树链接到既有 312 GiB ImageNet cache；第二次 smoke 完整通过。

## 5. 5-epoch 大配方代理实验结果

两组均使用完整 ImageNet-1K、5 epochs、per-rank batch 96、2 GPUs、gradient
accumulation 2、global batch 384、seed 2026。共同起点是 Top-1 51.350%、Top-5
75.558%，loss 2.209670。

### 5.1 phase peak LR 0.002

| epoch | raw loss | raw Top-1 | raw Top-5 | EMA Top-1 | EMA Top-5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.473465 | 47.698% | 72.054% | 50.128% | 74.416% |
| 2 | 2.411436 | 48.082% | 72.616% | 49.682% | 74.118% |
| 3 | 2.393440 | 48.010% | 72.598% | 49.298% | 73.676% |
| 4 | 2.386324 | 48.206% | 72.828% | 48.964% | 73.394% |
| 5 | 2.372605 | 48.272% | 72.844% | 48.790% | 73.206% |

epoch 5 的平均相位位移是 0.0088239 rad。

### 5.2 phase peak LR 0.007

| epoch | raw loss | raw Top-1 | raw Top-5 | EMA Top-1 | EMA Top-5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.472388 | 47.726% | 72.126% | 50.180% | 74.448% |
| 2 | 2.411153 | 48.130% | 72.658% | 49.710% | 74.136% |
| 3 | 2.394255 | 47.990% | 72.564% | 49.294% | 73.684% |
| 4 | 2.386966 | 48.212% | 72.834% | 49.020% | 73.416% |
| 5 | 2.373369 | 48.308% | 72.900% | 48.838% | 73.204% |

epoch 5 的平均相位位移是 0.0277205 rad，为 0.002 配方的约 3.14 倍。所有 epoch 的
8/8 层相位梯度均 finite/non-zero，所有光学 gate 均不低于 0.5。

### 5.3 严格结论

- 两组 `best_epoch` 都是 0；任何训练后 epoch 都未超过起点，也未达到 51.448% 的代理晋级线。
- 强 Mixup/CutMix/RandAugment 破坏了已有分类边界，raw 指标在缓慢恢复，但 EMA 持续下降。
- 失败不能归因于梯度消失或光学 gate 关闭。
- 若只比较两种相位学习率，0.007 的相位更新更充分，epoch 5 raw Top-1 高 0.036 percentage
  point；但这个差距不构成“配方成功”。

## 6. 探索性长程训练与止损门

由于 0.007 在保持梯度健康的同时产生了更充分的相位运动，2026-09-02 启动一条独立的
100-epoch **探索性**续训，而不是把它登记为已经通过代理门的正式配方：

- config：`large_recipe_formal_100e_phase7e3_5gpu_gb480.yaml`；
- output：`p11_large_recipe_formal_100e_phase7e3_5gpu_gb480`；
- 物理 GPU：0、1、2、3、4；global batch 480；
- launcher PID：`1232144`；
- 重新评估起点：Top-1 51.344%、Top-5 75.558%。

任务已通过进程、5 个 DDP rank、GPU 显存和 batch 日志四项核验。它不得仅凭训练 loss 下降
继续消耗算力，采用以下人工审查门：

第 1 个 epoch 已完成：raw Top-1/Top-5 为 47.856%/72.600%，EMA Top-1/Top-5 为
50.350%/74.732%。这与 5-epoch 代理的首轮跌落一致，尚未到 epoch-10 决策点，也没有产生
新最佳。第 2 个 epoch 的 raw Top-1 为 47.752%，EMA Top-1 为 50.146%，暂未表现出恢复
斜率。第 3 个 epoch 的 raw Top-1 回升至 48.298%，EMA Top-1 为 49.714%；前三轮 raw
线性斜率为每 epoch +0.221 percentage point，但仍比起点低 3.052 percentage points。
只读门判定为 `wait_for_epoch_10`，不能把这一次回升提前解释为配方成功。

1. epoch 10：raw Top-1 至少 49.0%，且最近斜率为正；
2. epoch 20：raw Top-1 至少 50.5%，且最近斜率为正；
3. epoch 30：raw Top-1 至少回到 51.35% 起点，且最近斜率为正；
4. 未通过时登记为停止候选，由人工结合 raw/EMA、loss 和相位运动决定，不由脚本自动杀进程；
5. 最终只有 Top-1 达到 51.648%、Top-5 不低于 75.552% 且 loss 不恶化，才记为性能突破。

只读评估工具是 `FixedFeedbackSFT/tools/assess_p11_longrun_gate.py`；它只给出
`wait_for_epoch_10`、`continue_to_next_gate` 或 `manual_review_stop_candidate` 判定，不发送
信号、也不删除 checkpoint。

## 7. 并行恢复支线

代理实验说明强增广可能不适合直接接在已收敛的 P11 分类边界上，因此已经实现独立的
clean-recovery 实验：从 0.007 代理的 epoch-5 **raw** 权重 strict-load，重置 optimizer，
移除 Mixup/CutMix/RandAugment/random erasing/label smoothing/drop-path，恢复普通交叉熵与
RRC+flip。注册配方为 5 epochs、单卡 global batch 96、phase LR 8e-4；保留模型内部
`mixer_dropout=0.10`，并用 5 个确定性视图避免五轮内重复。

源 checkpoint 的 SHA-256、format、role、epoch、config digest 和 stem identity 都被写死；
源 optimizer/scheduler/scaler 不复用。该实验与探索性长程训练使用独立代码、输出目录、
checkpoint format 和实现 manifest，不能互相 resume 或覆盖。当前 GPU 5 已被其他用户占用，
因此只完成本地静态验证，正式启动由空卡门保护，待服务器测试和 GPU 释放后执行。

## 8. 后续证据链

若 ImageNet-1K 指标形成新最佳，还必须补 seeds、linear probe、full fine-tune 与多下游迁移；
ImageNet-22K 路线则必须先固定数据版本、类别 manifest 和样本 manifest，再执行大类目预训练及
同配方 ImageNet-1K 回迁对照，不能把一次 22K 训练 loss 当作 backbone 有效性的证据。

## 9. ImageNet-21K/22K 独立训练链路

已新增独立工程
`FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet22k_backbone`，
没有修改当前长程任务实现 manifest 中的任何文件。主要契约如下：

- 原始 Fall11：21,841 类、14,197,122 张、90 epochs、无验证选择，只导出 last raw/EMA；
- MIIL-P Fall11：11,221 类、训练 11,797,632 张、验证 561,052 张、80 epochs；
- 一次性生成 TSV + uint64 offsets 磁盘索引，运行时 mmap，避免每个 DDP worker 保存一千多万
  Python 路径对象；
- 由 release、WNID 顺序、精确样本数和文件 SHA 共同确定数据身份；
- 从冻结 P11 只加载 backbone，严格证明 1,000 类旧读出头没有被复制，再新建 10,450、
  11,221 或 21,841 类临时头；
- soft-target CE、Mixup/CutMix、AMP、EMA、rank-specific RNG、DDP 与严格 resume；
- 临时分类头不计入 backbone 光学参数占比，并硬检查光学占比和每层 gate 均不低于 0.5；
- ImageNet-1K 图片配 21,841 类头的 100-batch 测试只标记为 plumbing/non-publishable，
  不允许报告为 ImageNet-22K 性能。

服务器审计没有发现 ImageNet-21K/22K 数据或可用访问凭证；`/DATA/DATA1` 约剩 777 GiB，
而 Fall11 原始包本身约 1.31 TB，不能在当前盘安全地下载、解包并重复缓存。正式启动器会在
创建输出目录、初始化 CUDA/NCCL 或占用 GPU 之前核验授权数据索引、精确 manifest、资产 SHA
和磁盘预算；当前缺数据时必须 hard fail。下一步需要把授权数据挂载到容量充足的共享盘，优先
建议使用已处理的 MIIL-P Fall11，然后先做 1 epoch 真实数据 pilot，再进入 80-epoch 配方和
30-epoch ImageNet-1K 回迁对照。

## 10. 提交与服务器验证

本轮实现提交为 `66cdfe79`，已通过增量 Git bundle fast-forward 到服务器工作树；同步没有
修改当前长程训练的任何 implementation-manifest 依赖文件。服务器验证结果：

- 新增早停、clean-recovery 和 ImageNet-large tests：`25 passed in 4.68s`；
- 7 个新增/相关 shell 入口全部通过 `bash -n`；
- 原始 Fall11 formal preflight 在缺少索引时按预期抛出 `FileNotFoundError`；之后确认没有
  创建输出目录或遗留训练进程；
- ImageNet-large plumbing 的 CPU preflight 已通过，明确返回
  `publishable_result=false`、`preflight_created_output=false`；
- 当时 GPU 5 被其他用户占用（约 2.35 GiB，利用率 62--79%），plumbing 与
  clean-recovery 两个启动器都被空卡门拒绝，且没有留下 run artifacts。

因此，当前可以确认“代码、身份锁、失败安全和启动接口”有效；还不能确认真实 21K/22K
训练性能，也不能声称 plumbing GPU forward/backward 已完成。后两项分别等待授权数据挂载和
GPU 5 真正空闲。
