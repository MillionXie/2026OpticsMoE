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

## 5. 正在运行的代理实验

启动时间：2026-09-02 01:17（Asia/Shanghai）。两组都使用完整 ImageNet-1K、5 epochs、
per-rank batch 96、2 GPUs、gradient accumulation 2、global batch 384、seed 2026。

| 组 | 唯一变化 | 物理 GPU | launcher PID | 初始 Top-1 / Top-5 |
|---|---:|---|---:|---:|
| low-phase | phase peak LR 0.002 | 0(4090) + 2(3090) | 978207 | 51.350% / 75.558% |
| high-phase | phase peak LR 0.007 | 1(4090) + 5(3090) | 978209 | 51.350% / 75.558% |

运行目录：

- `FixedFeedbackSFT/runs/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/p11_large_recipe_proxy_5e_phase2e3_2gpu_gb384`
- `FixedFeedbackSFT/runs/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/p11_large_recipe_proxy_5e_phase7e3_2gpu_gb384`

日志软链接：

- `.../logs/p11_proxy_phase2e3.latest.log`
- `.../logs/p11_proxy_phase7e3.latest.log`

启动后核验：GPU 0/1/2/5 各占用约 9.2--9.4 GiB，利用率 95--100%；两个 torchrun
launcher 与四个 DDP rank 均存活。首个 optimizer step 已越过强制的 8/8 层 phase
finite/non-zero 检查。

## 6. 下一决策

当前没有启动 100-epoch 正式训练。待两个 5-epoch proxy 完成后：

1. 比较 raw/EMA Top-1、Top-5、validation loss、phase motion 与 gate；
2. 达到 51.448% 或呈明确持续上升趋势的配方才可晋级；
3. 选择一个 phase LR，在确认五张 GPU 空闲后启动独立 100-epoch run；
4. 正式结果需达到 Top-1 51.648%、Top-5 不低于 75.552% 且 loss 不恶化，才记为突破；
5. 新最佳必须再做 seeds、linear probe、full fine-tune 与下游迁移，不能只报告 ImageNet
   单次 Top-1。
