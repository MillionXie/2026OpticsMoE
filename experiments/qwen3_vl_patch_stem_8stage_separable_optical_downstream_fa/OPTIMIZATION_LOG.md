# P12 构建、优化与训练复盘日志

本文件只记录可由代码、配置、checkpoint 或日志复核的事实。运行中的
临时指标必须标为“中间值”，不存在的结果保持 `—`，不得用预期值补齐。

## 2026-08-31：实验问题与预算锁定

用户要求开始构建并训练 P12，三个任务全部统一为 50 epoch，并使用五张
空闲 GPU，而不是仅使用一张。为避免“50 epoch”解释不一致，协议锁定为：

```text
每个 task/seed：
  P11 -> 50 epoch frozen-backbone head-only -> NoFT/common_start
  common_start -> 50 epoch BP-current
  common_start -> 50 epoch FA-pretrained
  common_start -> 50 epoch FA-random
```

正式 seed 为 `2026, 2027, 2028`，任务为 Caltech-101 分类/检索、ISIC
2016 分割和 LSP 姿态。主表始终只有 NoFT、BP-current、FA-pretrained、
FA-random 四组。

### 源模型证据

- 来源：P11 ImageNet-1K 90-epoch 预训练的验证最优 `backbone.pt`；
- 验证选择：epoch 88；
- 已记录源 Top-1/Top-5：`51.348%/75.552%`；
- 八层结构：`[token-axis, channel-axis] x 4`；
- 光学相位参数：`1,204,224`；
- 可训练电子骨干：adapter `231,648` + 八层 mixer/融合门 `733,472` =
  `965,120`；每层 mixer 宽度 96，含独立空间/通道门控残差；
- 排除临时任务头的可复用骨干光学参数占比：`55.511%`；
- Qwen Patch/Position Stem：始终冻结，只含 patch/position 前端，不含
  Qwen Transformer 或 language model。

上述只是预训练源模型证据，不能当作 P12 下游性能。

### 本轮架构动作及理由

1. 建立一个共享 P11 下游 wrapper，严格加载架构签名、源相位和 stem
   哈希；移除旧 ImageNet 分类 readout，避免把预训练临时头带入迁移。
2. 全部任务只从第 8 stage 读取特征，防止稠密解码器给早期层建立绕过
   fixed-feedback 的短精确梯度路径。
3. Caltech 使用 448-D mean/max 全局描述符、256-D 小型 MLP 分类头，并
   复用归一化 hidden 做检索；不增加第二个独立训练头。
4. ISIC/LSP 使用同一族宽度 64、depthwise-separable、无 attention 的
   渐进式空间解码器。ISIC 输出 224x224，LSP 输出 14x56x56，单个临时
   稠密头强制小于 1M 参数。
5. 四组共享一个 seed-specific NoFT common start。这样比较的是反向
   连接，而不是不同任务头初始化或不同的 50-epoch warm-up 轨迹。
6. 保留较积极的 phase LR `3e-3`，phase 不做 weight decay，并记录相位
   环形漂移，避免“相位几乎没动”却把结果解释成光学层有效适配。
7. 使用依赖感知队列占用物理 GPU 1--5。每张卡每次只有一个 run，派发
   前检查 compute-app 所有权；相应 NoFT 完成后才释放三个适配作业。

### 当前实现口径

- 数据 split 和增强由 seed 固定并生成 manifest SHA；test 不参与调参。
- Caltech 损失：label smoothing 0.1 的交叉熵；主指标 Top-1。
- ISIC 损失：BCE、Dice、soft-IoU、boundary 的固定加权组合；主指标
  mean IoU。
- LSP 损失：masked heatmap MSE + `0.1 * masked coordinate loss`；
  主指标 PCK@0.2 torso。
- 更新组训练 adapter、八层 phase、电子残差 mixer 和临时任务头；冻结
  Qwen stem。NoFT 只训练任务头。
- FA 只替换跨 stage 的误差连接；当前相位仍用于正常前向和本层相位
  局部梯度。电子路径始终使用普通 BP。
- 每个验证最优模型执行 normal、optical-off、phase-random、
  electronic-skip-off 四种最终推理；这些是依赖诊断，不是四个新增方法。

### 2026-08-31：正式启动前的公平性与可恢复性加固

在 GPU 正式训练前完成了第二轮代码审计，并把以下问题作为阻塞项修复；
这些动作改变的是实验可信度与故障恢复，不增加新的比较组：

1. 更新组在训练 epoch 1 前先评估同一个 common start，并把它保存为可选的
   epoch 0。若 50 epoch 适配全部退化，验证选择可诚实回退 NoFT，而不会被迫
   报告更差的 epoch 1；NoFT 本身仍只允许在完整训练过的 epoch 1--50 中选择。
2. `result.json`、`last.pt`、`best.pt` 和 common start 均校验 task/method/seed、
   split manifest、固定源 P11 SHA、common-start SHA、feedback manifest、源 phase
   SHA 与实现指纹。实现指纹覆盖 P12 以及实际调用的光学层、stem、数据、损失和
   指标源码；代码语义变化后禁止静默续接旧 checkpoint。
3. 源 P11 `backbone.pt` 的预注册 SHA-256 锁定为
   `c3ad0b780dfbb3e5f8e1f7b7850c06fcb5c6d977e106f351b4602fcaadf210d2`。
   错误或被替换的 P09/P10/P11 文件在训练前失败，而不是产生一组新身份不明结果。
4. 每 batch 在 backward/step 前执行有限 loss 与 unscale 后梯度范数门禁；初始、
   epoch 1/5/10/20/30/40/50 和最终选中点保存逐层 BP/FA phase-gradient cosine、
   norm ratio，以及 adapter/residual/head 梯度组健康状态。
5. checkpoint 改为先原子保存改善后的 `best.pt`，再保存 `last.pt`，缩小意外中断
   时 last 宣称一个尚未落盘 best 的窗口；初始 launch 不再被 resume 覆盖，恢复
   事件追加到 `resume_lineage.json`。
6. 冻结 Qwen stem 的完整 state digest 在源、common、checkpoint 和最终选中状态间
   强制一致；逐 epoch 检查光学门不低于 0.5，并落盘电子 mixer 双门、峰值 CUDA
   显存、吞吐、相位漂移和三种破坏性消融。
7. Caltech 检索改为 505 张 validation gallery 对完全不相交的 test query，取消
   同集合检索；formal split 计数锁定为 Caltech `2525/505/5647`、ISIC
   `720/180/379`、LSP `9385/1043/1000`。
8. smoke 使用独立 `--output-root`。BP/FA smoke 只在该隔离目录创建带
   `synthetic_smoke_only` 标记的临时 common，以实际测试严格 common-load 链；
   任何 smoke 文件都不能进入正式 run root。

本节记录的是已实现的工程动作，不表示下游 50 epoch 结果已经产生。正式状态仍只
由下表、队列状态和通过身份校验的 `result.json` 更新。

## 启动前验收清单

以下项目全部通过后才能把作业标为“formal”：

- [ ] 本地与服务器 Git commit 一致，正式修改均已提交，服务器使用
  `git pull --ff-only` 同步；无关 dirty 文件未被覆盖。
- [ ] 三个数据根目录、P11 `backbone.pt` 和冻结 stem checkpoint 均存在。
- [ ] 数据审计通过：split 无图像泄漏，实际计数和 manifest SHA 已写入。
- [ ] 单元测试与语法检查通过；真实 P11 checkpoint 的单 batch GPU smoke
  在 Caltech/ISIC/LSP 和四种 feedback 配置上均通过。
- [ ] P11 参数量、架构签名、axis schedule、源 phase SHA 和 stem SHA
  与锁定值一致；P09/P10 strict-load 明确失败。
- [ ] common start 时各方法 normal forward 数值一致。
- [ ] BP-current 与 FA-pretrained 初始化逐层梯度 cosine 接近 1，且所有
  应更新 phase/adapter/residual/head 梯度有限、非零。
- [ ] 光学门下界不低于 0.5；冻结 stem 参数和 buffer 不变。
- [ ] GPU 1--5 在启动瞬间均通过 compute-app 所有权检查；每卡一作业，
  每个作业有独立 PID、log 和 run 目录。
- [ ] resume 测试通过，随机反馈能由 seed 重建；正式 run 默认 resume。

## 正式训练状态

下表必须从 `result.json`/队列状态更新；“进程消失”不能填成“完成”。

| Task | Seed | NoFT 50e/common | BP 50e | FA-pretrained 50e | FA-random 50e | 备注 |
|---|---:|---|---|---|---|---|
| Caltech-101 | 2026 | complete (50/50; test Top-1 0.78644) | — | — | — | NoFT common SHA `27aed054...c9ec` |
| Caltech-101 | 2027 | — | — | — | — | 未在本记录中写入正式结果 |
| Caltech-101 | 2028 | — | — | — | — | 未在本记录中写入正式结果 |
| ISIC 2016 | 2026 | — | — | — | — | 未在本记录中写入正式结果 |
| ISIC 2016 | 2027 | — | — | — | — | 未在本记录中写入正式结果 |
| ISIC 2016 | 2028 | — | — | — | — | 未在本记录中写入正式结果 |
| LSP | 2026 | — | — | — | — | 未在本记录中写入正式结果 |
| LSP | 2027 | — | — | — | — | 未在本记录中写入正式结果 |
| LSP | 2028 | — | — | — | — | 未在本记录中写入正式结果 |

## 逐 run 复盘模板

每次完成、恢复、失败或主动停止一个 run，都追加一段，不覆盖旧记录：

```markdown
### YYYY-MM-DD HH:MM — <task>/<method>/seed_<seed>

- 状态：started | resumed | interrupted | failed | complete
- Git commit / config digest：
- 物理 GPU / GPU UUID / launcher PID / worker PID：
- 源 checkpoint SHA / stem SHA / manifest SHA / common-start SHA：
- 预算：head-only 50 或 adaptation 50；完成 epoch：x/50
- batch、有效每 epoch step、AMP、峰值显存、吞吐：
- 验证最优：epoch x，primary metric = ...
- 最终 test normal：主指标与次指标
- phase：平均/逐层漂移、>0.1 rad 比例
- gradient geometry：逐层 cosine、norm ratio，记录 epoch 0/早期/最优
- optical gates / electronic residual gates：
- destructive ablations：optical-off / phase-random / electronic-skip-off
- checkpoint 健康：last/best 可加载，history 行数，result.status
- 异常、处置与是否改变正式 recipe：
```

若发生 OOM，只允许把该任务的 batch size 对四组统一下调并重新计算
scheduler；不得只给某个方法减 batch。若代码或配置有科学含义变化，应
生成新的 run identity，并在此记录旧结果为何不可直接合并。

## 跨 seed / 跨任务结果模板

主表报告 `mean +/- sample std`，同时保存每 seed 的配对差值和 bootstrap
置信区间。以下表格在三 seed 全部完成前保持空白：

| Task（主指标） | NoFT | BP-current | FA-pretrained | FA-random | FA-pretrained BP-gain recovery |
|---|---:|---:|---:|---:|---:|
| Caltech-101（Top-1） | — | — | — | — | — |
| ISIC 2016（mean IoU） | — | — | — | — | — |
| LSP（PCK@0.2 torso） | — | — | — | — | — |

解释时依次检查：

1. BP-current 是否确实高于 NoFT；否则该任务不用于 recovery 结论。
2. FA-pretrained 是否恢复至少 80% BP 增益。
3. FA-pretrained - FA-random 的配对差是否跨 seed 同号；若变号，在强
   结论前将该任务扩展到五 seed，但不改变四组定义。
4. 性能排序是否与逐层 gradient cosine 排序一致。
5. 结论是否覆盖至少两类输出形态；单一任务不写“通用 backbone”。

## 命令和记录位置

- 方案及结构说明：`README.md`
- 共享正式配置：`configs/base_50e.yaml`
- 可复现实验命令：本实验目录
  `commands/P12_DOWNSTREAM_FA_50E_COMMANDS.md`
- Linux 五卡入口：`commands/p12_downstream_fa_50e.sh`
- Windows 到服务器入口：`commands/p12_downstream_fa_50e.ps1`
- 正式 run：`runs/p12_downstream_fa_50e/`

本文档创建时没有因“写文档”而启动或伪造任何正式训练。之后每次代码、
配置、启动方式或训练状态变化，都应在本文件追加有时间戳的事实记录。

当前机器汇总器为 v2：它可以审计完成数、逐 run 主/次指标、
相对 NoFT 的配对差值、mean/sample std、配对 bootstrap CI、BP-gain
recovery，以及梯度几何、相位、门控、吞吐和破坏性消融汇总。缺失值
保持 `null`，三个 seed 完成前不将少样本 CI 或中间值写成论文结论。

## 2026-08-31 01:05--01:08 +08:00：服务器验收与正式启动

### Git/worktree 证据

- 本地 `main`、`origin/main` 与本次训练源码均锁定到
  `e305e0b0b20f37757d44136c158efa92cec7f4cc`
  (`Build P11 downstream fixed-feedback transfer suite`)。
- 服务器原工作树 `main=113061a` 存在未推送的分叉提交，并有其他项目的
  tracked/untracked dirty 文件；`git pull --ff-only` 因非 fast-forward 拒绝。未使用
  reset/checkout/stash，未覆盖或纳入这些无关变更。
- 正式训练在服务器独立 detached worktree
  `/DATA/DATA1/guest3/2026OpticsMoE_p12_e305e0b` 运行，HEAD 精确为上述已推送
  commit。只软链接两个 ignored checkpoint 文件和三个数据目录，没有链接
  dirty 源码或旧 P12 run。
- 共享配置文件 SHA-256：
  `357ee089be6857240e09801580fc38b64044ab2d5072ee340e01572b784c9bfa`；
  implementation SHA-256：
  `c61ee3bbbabe6937f574987bb48452c0bb7d74502ef839628e55c698910d6fbd`。
- P11 source SHA-256：
  `c3ad0b780dfbb3e5f8e1f7b7850c06fcb5c6d977e106f351b4602fcaadf210d2`；
  frozen stem checkpoint SHA-256：
  `e3b12b274211d29f928eee95fdfc60b32d10751f1bbdc98cd63f0cccd0792485`。

### 服务器测试与隔离 smoke

- Python：`/home/guest3/miniconda3/envs/xml/bin/python`；主机：`100right`。
- P12 服务器测试：`59 passed in 8.72s`；Linux 启动脚本 `bash -n`通过；
  `run --help`、`queue --help`、`summarize --help` 全部正常退出。
- smoke 根目录为
  `/DATA/DATA1/guest3/2026OpticsMoE_p12_e305e0b/experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/runs/p12_smoke_e305e0b_20260831`，
  与正式 root 隔离。所有更新组只在各自 smoke root 使用
  `synthetic_common_start_only=true` 的临时 common。

| Task | Method | 配置 batch | 峰值 CUDA GiB | 单批验收耗时 s | gradient cosine(1--7) | 结果 |
|---|---|---:|---:|---:|---:|---|
| Caltech-101 | NoFT | 96 | 2.022 | 5.452 | 不适用 | passed |
| Caltech-101 | BP-current | 96 | 8.293 | 7.322 | 1.000000 | passed |
| Caltech-101 | FA-pretrained | 96 | 7.578 | 7.629 | 1.000000 | passed |
| Caltech-101 | FA-random | 96 | 7.579 | 6.217 | 0.716147 | passed |
| ISIC 2016 | BP-current | 32 | 7.590 | 34.294 | 1.000000 | passed |
| ISIC 2016 | FA-pretrained | 32 | 7.288 | 54.874 | 1.000000 | passed |
| LSP | BP-current | 32 | 3.262 | 12.989 | 1.000000 | passed |

Caltech/ISIC/LSP manifest SHA 分别为
`7824b8dd4e2cd9f969205d62e298f1e26d10c86f274859b54fe7f3bdf991a5cf`、
`325727b9a5fdf11bc12c932a6da95f64a78e5be772bf039e391829de81d95421`、
`de8b559f16761fa60a34208d978e7f8d184ae3e808d542dbd5c2e29b5ceeb016`。三个任务
的实测 split 数与锁定值 `2525/505/5647`、`720/180/379`、
`9385/1043/1000` 一致，split leakage 为 0。上表只是工程验收，其验证
指标不具备性能结论含义。

### 正式 queue 与 GPU 所有权

实际启动命令：

```bash
P12_GPU_LIST=0,1,3,4,5 \
P12_REPO_ROOT=/DATA/DATA1/guest3/2026OpticsMoE_p12_e305e0b \
P12_POLL_SECONDS=20 P12_MAX_RETRIES=2 \
bash /DATA/DATA1/guest3/2026OpticsMoE_p12_e305e0b/experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_downstream_fa_50e.sh launch
```

- launcher/manager PID：`929693`；候选物理 GPU：`0,1,3,4,5`；common seeds：
  `2026,2027,2028`；adaptation seeds：`2026`；首轮共 18 run（9 个
  NoFT/common + 9 个 seed-2026 adaptation）。
- 01:05 启动时，GPU 0、1、4 已分别有 `guest0`/`hmh`/`lxy` 等其他用户
  compute-app；GPU 2 也被 `hmh` 占用且未列入候选。因此未强行叠加作业；
  队列首先合法使用当时唯一空闲的 GPU 3、5，并在每次派发前重新检查
  UUID/PID。候选中的占用卡释放后，队列会自动扩展，最多同时使用
  5 张卡。
- 01:06 队列快照：`running=2, pending=7, blocked=9, complete=0,
  failed=0`。blocked 为正常的 common-start 依赖，不是失败。

| Run | 物理 GPU / UUID | worker PID | 配置 digest | 01:08 中间进度 |
|---|---|---:|---|---|
| Caltech-101/NoFT/2026 | 3 / `GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd` | 929723 | `df55bfc102d76c2bbf8c24b7ff6284e4c58e310d0282f2b58e31084b7d1b584b` | 16/50，val Top-1 最佳 0.74257@15 |
| ISIC2016/NoFT/2026 | 5 / `GPU-d53ce4c8-272d-c2fb-dc09-f182d586c4eb` | 929790 | `c02bc09f171df2c70ffe09de6cd00e338aebf0ce55f81cb840ed0c310a3652248` | 2/50，val mean IoU 最佳 0.67688@2 |

上述数字都是启动健康检查的中间值，不是密封 test 结果。只有
`result.json(status=complete)`、50 行完整 history、可加载 best/last checkpoint
及全部身份哈希一致后，才会在本日志将单个 run 标记为 complete。

### 2026-08-31 01:12 +08:00 — Caltech-101/NoFT/seed_2026

- 状态：`complete`；head-only 预算完成 `50/50`，`history.json` 为 50 个完整
  epoch；无 resume/OOM/NaN，`result.status=complete`。
- 身份：训练 commit `e305e0b0b20f37757d44136c158efa92cec7f4cc`；
  config digest
  `df55bfc102d76c2bbf8c24b7ff6284e4c58e310d0282f2b58e31084b7d1b584b`；
  implementation SHA 与上述启动记录一致。
- 运行：物理 GPU 3 / `GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd`；worker
  PID `929723`；batch 96；峰值 CUDA 显存 `2.024 GiB`。
- 验证选择：epoch 38，Top-1 `0.770297`。
- sealed test normal：Top-1 `0.786435`，balanced accuracy `0.746909`；
  validation-gallery/test-query retrieval mAP `0.564418`，Recall@1 `0.658403`。
- 破坏性消融 Top-1：optical-off `0.194440`，phase-random `0.024084`，
  electronic-skip-off `0.006552`。这说明当前 NoFT 结果并非只靠临时电子头绕过
  光学骨干；不说明后续 BP/FA 排序。
- common-start SHA-256：
  `27aed054d3c6aea85b437f557a7be4d99c08712be10e9d0bf1bd22957072c9ec`；
  frozen stem state SHA-256：
  `ce6d35e62e5a497b8416e3876000f254acc427d1bfe46652411b9b42946c81279`。
- checkpoint 健康：`best.pt` 可加载且 epoch=38，`last.pt` 可加载且
  epoch=50；两者 task/method 身份均为 `caltech101/noft`。NoFT 冻结骨干，
  因此 phase 相对源模型的数值级变化约为浮点往返误差，不解释为相位学习。

## 2026-09-01：Scratch-P11-body 预训练价值控制实现

- 目的：回答 P11 ImageNet-1K 90-epoch body 预训练究竟贡献多少，而不改变
  下游任务、数据 split、临时头、优化器或 50+50 epoch 预算。
- 锁定边界：继续使用相同冻结 Qwen patch/position stem；不读取 P11 ImageNet
  backbone checkpoint。adapter、8 个光学相位面、8 个 Slim Spatial Token
  Mixer 和融合门均在 `init_seed=2026` 下全新构造。因此本控制称为“no ImageNet
  optical-body pretraining”，不能称作完全 from-scratch。
- 可复现 source：`scratch_source.py` 在隔离 RNG 域内构造 P11，通过标准
  `backbone_state_dict()` 导出，并记录 `source_regime`、`init_seed`、stem 文件
  SHA-256、完整/仅 stem/非 stem semantic tensor SHA-256、P11 架构签名和 model
  report。已存在 source 只有在这些身份全部一致时才复用，否则拒绝覆盖。
- 两阶段配置：source 序列化完成后才把它的真实文件 SHA-256 写入 resolved
  config；未提交虚假或占位 SHA。source、config 和结果统一隔离在
  `runs/p12_scratch_*` 下。
- 初始矩阵：3 tasks × downstream seed 2026 × 4 groups，共 12 runs；通过后可
  扩为 seeds `2026,2027,2028`。Scratch 控制里的 `fa_pretrained` 代码键必须在
  图表中改写为 `FA-source-init`，因为它固定的是随机 source 初相位。
- 公平比较：按 task/method/downstream seed 将主 P12 与 Scratch P12 配对。
  `noft` 给出随机特征探针，`bp` 给出同等下游预算能否从随机 body 学起；不能把
  Scratch 的较差成绩归因于 stem，因为两边 stem artifact 完全相同。
- 命令：先执行
  `commands/p12_scratch_downstream_50e.sh prepare`，再执行 `launch`；完整多 seed、
  status、tail 与 summarize 指令见
  `commands/P12_SCRATCH_DOWNSTREAM_50E_COMMANDS.md`。
- 本条仅记录本地实现和待执行方案；尚未生成服务器 source、resolved config 或
  启动正式训练，因此没有填写任何 Scratch 性能数字。
