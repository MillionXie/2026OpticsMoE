# FA 运行产物登记表

本文件解决两个问题：

1. 长任务实际运行在哪个 server worktree；
2. 被 Git 忽略的 log、metrics 和 checkpoint 在哪里、是否已完整收口。

它是可提交的**索引**，不是产物本身。所有“运行中”状态都必须带审计时间；没有 terminal manifest、完整 metrics 和 checksum 时，不得因为 GPU 进程消失就写成“训练完成”。

## 1. P13 8→16 正式运行（已完成）

### 身份

| 字段 | 值 |
|---|---|
| 项目 | P13 `qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone` |
| run name | `p13_growth16_fa_source_20e_gb192` |
| 目的 | 从正式 P11 epoch-88 source 函数保持式增长到 16 stages，20 epoch，full-depth `fa_source` |
| 服务器 | `guest3@202.120.62.181:24096` |
| 独立 worktree | `/DATA/DATA1/guest3/2026OpticsMoE_p13_488f9d48` |
| Git 状态 | detached commit `a5628e5` |
| 审计时间 | **2026-09-01 23:36 CST** |
| 审计状态 | **completed：20/20 history、最终 normal/三项消融、backbone export 与 terminal `result.json` 均已完成** |
| 完成时间 | `2026-09-01 23:24 CST`（以 `result.json` mtime 为准） |
| config digest | `c5aa8c11d0d2ee711adaa09263ddf91cf1c81fcb56944b7e76612f1cb1775837` |
| implementation aggregate | `be935f983cc5b314a4ea587690d54251cdfae68839275b4b3ab8cac97490953a` |
| torchrun PID | `3527459`，终态审计时已正常退出 |
| worker PIDs | `3527596–3527599`，终态审计时已退出 |

PID 只用于定位该次审计，进程重启后可能变化。判断终态应使用第 1.4 节的产物判据。

### 精确路径

实验根：

```text
/DATA/DATA1/guest3/2026OpticsMoE_p13_488f9d48/
  experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone
```

日志：

```text
/DATA/DATA1/guest3/2026OpticsMoE_p13_488f9d48/
  experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/
  logs/p13_growth16_fa_source_20e_gb192.20260831T214423Z.fresh.3527417.log
```

run：

```text
/DATA/DATA1/guest3/2026OpticsMoE_p13_488f9d48/
  experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/
  runs/p13_growth16_fa_source_20e_gb192
```

关键文件相对于该 run：

```text
checkpoints/best_full_depth.pt
checkpoints/backbone_full_depth.pt
checkpoints/best_any.pt
checkpoints/last.pt
manifest.json
metrics/latest.json
metrics/history.json
result.json
```

这些路径仍是**旧 pinned worktree 的旧布局**。本次仓库整理不会、也不应在训练中改写它们。

### 终态结果

同一训练器先重新评估了 8-stage P11 source：

| 验证点 | 新层 alpha | Top-1 | Top-5 |
|---|---:|---:|---:|
| 8-stage initial re-evaluation | 不适用 | `51.346%` | `75.560%` |
| 16-stage epoch 19 | `1.0` | `51.428%` | `75.752%` |
| 16-stage epoch 20 | `1.0` | `51.352%` | `75.762%` |
| best 相对 8-stage 起点 | — | `+0.082 pp` | `+0.192 pp` |

best epoch 19 审计显示：16 个 phase tensors 均存在、finite、non-zero 且获得梯度；平均绝对 phase motion 为 `0.756711 rad`。模型含 `2,408,448` 个 phase 参数和 `965,128` 个电子 backbone 参数，可训练 backbone 的光学参数占比为 `71.39%`。这排除了“新增层完全没训练”的简单故障，但性能差值很小，只能写成“恢复并略高于同 run 的 8-stage 起点”，不能写成显著深度收益。

最终破坏性消融：

| 消融 | Top-1 | Top-5 |
|---|---:|---:|
| optical off | `0.476%` | `1.864%` |
| phase random | `0.108%` | `0.424%` |
| electronic skip off | `0.096%` | `0.436%` |

这些数值说明训练后的共适应网络同时依赖光学路径和电子 skip；它们不是纯光/纯电子模型的独立性能，不能据此计算硬件计算比例或能耗收益。

### 终态验收与 SHA-256

- [x] `history.json` 含 20 个完整 epoch，epoch 20 validation 已写入；
- [x] normal、`optical_off`、`phase_random`、`electronic_skip_off` 全部完成；
- [x] `result.json` 明确记录 `status: complete`，完整深度 backbone 已导出；
- [x] torchrun/worker 已退出，日志关键词扫描无 traceback、NCCL failure、OOM、killed 或 terminated；
- [x] 关键 JSON 已下载到 [`evidence/p13_growth16_fa_source_20e_gb192/`](evidence/p13_growth16_fa_source_20e_gb192/) 并复核 SHA-256；
- [ ] 同预算 8-stage continuation 尚未完成，因此深度收益归因仍保持开放；
- [ ] 新布局 strict checkpoint-load 审计将在 clean server worktree 建立后补登记。

| 服务器文件 | SHA-256 |
|---|---|
| `result.json` | `eeb824de2533c56688e21bef450b1410f48a3231dbc35d70fab0e4e5b2c5a360` |
| `metrics/history.json` | `5ff9ce4c65f0ae55d393546caf94187b852555f918c6aa7dec0df40b2efa688d` |
| `metrics/latest.json` | `ab541f4cf427d83bc80e4aa2bfe3f90015312180a2cc6230896c6d2b9c0766d7` |
| `manifest.json` | `5181051a8ed9ce149b61663930eccbf6405b7b4741325cb5caa6d4de7fa968d1` |
| `checkpoints/best_full_depth.pt` | `97ae579e9b33a2fc5825debde2fbad8d4fbddcbf91df5e39584036eb125ed6ec` |
| `checkpoints/best_any.pt` | `bb4600ec65ae583c0a050f52fc2302f32ec7eb22b8693aeea7efb6a3ed74f3b5` |
| `checkpoints/last.pt` | `771625a0775a253e0a74a887d116adb85c7e695ea0faa884700a7936d9aaa598` |
| `checkpoints/backbone_full_depth.pt` | `80b9b7b0f4415fd789bf46312dc23ccaf3600b5c4df9885c2972ce465dc9129d` |

## 2. 为什么普通服务器目录里没有这个 run

常用主目录是：

```text
/DATA/DATA1/guest3/2026OpticsMoE
```

P13 实际目录却是：

```text
/DATA/DATA1/guest3/2026OpticsMoE_p13_488f9d48
```

二者是不同 Git worktree。其次，根 `.gitignore` 忽略 `runs`、`results` 和 `*.pt`，所以：

- 在普通主目录 `git pull` 不会出现 P13 checkpoint；
- 本地 clone/pull 也不会出现服务器 history/log；
- `git status` 看不到 ignored run，不代表磁盘上没有；
- 应从本 registry 的绝对路径或对应 status script 查找。

## 3. 2026-09-01 服务器 worktree 清单

以下是同一次审计看到的工作目录。除 P13 外，suffix 可帮助定位历史用途，但每次复现前仍需重新核对 `git rev-parse HEAD`、dirty state 和 run manifest。

| 目录 | 用途 | 处理原则 |
|---|---|---|
| `/DATA/DATA1/guest3/2026OpticsMoE` | 旧服务器主工作树 | 已分叉且脏；不可直接 pull/reset/rebase |
| `/DATA/DATA1/guest3/2026OpticsMoE_p12_e305e0b` | P12 正式下游任务 | 保留原 commit 与原 runs；只读登记 |
| `/DATA/DATA1/guest3/2026OpticsMoE_p12_mechanism_e305` | P12 机制/P-E-H 审计 | 保留机制产物 provenance |
| `/DATA/DATA1/guest3/2026OpticsMoE_p12_phase_only_e7ae69e7` | P12 phase-only 面板 | 保留冻结电子 body 协议与结果 |
| `/DATA/DATA1/guest3/2026OpticsMoE_p13_488f9d48` | 已完成的 P13 8→16 growth | 保留 pinned commit 与原 runs；不得用新布局原地 resume |
| `/DATA/DATA1/guest3/2026OpticsMoE_report_646f847e` | 老师汇报/图表生成 | 报告源码已纳入 `FixedFeedbackSFT/reports`；服务器产物按 checksum 登记 |
| `/DATA/DATA1/guest3/2026OpticsMoE_scratch_2981fe23` | scratch / No-ImageNet body 等隔离实验 | 不与正式 P12 runs 混合 |

### 旧主工作树 Git 状态

审计快照：

```text
path:        /DATA/DATA1/guest3/2026OpticsMoE
local main:  c84fca0
origin/main: 3e639199
ahead/behind: 11 / 48
worktree:    modified and untracked files present
```

因此同步新布局时应建立 clean worktree。不能对这个目录直接 `git pull`，也不能用 `git reset --hard` 或 `git clean` 消除分叉；这些动作可能删除服务器独有提交和用户文件。

## 4. 旧结果与新结果的统一登记策略

### 旧 run

- 物理文件继续留在其 pinned worktree 的 `experiments/<project>/runs/...`；
- 旧 config、digest、implementation manifest 与 checkpoint 不改；
- registry 记录绝对路径、Git SHA 和关键 SHA-256；
- 如需复制，只创建已校验的只读归档，不删除原件。

### 新 run

源码、配置和命令位于：

```text
FixedFeedbackSFT/projects/<project>/
```

生成产物统一位于：

```text
FixedFeedbackSFT/runs/<project>/<run>/
```

也可用 `FIXED_FEEDBACK_RUNS_ROOT` 将 `FixedFeedbackSFT/runs` 映射到服务器大容量盘。该目录仍被 Git 忽略；本 registry 才是进入 Git 的索引。

新布局不得 resume 旧布局 run。迁移旧权重时，应启动新 run 并记录 `parent_checkpoint_path`、`parent_checkpoint_sha256`、旧 Git SHA、新 Git SHA、旧/新 config digest 和 migration report。

## 5. 新增 run 的登记模板

复制下面一节并填写，禁止省略审计时间或把 unknown 写成成功：

```markdown
### <project>/<run_name>

| 字段 | 值 |
|---|---|
| scientific purpose | |
| status | planned / running / closure pending / completed / failed |
| last audited at | YYYY-MM-DD HH:MM TZ |
| server/worktree | |
| Git SHA + dirty flag | |
| config path + digest | |
| command file | |
| GPUs / world size / global batch | |
| log path | |
| run path | |
| terminal metrics path | |
| best checkpoint + SHA-256 | |
| last checkpoint + SHA-256 | |
| parent checkpoint + SHA-256 | |
| result summary | |
| limitations / next action | |
```

## 6. 最小只读核验

服务器核验时至少检查四类证据：

```bash
git -C <worktree> rev-parse HEAD
git -C <worktree> status --short
ps -fp <launcher-or-torchrun-pid>
tail -n 80 <absolute-log-path>
```

然后读取 `metrics/latest.json` 与 `metrics/history.json`，核对 checkpoint 内的 format、depth、epoch、config digest 和 implementation manifest。`nvidia-smi` 只能说明 GPU 是否忙，不能证明特定程序完成了正确训练。

## 7. 待补登记

- [x] P13 epoch 20、最终消融终态与关键 SHA-256；
- [ ] P11 source `backbone.pt` 与 epoch-88 `best.pt` 的绝对路径/SHA-256；
- [ ] P12 formal、No-ImageNet body、phase-only 和 mechanism 的 run roots 与 terminal summaries；
- [ ] 新布局 clean server worktree 的路径与部署 commit；
- [ ] 需要同步到本地的精选 checkpoint 清单及本地校验结果。

最后更新：2026-09-01；P13 终态审计时间：2026-09-01 23:36 CST。
