# FA 运行产物登记表

本文件解决两个问题：

1. 长任务实际运行在哪个服务器项目目录或历史 Git tag；
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
| 历史源码身份 | Git tag `archive/worktree-p13-20260904`（原 worktree 已注销） |
| 当前精选产物 | `/DATA/DATA1/guest3/2026OpticsMoE/FixedFeedbackSFT/evidence/p13_growth16_fa_source_20e_gb192` |
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

原始实验根（历史路径，2026-09-04 后已不存在）：

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

这些是审计时记录的**历史路径**。精选 checkpoint 和终态 JSON 已按 SHA-256 迁至主仓库 `FixedFeedbackSFT/evidence/`；完整来源映射见 `evidence/worktree_cleanup_20260904/retention_manifest.json`。

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
- [x] 2026-09-02 在新布局 commit `1dedeb6d` 实际重建 16-stage 模型：training checkpoint 严格加载为 0 missing/0 unexpected；backbone export 仅缺预期的临时 ImageNet readout 7 个键，0 unexpected；两者 depth-alpha 与静态 stem SHA 均匹配。

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

P11 source 也已通过 PyTorch 只读加载、finite/non-empty tensor 与 SHA-256 审计：

| P11 文件 | epoch | SHA-256 |
|---|---:|---|
| `checkpoints/backbone.pt` | 88 | `c3ad0b780dfbb3e5f8e1f7b7850c06fcb5c6d977e106f351b4602fcaadf210d2` |
| `checkpoints/best.pt` | 88 | `a30d5c06b61a635bb3dc379aeaca4c371c1d27e6b862c5ffd4977ce738b33034` |

## 2. 2026-09-04 归档后的查找方式

常用主目录是：

```text
/DATA/DATA1/guest3/2026OpticsMoE
```

P13 的当前精选产物目录是：

```text
/DATA/DATA1/guest3/2026OpticsMoE/FixedFeedbackSFT/evidence/p13_growth16_fa_source_20e_gb192
```

原 worktree 已注销，源码历史由 `archive/worktree-p13-20260904` 保存。根 `.gitignore` 忽略 `runs`、`results` 和 `*.pt`，所以：

- 在普通主目录 `git pull` 不会出现 P13 checkpoint；
- 本地 clone/pull 也不会出现服务器 history/log；
- `git status` 看不到 ignored run，不代表磁盘上没有；
- 应从本 registry、`FixedFeedbackSFT/evidence/` 和 retention manifest 查找。

## 3. 历史 worktree 归档清单

以下目录曾用于隔离实验，已于 2026-09-04 完成精选产物迁移并注销。不要再把这些路径写入新命令。

| 原用途 | 当前来源身份 | 当前产物位置 |
|---|---|---|
| P12 正式下游 | `archive/worktree-p12-formal-20260904` | `FixedFeedbackSFT/evidence/p12_downstream_fa_50e/` |
| P12 机制/P-E-H | `archive/worktree-p12-mechanism-20260904` | `FixedFeedbackSFT/evidence/p12_downstream_fa_50e_mechanism/` |
| P12 phase-only | `archive/worktree-p12-phase-only-20260904` | `FixedFeedbackSFT/evidence/p12_phase_only_fa_50e/` |
| P12 scratch/control | `archive/worktree-p12-scratch-20260904` | `FixedFeedbackSFT/evidence/p12_scratch_control/` |
| P13 8→16 growth | `archive/worktree-p13-20260904` | `FixedFeedbackSFT/evidence/p13_growth16_fa_source_20e_gb192/` |
| 老师报告 | `archive/worktree-report-20260904` | `FixedFeedbackSFT/reports/` |
| 新布局验收 | `archive/worktree-fa-reorg-20260904` | `FixedFeedbackSFT/evidence/fa_reorg_runs_20260904/` |

### 旧主工作树 Git 状态

审计快照：

```text
path:        /DATA/DATA1/guest3/2026OpticsMoE
local main:  c84fca0
origin/main: 3e639199
ahead/behind: 11 / 48
worktree:    modified and untracked files present
```

以上是历史快照。当前仍不能在脏主仓库上直接执行破坏性的 `reset --hard` 或 `clean`；同步前应先检查状态并只提交/迁移明确属于本任务的文件。新隔离工作如确有必要，可在仓库内部建立临时 worktree，验收后必须归位产物并注销。

### 新布局临时 clean worktree 验收（已注销）

2026-09-02 的部署与验证结果：

```text
path:        /DATA/DATA1/guest3/2026OpticsMoE/FixedFeedbackSFT/runs/_worktrees/fa_reorg
branch:      codex/fa-reorg
validated:   1dedeb6d397f5d9cfdef8e4ef72d886e421366a0
worktree:    clean（仅有 Git 忽略的 run/stem symlink）
layout:      3 passed
projects:    213 passed（pytest --import-mode=importlib）
CLI:         9/9 --help entrypoints passed
shell:       161 bash scripts passed bash -n
PowerShell:  2 scripts parsed with 0 errors
P13 load:    best_full_depth.pt and backbone_full_depth.pt compatible
```

目录边界：新部署必须位于 `/DATA/DATA1/guest3/2026OpticsMoE/` 内。曾误建的同级目录、7 个历史 worktree 和项目内部验收 worktree 均已在 2026-09-04 注销；`.codex_transfer` 仅作临时传输并已清空。历史身份由归档 tag 和 retention manifest 提供。

不同工程中同名的 `test_core.py` 会被 pytest 默认导入模式视为同一顶层模块，因此全套测试使用 `--import-mode=importlib`；逐工程执行不受影响。这是测试收集命名问题，不是源码导入冲突。

## 4. 旧结果与新结果的统一登记策略

### 旧 run

- 精选物理文件位于主仓库 `FixedFeedbackSFT/evidence/...`；
- 旧 config、digest、implementation manifest 与 checkpoint 不改；
- registry 和 retention manifest 记录旧路径、归档 tag、Git SHA 与关键 SHA-256；
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

### 当前统一入口

旧符号链接随临时 worktree 一并注销；当前统一入口如下：

| 中央 run 名 | 原始物理位置 |
|---|---|
| P11/P08 selected assets | `FixedFeedbackSFT/evidence/fa_reorg_runs_20260904/selected_checkpoints/` |
| P12 `p12_downstream_fa_50e` | `FixedFeedbackSFT/evidence/p12_downstream_fa_50e/` |
| P12 mechanism | `FixedFeedbackSFT/evidence/p12_downstream_fa_50e_mechanism/` |
| P12 phase-only | `FixedFeedbackSFT/evidence/p12_phase_only_fa_50e/` |
| P12 scratch/control | `FixedFeedbackSFT/evidence/p12_scratch_control/` |
| P13 `p13_growth16_fa_source_20e_gb192` | `FixedFeedbackSFT/evidence/p13_growth16_fa_source_20e_gb192/` |

静态 stem 的规范路径是 `FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/assets/qwen3_vl_static_stem_224.pt`；SHA-256 为 `e3b12b274211d29f928eee95fdfc60b32d10751f1bbdc98cd63f0cccd0792485`。

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
- [x] P11 source `backbone.pt` 与 epoch-88 `best.pt` 的绝对路径/SHA-256；
- [x] P12 formal、No-ImageNet body、phase-only 和 mechanism 的 run roots 已接入统一入口；终态指标见 `PROJECTS.md` 与老师报告；
- [x] 新布局 clean server worktree 的路径、部署 commit、测试和 strict checkpoint load；
- [ ] 需要同步到本地的精选 checkpoint 清单及本地校验结果。

最后更新：2026-09-02；P13 终态审计时间：2026-09-01 23:36 CST；新布局验收时间：2026-09-02 CST。
