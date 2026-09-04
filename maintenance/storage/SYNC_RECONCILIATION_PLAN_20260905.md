# 本地、服务器与 GitHub 同步整理方案（2026-09-05）

本文件是只读审计后的执行方案。本轮没有删除数据集、训练结果或仍可能被正式工程引用的源码。

## 1. 当前体量

服务器 `/DATA/DATA1/guest3`：

| 区域 | 字节 | GiB | 说明 |
| --- | ---: | ---: | --- |
| `guest3` 总计 | 1,020,657,836,032 | 950.56 | 十进制约 1.02 TB |
| `2026OpticsMoE` | 990,338,469,888 | 922.32 | 唯一代码根 |
| `2026OpticsMoE/data` | 727,519,133,696 | 677.56 | 数据集为主 |
| `2026OpticsMoE/experiments` | 256,197,177,344 | 238.60 | 运行、缓存、硬件会话和交付包为主 |
| `guest3/.cache` | 27,211,149,312 | 25.34 | 主要包含模型/框架缓存，暂不清理 |

整个 `/DATA/DATA1` 分区当前约 15 TB，已用约 13 TB，可用约 750 GB，使用率 95%。这不是 `guest3` 单独造成的，但后续仍应避免重复缓存和 ZIP。

本地 `experiments` 约 4.43 GiB、77 个目录。它的主要问题是版本与归属混乱，不是磁盘容量。

## 2. 三端尚未同步的部分

### Git 历史

- GitHub `main`：`105ccd6d`。
- 本地 `main`：`a064d309`，原始提交图相对 GitHub为 ahead 35 / behind 29；按 patch 去重后，本地独有 15 个普通提交，GitHub 独有 9 个普通提交。
- 服务器 `main`：`0df93d16`，原始提交图相对 GitHub 为 ahead 22 / behind 97；按 patch 去重后仅 1 个服务器提交仍独有，服务器的主要差异实际在未提交工作区。

### 未提交文件

- 本地：625 条状态，其中 7 个已跟踪文件被修改、618 个文件未跟踪，全部位于 `experiments`。
- 服务器：238 条已跟踪/暂存状态（184 A、12 AM、8 D、34 M），另有 2,138 个未跟踪文件；196 个路径有暂存区变化，54 个路径有工作区变化。
- GitHub 当前跟踪 47 个 `experiments/*` 目录；本地文件系统有 77 个，30 个尚未进入 GitHub；服务器有 59 个，12 个尚未进入 GitHub。

### 文件级源码差异

对 `.py/.md/.yaml/.yml/.sh/.ps1/.toml/.ini/.cfg/.bat` 做 SHA256，对运行、缓存、数据、图、权重、交付包和硬件会话目录排除后：

- 本地维护型文件：2,058 个；服务器：1,890 个。
- 完全相同：1,198 个。
- 仅本地：482 个。
- 仅服务器：314 个。
- 两端同路径但内容不同：378 个。

因此不能用一次 `git pull`、`scp -r` 或覆盖式复制解决。必须按工程族合并。

详细清单：

- `experiment_sync_audit_20260905.csv`：工程存在位置、大小、Git 状态和 Python 工程间依赖。
- `maintained_code_diff_20260904.csv`：本地与服务器维护型文件的 SHA 差异。
- `local_experiments_size_20260904.csv`、`server_experiment_sizes_20260904.csv`：两端工程体量。

## 3. 为什么不能直接删“旧工程”

当前正式工程仍直接 import 早期实验目录：

- Caltech 光路由依赖 `caltech101_electronic_retrieval`、`10cm_robust`、`warmstart5` 和 `robust_hybrid_retrieval`。
- Caltech early-robust 也依赖上述目录。
- LGVQ 正式 16/36 帧工程依赖 `spatiotemporal_optical_router_vqa` 与 `lgvq_four_stage_optical_electronic_109_no_attention_vqa`。
- MNIST v2 仍依赖旧的 10 cm 单层工程。
- 部分 Caltech 数据准备代码仍复用 Grocery10 工程中的公共加载/检查点逻辑。

这些目录可被标记为“依赖库”，但在公共逻辑迁移到 `opticalmoe` 或明确的 `experiments/common` 之前，不能整目录删除或移动。

## 4. 推荐的同步顺序

### 阶段 A：建立唯一的源码真相

1. 从 GitHub `origin/main` 创建临时 `reconcile/20260905` 分支和干净工作树。
2. 保存本地 7 个 tracked 修改、618 个 untracked 文件的清单和 patch；服务器同样保存 staged、unstaged 和 untracked 清单。
3. 先合并本地 15 个 patch-unique 提交，再逐工程吸收服务器工作区中尚未提交的有效源码；不整体复制服务器目录。
4. 每个工程族单独提交，顺序如下：
   - 公共层：`opticalmoe`、`hardware_sdk`。
   - Caltech/MNIST：依赖基础 → `warmstart5/robust` → `early_robust/router` → `lab_qwen`。
   - LGVQ：`spatiotemporal/highalpha/no_attention` → `single_metric` → `lab_lgvq/timing/baseline`。
   - FA/D2NN：`FixedFeedbackSFT`、CIFAR D2NN、ImageNet patch-stem 单独成组。
   - 其他任务：Grocery、ISIC、LSP、OpenMoji、saliency 等按所有者逐组确认。
5. 每组执行 import smoke test、单元测试和关键配置解析；全部通过后合并并推送 GitHub `main`。

### 阶段 B：让本地和服务器跟随 GitHub

1. GitHub 合并完成后，为本地和服务器旧 HEAD 建安全标签；已有标签继续保留。
2. 先验证运行产物均被 `.gitignore` 正确排除。
3. 再让两端 tracked 源码对齐 `origin/main`。由于会改写当前脏工作树，这一步必须在 patch/清单验证后单独执行，不能现在直接 reset。
4. 对齐后目标：两端 `git status` 仅允许出现明确登记的本地实验配置，不允许出现未归属源码。

### 阶段 C：产物只同步索引，不追求三端物理复制

- GitHub：源码、配置、说明、指标摘要、SHA256 清单；不放数据集、大缓存、PT 和实验 ZIP。
- 服务器：数据集、可复用特征缓存、正式训练运行、正式 checkpoint 的权威副本。
- 本地：硬件配置、CCD 采集、轻量预览、当前需要交付的 ZIP；不镜像服务器 922 GiB。
- 每个正式工程补 `artifact_manifest.csv`，字段至少包含逻辑名称、相对路径、SHA256、字节数、生成配置、保留级别、权威位置和可否重建。

## 5. `experiments` 的整理策略

`experiment_sync_audit_20260905.csv` 已给出 84 个工程的初步角色。所有行的 `safe_to_delete_whole_project_now` 目前均为 `no`。

### 直接保留并优先发布

- `hardware_sdk`、`lab_qwen`、`lab_lgvq`、`qwen_optical_platform_handoff`。
- MNIST 10 cm v2。
- Caltech early-robust、光路由正式工程及其依赖闭包。
- LGVQ single-metric、no-attention、frame-count timing 及其依赖闭包。

### 保留为正式参考

- Caltech `warmstart5`。
- LGVQ high-alpha、spatiotemporal、linear baseline。

### 需要项目所有者确认

- 早期 BDD/weather、Fashion-MNIST、旧 SPAQ/KADID、8B MLP 等本地独有原型。
- 服务器独有的四个 ImageNet patch-stem 版本和三个 D2NN/FA 工程。
- LSP、OpenMoji、ISIC、saliency、Grocery 等已完成任务：优先保留源码和最终指标，清理重复 runs/cache，不直接删除源码。

## 6. 下一轮优先清理候选

以下均先生成 SHA/依赖/最优 checkpoint 清单，再删除主体：

| 候选 | 当前约占用 | 建议 |
| --- | ---: | --- |
| `data/bench2drive` | 16.63 GiB | 用户已同意删除；先确认无残余正式依赖后执行 |
| LGVQ `lab_bundles` | 24.73 GiB | 5 个当前交付包；确认师姐服务器 SHA 后，源服务器只保留清单或每种布局一个权威副本 |
| Caltech early-robust `lab_exports` | 6.19 GiB | 4 个内容重叠的旧交付 ZIP，保留最新版及 SHA |
| Caltech language2 `hardware_sessions` | 16.59 GiB | 旧单次会话占 16.56 GiB；保留 manifest、指标和代表帧后再决定 |
| Grocery10 `hardware_sessions/archive` | 2.09 GiB | 明确是 archive，可在核对正式会话后移出热存储 |
| Optical MLP ImageNet `runs` | 26.33 GiB | FA 工程；建立 all-run index，只保留正式/最优 checkpoint |
| 三组 D2NN/CIFAR `runs` | 约 21.06 GiB | 按 P/正式结果闭包保留 checkpoint，其余只留指标和配置 |
| KADID + SPAQ cache | 约 24.29 GiB | 可重建但成本较高；仅在任务结束或已有权威缓存时清理 |

LGVQ 的 `artifacts` 约 27.67 GiB，主要是 4/9/16/36 帧前端特征缓存，目前仍支持复训和重新打包，暂不建议删除。

保守估计：完成正式权重闭包和收件端 SHA 验证后，可再释放约 60–120 GiB；若删除仍可能复用的模型特征缓存，则空间会更多，但重建成本明显上升。

## 7. 执行边界

- 不把数据集、runs 或缓存提交到 Git。
- 不直接覆盖本地或服务器当前工作区。
- 不按目录名称判断“旧”并删除，必须检查反向 import 和正式 checkpoint 来源。
- 不在同步源码前清理服务器 staged/untracked 文件。
- 任何整目录删除都要先记录绝对路径、大小、文件数、SHA 清单、依赖检查和恢复来源。
