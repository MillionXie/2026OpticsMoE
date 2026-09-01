# FA 工程迁移与双端同步说明

## 1. 本次整理解决什么

旧仓库把 FA 主线、其他光学实验、生成产物和临时报告都堆在根 `experiments/`，导致项目边界、run 位置和命令入口难以辨认。本次整理把九个 FA 工程物理移动到：

```text
FixedFeedbackSFT/projects/<same_project_basename>/
```

同时把架构比较和老师汇报集中到 `FixedFeedbackSFT/reports/`。目录 basename 和 Python 公共名称不变；本次提交不趁机重命名模型、不改变 FA 数学定义，也不搬动运行中的服务器 worktree。

这是一次**带兼容层的物理布局迁移**，不是简单剪切文件。只有完成第 9 节的验收后，才可以把它称为整理完成。

## 2. 迁移映射

以下九个目录从 `experiments/` 移到 `FixedFeedbackSFT/projects/`，basename 原样保留：

1. `d2nn_cifar100c10_fixed_feedback_20stage400`
2. `d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400`
3. `d2nn_cifar10_high_performance_optical_backbone`
4. `qwen3_vl_patch_stem_8stage_optical_imagenet_backbone`（P08）
5. `qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone`（P09）
6. `qwen3_vl_patch_stem_8stage_dual_scale_optical_imagenet_backbone`（P10）
7. `qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone`（P11）
8. `qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa`（P12）
9. `qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone`（P13）

报告迁移：

- `experiments/P09_P10_P11_IMAGENET_BACKBONE_COMPARISON_2026-08-29.md` → `FixedFeedbackSFT/reports/`；
- `document/19_fixed_feedback_optical_backbone_teacher_report/` → `FixedFeedbackSFT/reports/teacher_report_2026-09-01/`。

共享的 ImageNet、Caltech、ISIC、LSP 和稠密任务基础设施仍保留在根 `experiments/`。依赖清单见 [`PROJECTS.md`](PROJECTS.md)。

## 3. 稳定接口：物理路径变，模块名不变

旧接口必须继续成立：

```python
from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone.model import ...
```

```bash
python -m experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa ...
```

实现方式是让 `experiments/__init__.py` 把 `FixedFeedbackSFT/projects` 追加到 `experiments.__path__`。这保留了：

- 既有 `experiments.<project>` import；
- 既有 `python -m experiments.<project>` CLI；
- checkpoint/provenance 中记录的模块身份；
- P09/P10/P11/P12/P13 之间的相互 import。

禁止把同一个工程同时以 `experiments.<project>` 和 `FixedFeedbackSFT.projects.<project>` 导入。后者不应成为公共 package；否则同一源码可能生成两套 module/class identity，使 `isinstance`、pickle 和 strict checkpoint load 出现隐蔽错误。

命令应写成：

```text
物理脚本/config：FixedFeedbackSFT/projects/<project>/...
Python 模块：     experiments.<project>...
```

## 4. 固定父目录层数必须消失

移动后，`Path(__file__).parents[1]` 或 `parents[2]` 不再可靠，服务器 detached worktree 也可能有不同目录名。FA 工程应通过 `FixedFeedbackSFT.paths` 查找仓库根、project root 和 run root，不得继续按“向上两层就是仓库”猜测。

仓库根的判据应同时看到 `.git`、`experiments/` 和 `FixedFeedbackSFT/`。测试临时目录或 exported bundle 若不满足该判据，应显式传入 root，而不是静默落到错误路径。

## 5. 配置摘要与旧 run：最关键的兼容边界

本次迁移已经对活动 `configs/*.yaml` 做了 **68 个文件的机械路径更新**：源码、配置和资产位置改为 `FixedFeedbackSFT/projects/<FA project>/...`，运行输出改为 `FixedFeedbackSFT/runs/<FA project>/...`。这是必要的新布局修改，但会改变配置文件字节及其 digest。

因此必须明确区分：

### 旧布局正式运行

- 旧配置精确字节保存在 Git commit `3e639199e340b5ae0e2924e34b71ca86d9f230a4` 以及对应 pinned server worktree；
- 旧 run 的 resolved config、config digest、implementation manifest 和 checkpoint 继续互相绑定；
- **不得**拿新布局下已改路径的 YAML 在旧 run 目录中原地 resume；
- P13 尤其会校验 implementation manifest，路径或文件摘要变化都应使错误 resume 被拒绝。

### 新布局运行

- 使用 `FixedFeedbackSFT/projects/.../configs/...` 的新字节和新 digest；
- 在全新的 output/run directory 启动，生成新的 resolved config 与 manifest；
- 如果从旧 checkpoint 继承权重，应把它声明为 migration/source parent，并记录 source SHA-256，而不是伪装成同一次 run 的 resume；
- P13 16-stage 正式 run 已于 2026-09-01 23:24 CST 完成；其 `best_full_depth.pt` 可作为后续新布局实验的候选 parent，但须先通过 strict compatibility test，并以新 run 身份记录旧/新 implementation manifest。

换句话说：**权重可以经过受检迁移成为新 run 的 source，旧 run 身份不能跨布局被重写。**

## 6. 为什么当时不用等训练结束才整理，也没有跟着搬

P13 当时运行在独立 worktree：

```text
/DATA/DATA1/guest3/2026OpticsMoE_p13_488f9d48
```

进程当时已经打开旧路径下的代码、日志和 checkpoint 文件，且 run manifest 锁定了当时的实现。对另一个 Git worktree 做 `git mv` 不会影响它，因此整理和训练可以并行；反过来，在活动 worktree 内 pull/rebase/切 commit/移动目录会破坏可复现性，甚至让后续 checkpoint 来自混合代码版本。该 run 现已完成并按原路径保留。

正确顺序是：

1. 让旧 pinned worktree 按原 config 完成训练和最终消融；
2. 验证 terminal result、history、checkpoint 和日志均完整；
3. 计算关键文件 SHA-256，更新 [`RUN_REGISTRY.md`](RUN_REGISTRY.md)；
4. 只读归档或复制产物，不改原 run；
5. 后续实验从整理后的新布局开新 run。

所以不需要等训练完才整理源码，但必须等训练完整收口后才归档/搬复制它的产物。

## 7. Git 同步与产物同步是两件事

根 `.gitignore` 忽略：

- `runs`
- `results`
- `figures`（已跟踪文件仍会继续跟踪）
- `*.pt`
- 数据与缓存目录

这使 checkpoint 和训练日志不会被误提交，但也意味着 `git push/pull` 永远不会自动把它们从服务器带到本地。

### Git 通道

应提交：源码、配置、命令、测试、Markdown、精选小型 source data、run registry。

不应提交：完整 checkpoint、重复日志、dataset、cache、传输 bundle 和大规模生成产物。不要用 `git add -A`，因为本地主工作树还有与 FA 无关的 OpenMoji、multiplane、grocery 等未跟踪/修改内容。

### 产物通道

旧产物留在原 worktree；新 run 推荐统一落到：

```text
FixedFeedbackSFT/runs/<project>/<run_name>/
```

或通过 `FIXED_FEEDBACK_RUNS_ROOT` 指向服务器大容量盘。无论实际根目录是什么，都应在 registry 记录绝对路径、Git SHA、config digest、关键 checkpoint SHA、状态和最后审计时间。

两端同步 checkpoint 时应采用“先复制到临时目标 → 校验文件大小/SHA-256 → 原子改名”的方式，避免把仍在写入的 `last.pt` 当成完整文件。正在运行的 `last.pt` 不应跨端复制；优先同步已经关闭写入的 `best*.pt` 和 terminal metrics。

## 8. 服务器旧主树不能直接 pull

2026-09-01 约 23:05 CST 的审计显示：

```text
worktree:     /DATA/DATA1/guest3/2026OpticsMoE
local main:   c84fca0
origin/main:  3e639199
relationship: ahead 11 / behind 48
state:        modified + untracked files
```

这是一个已经分叉且脏的工作树。直接 `git pull` 可能产生大规模冲突；`git reset --hard` 或覆盖式同步会丢失服务器本地代码/记录。因此新布局应部署到新的 clean worktree，或先把旧主树的 11 个本地 commit 和未提交文件做只读审计、分支化保存，再由人工决定如何合并。

在完成该审计前：

- 不在旧主树运行 `pull --rebase`；
- 不 reset、clean 或 checkout 覆盖用户文件；
- 不把旧主树当成 P13 实际运行目录；
- 不用旧主树是否存在 `runs/` 判断任务状态。

## 9. 迁移验收清单

以下项目全部通过后，才允许合并整理分支并启动 new-layout formal run：

- [ ] 九个工程均能通过旧名称 `import experiments.<project>` 导入；
- [ ] 关键 class 只对应一个 module identity，不存在双命名加载；
- [ ] V1/V2/高性能骨干/P08–P13 的测试套件通过；
- [ ] `python -m experiments.<project>` 的 smoke/CLI 仍可启动；
- [ ] 所有 shell 脚本通过 `bash -n`，PowerShell 脚本通过 parse；
- [ ] 活动命令的物理路径全部指向 `FixedFeedbackSFT/projects/...`；
- [ ] 全仓扫描后，仅历史报告、旧 provenance 或明确 legacy 示例保留旧物理路径；
- [ ] P09/P10/P11 的 strict checkpoint load 和架构签名拒错行为不变；
- [ ] P12 的 source checkpoint、dataset manifest 和四组公平性检查通过；
- [ ] P13 P11→16 migration、alpha=0 等价、full-depth feedback 和 phase gradient 测试通过；
- [ ] 旧 `3e639199` 配置可在 pinned worktree 原样读取，新布局 config digest 明确不同；
- [ ] 新 run 输出根可写且被 Git 忽略，registry 模板可完整填写；
- [ ] `git diff --check` 无空白错误，提交只包含 FA 整理范围。

## 10. 回滚与故障定位

若新布局测试失败，不要改旧服务器 run 来“适配”新代码。先按以下层次定位：

1. import 失败：检查 `experiments.__path__` 与 `FixedFeedbackSFT/projects`；
2. 文件找不到：检查是否仍用固定 `parents[n]` 或旧物理路径；
3. digest/manifest 不匹配：确认正在做 new run 还是错误地 resume old run；
4. class/checkpoint 类型异常：检查是否双模块名导入；
5. 结果目录为空：先查 [`RUN_REGISTRY.md`](RUN_REGISTRY.md) 的 worktree 和 ignore 策略；
6. 服务器无法同步：不要在脏旧主树强行 pull，改用 clean worktree。

迁移的安全回滚点是旧 commit `3e639199` 与各自 pinned worktree；它们保留旧物理路径和正式运行身份。本次整理不应删除这些 worktree 或原始 run。
