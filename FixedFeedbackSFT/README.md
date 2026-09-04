# FixedFeedbackSFT：固定光学反馈与通用光电骨干

`FixedFeedbackSFT` 是本仓库中 fixed-feedback（文中简称 FA）课题的统一工程入口。FA 主线源码集中在 [`projects/`](projects/)；服务器命令仍使用 `experiments.<project>` Python 名称。2026-09-04 已注销全部临时 worktree，把精选 checkpoint、终态指标和 provenance 清单收拢到服务器主仓库的 `FixedFeedbackSFT/evidence/`，历史提交则由 `archive/worktree-*` Git tag 保留。

## 当前状态快照

> 以下状态在 **2026-09-01 23:36 CST** 依据进程、terminal JSON、20 条 history、日志和 SHA-256 复核。

- P13 的 8→16-stage ImageNet growth 已于 `23:24 CST` 正常完成训练、最终验证和三项破坏性消融；精选终态结果现位于主仓库 `FixedFeedbackSFT/evidence/p13_growth16_fa_source_20e_gb192/`，原 worktree 已注销。
- 最佳完整深度 checkpoint 出现在 epoch 19：Top-1 `51.428%`、Top-5 `75.752%`；epoch 20 为 `51.352%/75.762%`。同一训练器重新评估的 8-stage P11 起点为 `51.346%/75.560%`，最佳差值仅 `+0.082/+0.192 pp`。
- 16 个 phase 张量均 finite、non-zero 且有梯度，最佳模型平均绝对相位移动 `0.7567 rad`；16-stage 光学参数 `2,408,448`，电子 backbone 参数 `965,128`，可训练 backbone 的光学参数占比 `71.39%`。
- 这说明 16 层确实参与学习且可完整导出，但当前只恢复到与 8 层几乎相同的性能；在同预算 8-stage continuation 完成前，不能宣称扩深带来有效或显著性能收益。
- 新布局曾在 commit `1dedeb6d` 的临时 clean worktree 完成验收：213 项项目测试、9/9 CLI、161 个 shell 和 2 个 PowerShell 语法检查通过；P13 training checkpoint 与 backbone export 均严格加载。该 worktree 已于 2026-09-04 注销，后续只使用 `/DATA/DATA1/guest3/2026OpticsMoE` 主仓库。

精确日志、run、checkpoint、SHA-256 和精选终态 JSON 见 [`RUN_REGISTRY.md`](RUN_REGISTRY.md) 与 [`evidence/p13_growth16_fa_source_20e_gb192/`](evidence/p13_growth16_fa_source_20e_gb192/)。

## 目录结构

```text
FixedFeedbackSFT/
├── projects/                  # 9 个 FA 主线工程的物理源码位置
├── commands/                  # 专题级检查/汇总入口
├── reports/                   # 架构对比、老师汇报和可追溯图表
├── literature/                # 论文与研究背景
├── README.md                  # 当前入口
├── PROJECTS.md                # 项目编号、关系、状态和结果
├── MIGRATION.md               # 本次搬迁的兼容约束与验收流程
├── RUN_REGISTRY.md            # 服务器产物、归档 tag 与非 Git checkpoint 登记
└── paths.py                   # 不依赖父目录层数的仓库路径发现
```

九个主线工程的完整目录映射见 [`PROJECTS.md`](PROJECTS.md)。最重要的 P08–P13 关系是：

```text
P08 冻结 Qwen Patch/Position Stem + 8-stage 基线
  └─ P09 width-96 Slim Spatial Token Mixer
      ├─ P10 局部/全局双尺度光学传播
      └─ P11 token/feature 交替轴向光学传播（当前 source backbone）
          ├─ P12 分类/分割/姿态的四组 FA 下游迁移
          └─ P13 8→16→32→64→100 函数保持式扩深
```

## 路径变了，但 Python 接口不变

物理路径现在是：

```text
FixedFeedbackSFT/projects/<project_name>/
```

但公共 Python 名称继续保持：

```bash
python -m experiments.<project_name> ...
```

`experiments/__init__.py` 会把 `FixedFeedbackSFT/projects` 加入 `experiments.__path__`，所以既有 import、`python -m experiments...` 和 checkpoint 中使用的模块名不需要改变。不要同时使用 `FixedFeedbackSFT.projects.<project_name>` 导入同一份源码；两个模块名可能让 Python 把同一个 class 加载两次，破坏类型判断和 checkpoint 兼容。

命令中必须区分两类路径：

- 脚本/config 的**物理路径**：`FixedFeedbackSFT/projects/...`；
- Python 的**模块路径**：`experiments....`。

迁移代码验收已通过，但仍不要把新布局当成旧 formal run 的原地 resume。旧 run 的 config digest 与 implementation manifest 必须保持不变；新实验应以旧 checkpoint 为受检 parent，建立新的 run 身份。详细原因和检查项见 [`MIGRATION.md`](MIGRATION.md)。

## 为什么 GitHub 上找不到 runs

这不是“没有启动”的充分证据，主要有两个原因：

1. 仓库根 `.gitignore` 明确忽略 `runs`、`results`、`*.pt` 等大体积/生成产物。Git 只同步源码、配置、命令和小型报告，不会通过 `git pull` 把 checkpoint、history 或日志带到另一端。
2. 精选 P12/P13 checkpoint 位于服务器主仓库 `FixedFeedbackSFT/evidence/`；文件级 SHA-256 和原 worktree/commit 见 `evidence/worktree_cleanup_20260904/retention_manifest.json`。旧 worktree 路径仅是历史 provenance，现已不存在。

因此，两端同步分成两条通道：

- **Git 通道**：源码、配置、命令、测试、Markdown、精选 source data；
- **产物通道**：用显式路径和 checksum 单独归档日志、metrics、checkpoint，并把位置登记到 [`RUN_REGISTRY.md`](RUN_REGISTRY.md)。

不要把整个 `runs/` 强行提交 Git，也不要仅凭普通主目录里没有 `runs/` 就判断任务没有执行。

## 当前最清楚的科学主线

FA-pretrained 固定的是预训练结束时各光学层的**层间反馈算子**，不是某个样本或 batch 的误差信号。微调时：

- 前向始终使用正在更新的当前相位；
- loss 和 output error 每个 batch 重新计算；
- 本层 phase 局部导数仍使用当前输入和当前相位；
- 只有跨光学 stage 的误差连接复用 source operator；
- adapter、电子 mixer、归一化、门控和任务头仍使用普通 BP。

所以准确术语是“混合光电计算中的固定层间光学反馈”，不是“整个网络不做 BP”，也不是“混合精度计算”。

当前证据应分四层阅读：

| 证据层 | 当前代表实验 | 能支持的结论 |
|---|---|---|
| Backbone 性能 | P09/P10/P11 ImageNet | P11 在单 pretraining seed 的受控筛选中最好，Top-1 `51.348%` |
| 下游反馈 | P12 三任务、四组、3 seeds | FA-source 在分类/分割/姿态上达到 exact BP 的描述性性能水平 |
| 机制解释 | P12 梯度余弦、phase-only、P/E/H 归因 | FA-random 总分可被电子支路补偿，不能只看最终任务分数判断光学更新质量 |
| 规模工程 | P13 16/64/100-stage | 全深度计算图和反馈连接可反传；16 层已收口，但当前只与 8 层起点基本持平，尚无可归因的深度收益 |

## 推荐阅读顺序

1. [`PROJECTS.md`](PROJECTS.md)：先建立 P08–P13 与早期 V1/V2 的对应关系。
2. [`RUN_REGISTRY.md`](RUN_REGISTRY.md)：确认当前服务器任务究竟在哪个 worktree、哪些产物只在服务器。
3. [`reports/teacher_report_2026-09-01/README.md`](reports/teacher_report_2026-09-01/README.md)：论文中心思想、已有结果、创新点和图表。
4. [`METHOD.md`](METHOD.md)：fixed-feedback 的数学定义和准确边界。
5. [`MIGRATION.md`](MIGRATION.md)：修改命令、恢复 checkpoint 或同步服务器前必读。
6. [`reports/P09_P10_P11_IMAGENET_BACKBONE_COMPARISON_2026-08-29.md`](reports/P09_P10_P11_IMAGENET_BACKBONE_COMPARISON_2026-08-29.md)：三种 8-stage 架构的受控 ImageNet 对比。

历史背景仍保留在 [`HANDOFF.md`](HANDOFF.md)、[`EXPERIMENTS.md`](EXPERIMENTS.md)、[`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) 和 [`PROJECT_BRIEF_FOR_GPT56_PRO.md`](PROJECT_BRIEF_FOR_GPT56_PRO.md)。这些早期文档中的 `experiments/<FA project>` 物理路径可能是迁移前写法；新工作以本页、`PROJECTS.md` 和 `MIGRATION.md` 为准。

## 维护原则

- 新任务不再在 `2026OpticsMoE` 外创建 worktree；隔离测试结束后必须将结果归位并注销临时 worktree。
- 已完成 run 的 config、digest、implementation manifest 和精选 checkpoint 不原位改写。
- 源码迁移尽量使用 `git mv`，保留历史；同一次提交不做架构语义重命名。
- 新 run 使用所属项目的 `runs/`；旧 run 通过 `archive/worktree-*` tag、retention manifest 与 `evidence/` 复现。
- 修改后必须通过旧模块名 import、CLI、测试、shell 语法、checkpoint load 和 P13 migration/gradient 验收后，才能称为“整理完成”。
- 不使用 `git add -A` 混入用户的其他实验、数据集、传输 bundle 或生成数据。

本入口更新：2026-09-04。
