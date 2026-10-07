# 2026OpticsMoE：只从这里进入

这是本机唯一日常开发主工程。先打开本目录，不再从旁边的 `*_handoff`、
`*_refinement`、`*_a100_formal` 副本开始改代码。

## 你日常只需要看三处

1. [LightGenV2 工程入口](LightGenV2/README.md)：各项任务的代码入口和复现步骤。
2. [当前模型与性能总表](LightGenV2/PROJECT_SCORECARD.md)：仿真、实拍、微调、测速分别看，
   不把旧模型测速套给新模型。
3. [源码、权重和数据位置](LightGenV2/TASK_REGISTRY.json)：最终版本在哪台机器、对应哪个
   commit/PT/SHA；尚未迁移完整的任务有明确状态。

## 找任务

| 你要做的事 | 正式入口 |
| --- | --- |
| 物品检索 | [T01](LightGenV2/tasks/t01_object_retrieval/README.md) |
| 关键点/LSP | [T02](LightGenV2/tasks/t02_keypoint_detection/README.md) |
| 显著性/SALICON | [T03](LightGenV2/tasks/t03_saliency/README.md) |
| OpenMoji 应用：分层布局、不同大小、电子缩减及对外复现 | [T04 应用线](LightGenV2/tasks/t04_semantic_interaction/README.md) |
| OpenMoji robust 消融 | [T04](LightGenV2/tasks/t04_openmoji_robust_ablation/README.md) |
| 视频分类规划 | [T05](LightGenV2/tasks/t05_video_classification/README.md) |
| 视频质量/LGVQ | [T06](LightGenV2/tasks/t06_video_quality_assessment/README.md) |
| ABO 图搜图 | [T07](LightGenV2/tasks/t07_abo_image_retrieval/README.md) |
| ABO 图搜文、文搜图 | [T08](LightGenV2/tasks/t08_abo_image_text_retrieval/README.md) |
| 图文／音文匹配（核心源码已归主线，原资产保留） | [T09](LightGenV2/tasks/t09_multimodal_matching/README.md) |
| 专家数扩展 | [T10](LightGenV2/tasks/t10_expert_scaling/README.md) |
| 病理终身学习 | [T11](LightGenV2/tasks/t11_lifelong_optics/README.md) |
| 图文编辑/文生图 | [T12](LightGenV2/tasks/t12_text_to_image/README.md) |
| 时序鲁棒训练 | [T13](LightGenV2/tasks/t13_temporal_robust_training/README.md) |
| 多模态终身学习最终版和必要对照 | [T16](LightGenV2/tasks/t16_zero_phase_ccd_lifelong/README.md) |

同一任务的具体复现步骤以其 `reports/reproduction/README.md` 为准。未闭合的依赖不代表
可以直接换权重运行，更不代表已经在另一光路复现。

## 为什么还有其他目录

- `LightGenV2/tasks/`：长期正式任务入口。
- `experiments/`、`opticalmoe/` 等：部分任务仍依赖的旧实现和历史证据，正在逐项迁移；
  不是新的日常开发入口，不要整批删除。
- `.worktrees/`：历史/现用 Git 工作目录。运行中的版本先保留，不能为目录整齐中断实验。
- `data/`、`handoffs/`、各任务 `runs/`：原数据、交付材料、权重与实拍记录，不是重复源码。
- `archive/`：停止维护的历史副本和恢复材料，不是第二个主工程。
- 服务器 `t12_assets`、`demo_reproduction_data`、`*_runs` 等：环境、数据或结果资产，
  不应因为位于主仓库外就删除。

外部目录的实际用途与本轮处理见
[外部工程收拢记录](maintenance/storage/EXTERNAL_PROJECTS_20261004.md)。
各任务未完成的迁移见[整体待办与完成后的结构](maintenance/storage/REMAINING_CLEANUP_PLAN_20261003.md)。

## Git 的当前边界

长期只维护 `main`，不再自动创建分支、工作树或独立工程。本机这个主文件夹已经切到
`main`；历史工作目录仍保留必要版本及独有内容，现用 OpenMoji 目录继续保护。
**主线已经发布的内容不等于所有旧运行目录都切换了 main**。
2026-10-07实验室的同一主目录`E:/code/guest/2026OpticsMoE`也已实际建立main入口，
不是另建工程副本。4497份原先缺失的主线文件通过Git落地，原有设置文件保持SHA，
14项任务导航检查通过；已有ABO/OpenMoji子工程、PT、CCD和设备配置原位保留。
以上是此前的历史处理记录，不表示已登录实验室电脑上的 GitHub 账号。
按用户最新要求，实验室电脑现已排除本轮治理范围：不再连接、同步、清理或切换其代码，
也不撤销此前入口；现有实验和文件保持原状。本轮仅整理本地、Linux 训练服务器与 GitHub。
这不表示所有任务已经从主线完成设备回归，历史边界见
[实验室入口收据](maintenance/storage/LAB_ROOT_MAIN_ADOPTION_20261007.json)。
2026-10-06训练服务器主目录也已实际切到`main`，不是只更新发布引用；1011项源码／说明
落地，295项旧独有文件原位保留。现用OpenMoji和封存ABO工作目录未切换，服务器主目录
原9项历史结果／审计差异已收敛：两份说明采用服务器已保存版本，7份生成审计产物
取消Git跟踪但本地文件和恢复身份保留。两端主目录已跟踪文件均无未提交差异。
具体保留与恢复见 [审计产物身份](maintenance/storage/T06_GENERATED_AUDIT_RETENTION_20261006.json)。原切换详情见
[服务器主目录切换收据](maintenance/storage/SERVER_ROOT_MAIN_CUTOVER_20261006.json)。
不要直接切分支、pull 或整仓覆盖。整理时按 [AGENTS.md](AGENTS.md) 做逐任务迁移，
保留最终代码、有效实验数据、必要 baseline，以及全部测速/功耗证据。

如果 Git 界面还有“6k+/10k+”，先看
[分支、未跟踪产物与版本差异的区别](maintenance/git_safety/GIT_NOISE_RECONCILIATION_20261004.md)。
这些不是同一种数量，不能通过整仓覆盖或隐藏源码让它们归零。
