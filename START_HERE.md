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

## 另外保留的历史研究与 baseline

这些不是新增工程，也不替代上表的最终模型：

- [EuroSAT RGB/SAR 及光学分类历史对照](LightGenV2/demo_check/README.md)：纯光学、输出融合、共享冻结前端三种协议分别保留；不要混用成绩。
- [Adrenal 病理结构/OEO 历史源码](LightGenV2/demo_check/adrenal_softsign_code_export_20260915_145336/MAIN_ENTRY.md)：另见[深度/OEO核查](LightGenV2/demo_check/reports/reproduction/ADRENAL_DEPTH_REPRODUCTION_20260916.md)及[固定权重阈值诊断](LightGenV2/demo_check/reports/reproduction/ADRENAL_THRESHOLD_AUDIT_20260916.md)。原导出和必要结构对照不是 T11/T16 的另一套最终版；旧路径与训练命令不能直接用于现在的机器。
- [六任务 baseline 代码交接](LightGenV2/reports/20260915_baseline_methods/CODE_HANDOFF.md)：原版本、原包与恢复方式；不是用最新模型替换旧对照。

原数据、权重、结果及全部测速仍保留，以上链接不表示已重跑或已在空机器复现。

## 为什么还有其他目录

- `LightGenV2/tasks/`：长期正式任务入口。
- `experiments/`、`opticalmoe/` 等：部分任务仍依赖的旧实现和历史证据，正在逐项迁移；
  不是新的日常开发入口，不要整批删除。
- `.worktrees/`：历史/现用 Git 工作目录。运行中的版本先保留，不能为目录整齐中断实验。
- `data/`、`handoffs/`、各任务 `runs/`：原数据、交付材料、权重与实拍记录，不是重复源码。
- `archive/`：停止维护的历史副本和恢复材料，不是第二个主工程。
  旧打包预览、Excel核验临时目录及传输字节码已整体归入
  `archive/maintenance_scratch_20261008/`，原文件未删，不再散在根目录。
- 旧 `ABO_Lab_8um` 和 `ABO_Lab_SHS_8um` 本机工程已完整归入
  `archive/legacy_lab_projects/`，原文件、SDK、标定、CCD及全部测速未删除。
  旧硬件入口若需要这些资产，须显式指定归档路径或按收据恢复原目录；旧命令不自动改写。
  两者都不是当前rank72最终图搜图入口，详见[用途与归档收据](maintenance/storage/LOCAL_LEGACY_LAB_PROJECTS_20261008.json)。
  Linux主目录中的旧DVP导入工程也已独立完整归档至同名历史区：6,557文件，约17.76 GB；
  包含旧权重、数据和测速，未删除，不与本机较小的归档混为一份。
- 服务器 `t12_assets`、`demo_reproduction_data`、`*_runs` 等：环境、数据或结果资产，
  不应因为位于主仓库外就删除。

外部目录的实际用途与本轮处理见
[外部工程收拢记录](maintenance/storage/EXTERNAL_PROJECTS_20261004.md)。
各任务未完成的迁移见[整体待办与完成后的结构](maintenance/storage/REMAINING_CLEANUP_PLAN_20261003.md)。

## Git 的当前边界

本地这个主文件夹与 Linux 的 `/DATA/DATA1/guest3/2026OpticsMoE` 都使用 `main`，
源码经 GitHub 同步。以后只从这两个主目录开发，不再新增分支、工作树或工程副本。
现用 OpenMoji 的未提交改动和封存 ABO 运行目录继续保护；主线同步不等于这些目录已切换。
本轮**不连接、不整理实验室电脑**，也不恢复对外上传。

原数据、权重、CCD、交付包和全部测速不靠 Git 同步，原位置及 SHA 从任务登记查找。
旧目录的归档恢复位置、必须保留的依赖和未解决项，统一看
[当前待办与例外](maintenance/storage/REMAINING_CLEANUP_PLAN_20261003.md)，不把历史盘点数字当实时状态。
服务器切换经过保存在[原切换收据](maintenance/storage/SERVER_ROOT_MAIN_CUTOVER_20261006.json)；
日常使用不需要逐份阅读历史审计报告。

存在未提交修改时不要直接切分支、pull 或整仓覆盖，按 [AGENTS.md](AGENTS.md) 保留并逐项处理。

如果 Git 界面还有“6k+/10k+”，先看
[分支、未跟踪产物与版本差异的区别](maintenance/git_safety/GIT_NOISE_RECONCILIATION_20261004.md)。
这些不是同一种数量，不能通过整仓覆盖或隐藏源码让它们归零。

## 当前可以验收什么，哪些还不能算完成

2026-10-08对已发布主线 `843d252f7cdc5f59c32d5179b985cc5f73ce14e7`
及本机材料分别检查：14项登记入口、导航和导入源码身份没有检查错误；
本机登记PT逐SHA核验、私有报告和恢复引用检查也没有错误。
这证明入口及登记身份可查，不证明每个任务可以在空机器上独立复现。

| 范围 | 当前结论 |
| --- | --- |
| 日常开发入口 | 本地和Linux主目录使用main；不再从历史副本开发 |
| ABO最终图搜图 | rank72最终身份已经确定；不再扩充历史试错源码作为收尾前置条件 |
| OpenMoji | 应用线与robust线分开；当前运行目录及未提交修改仍受保护，不为统一目录强行切换 |
| 原资产和测速 | 保留原位置、恢复材料和身份；不是全部已复制到本机或GitHub |
| Git待提交材料 | 最后一批27份说明、协议、结果与统计已纳入Git；两份旧统计表备份完整归档，现用窗口以后产生的修改按任务继续提交 |
| 科学复现例外 | T01历史前端版本、T02资产闭包、T08适配资产、T12环境资产仍未全部闭合；T05尚未开展 |

实验室电脑不在本轮整理范围，实验室切换不作为本轮完成条件。
新clone仍需另行取得三份私有交付材料：ABO rank72实拍报告、T12 17M交付README及报告；
源码仓库检查不会把缺少私有数据误称为“全部复现完成”。
完整旧科学材料继续保留，不需要为降低Git数字再训练、重拍或重写历史结果。
