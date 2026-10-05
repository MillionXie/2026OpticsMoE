# 历史工作目录原位冻结：分支名 15 → 12

## 最新覆盖：10 → 9，T12 图文编辑审计历史目录

`.worktrees/t12_audited_editors` 已原位固定于
`a3f5e6fe7027e92fe29be2934750f427cdfd91e7`，取消旧开发分支名
`codex/t12-audited-editors-20260926`。这是图文编辑的历史审计／baseline源码，
不是当前服务器最终训练入口；服务器同身份目录此前已按相同保护方式冻结。
本轮4779个内容文件前后SHA一致，无跟踪／未跟踪修改及进程命令引用。
全部历史代码、baseline、数据、结果、缓存和测速原位保留；不声明所有历史入口
已经迁入main，也不将旧测速套给最终模型。

恢复引用：`refs/archive/frozen-local-20261005/t12-audited-editors-20260926`。
完整Git包verify通过，SHA256为
`8993d128fae4bebdfc77b4bc5b6e1a8a1355623dd5988fd9e08274256090ee17`；
私有收据在 `.codex_tmp/t12_audited_frozen_identity_20261005`。
主目录HEAD、索引中的用户修改未改变，14个工作树仍保留，不计新增空间释放。

## 最新覆盖：11 → 10，T13 时序鲁棒历史目录

`.worktrees/t13_temporal_robust_20260927` 原位冻结于
`493eb38375f3bd9e4e8e7edfe278468b351365f1`；取消开发分支名
`codex/t13-temporal-robust-20260927`。3400个内容文件操作前后SHA相同，
无跟踪修改、无未跟踪文件、未发现进程命令引用；完整bundle已verify。
恢复引用为 `refs/archive/frozen-local-20261005/t13-temporal-robust-20260927`，
bundle SHA为 `f4644eb193e965de2ab1bd04bbf74cecfb152ca07fda78c61558f84eb3b24bdc`，
私有清单／收据／包位于 `.codex_tmp/t13_history_frozen_identity_20261005`。

此处保留一项明确例外：`public_simulation/` 是重新组织的独立仿真导出，
不是正式训练运行时；27项源码仍留在冻结目录和恢复历史中，未覆盖 main。
主线已有正式运行源码、教师依赖及结果说明。私有连接脚本也未公开纳入 main。
本次只退出旧开发线，不宣称导出包已完成完整训练复现。
源码、数据、PT、报告、全部测速及缓存均未移动或删除，14个工作树仍在。

## 后续覆盖：12 → 11，T02 个人照片历史工程

`.worktrees/t02_personal_pose` 已在原提交
`2fa901dc7800d8798658b29e242e38bebac97989` 原位冻结，不再维护
`codex/t02-personal-pose-20260928` 开发分支。用途是个人照片域迁移／伪标签审核／
逐样本图例交付，不是官方 LSP 真值评估，也不是 OpenMoji。该任务源码此前已归 main，
与 main 的任务目录差异只剩说明文件；本轮不再重复导入模型或运行实验。

冻结前核验无跟踪修改、无未跟踪文件、未发现进程命令引用；3619个内容文件前后
SHA完全相同，主目录HEAD和跟踪状态不变。所有数据、图例、测速、缓存和源码原位
保留，14个注册工作树没有减少，不声称释放空间。完整恢复历史在
`refs/archive/frozen-local-20261005/t02-personal-pose-20260928`，独立 Git bundle
已 verify；SHA256为 `79e9e9107976da0ce7e387d79494cf09067a111f63dbe45c45d1b662d8283694`。
逐文件清单、收据和恢复包在私有 `.codex_tmp/t02_personal_frozen_identity_20261005`。

以下三目录／15→12数字是之前一批记录，仍保留追溯。

2026-10-05，三套无跟踪修改、无未跟踪文件、未发现进程命令引用的本地历史目录，
已改为在原 HEAD 上 detached 的冻结 checkout。没有移动或删除目录、源码、数据、
权重、测速或缓存，没有合并科学实现，也没有切换主工作目录。

| 原位置（主仓库相对路径） | 用途 | 原提交／冻结提交 | 核对文件数 |
| --- | --- | --- | ---: |
| `.codex_tmp/mnist_publish` | 旧 MNIST／DVP ROI 和光学失配诊断发布副本 | `52db52d19dfd35382e62d1cc9bda3373be5cff81` | 4751 |
| `.worktrees/t11_joint_d2nn` | 病理终身学习的联合 D2NN／历史协议对照源码 | `8b6b8398520915730600695d730a5ca983a7e842` | 4910 |
| `.worktrees/t12_d2nn_baselines` | 旧编号 T12 的四模态终身学习 D2NN 对照；不是图文编辑 T12 | `5d1ee1174ba445fb807380f7dd0c28c03a592f9d` | 5062 |

三目录全部 14723 个内容文件在操作前后逐文件 SHA256 完全相同；`.git` 元数据指针
单独排除，不把 checkout 身份变动算成科学文件变动。原主目录 HEAD、跟踪状态未变；
原18项用户修改另行 SHA 核对，零变化。14个注册工作树仍保留。

已取消的日常分支名分别为 `codex/mnist-dvp-20260924`、
`codex/t11-joint-d2nn-baseline`、`codex/t12-d2nn-baselines`。
完整历史仍通过对应 `refs/archive/frozen-local-20261005/<分支末段>` 可达，
并有独立验证通过的 Git bundle。恢复包在私有 `.codex_tmp/<case>_frozen_identity_20261005`：

| case | bundle SHA256 |
| --- | --- |
| mnist | `27ff3fc965ce8a81c76463ce347dfd5662409087a6ce809b835eacf2b574b641` |
| t11_baseline | `1587c292f3cfd34c5964338fc41136bbc60e5e92e5b11a68603a9a3611e4808e` |
| t12_baseline | `9c9fcb560c93569be9199d636331469e80ea9f177a64993f8b39956add7bdf21` |

每个私有目录都有 `file_identity.json` 和 `receipt.json`；Git bundle 不公开提交。
这是减少开发分支，不是释放空间。baseline 与测速仍可按原路径和原提交读取。
后续长期开发入口是已发布 main；冻结目录不能被当作最新主工程。
OpenMoji 现用树、ABO 封存工程和服务器运行 checkout 均未改变。

如确需恢复某个旧分支名，先检查无同名分支并取得用户指示，再从上述 archive ref
恢复；不应在日常新任务中重新启用这些历史开发线。
