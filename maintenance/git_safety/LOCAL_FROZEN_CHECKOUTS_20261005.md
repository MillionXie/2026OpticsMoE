# 历史工作目录原位冻结：分支名 15 → 12

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
