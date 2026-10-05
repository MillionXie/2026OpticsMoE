# 服务器历史 checkout 原位冻结（2026-10-05）

本次两套历史源码目录均无跟踪修改、无未跟踪文件，未发现进程 cwd、命令行或
打开文件引用。原位 detached 到完全相同的旧提交，然后取消日常开发分支名。
没有移动或删除任何文件；原主工程的 HEAD、索引 SHA 和跟踪状态均保持不变。

| 原目录 | 原用途 | 固定提交 | 核验文件数 |
| --- | --- | --- | ---: |
| `/DATA/DATA1/guest3/t12_cross_modal_20260920` | 旧编号 T12 的四模态终身学习／stage D 历史运行源码，现归 T16 保留 | `2f2f6946f38c7d06320eb29314654a6260f3d618` | 5958 |
| `/DATA/DATA1/guest3/t12_git_audited_20260926` | 图文编辑 T12 审计发布源码与历史 baseline／硬件代理限制 | `a3f5e6fe7027e92fe29be2934750f427cdfd91e7` | 4874 |

10832 个文件的操作前、紧邻切换前、切换后完整身份一致。普通文件逐字节 SHA256；
符号链接记录目标、不跟随读取外部数据。测速、baseline、结果与数据保持原位。
服务器开发分支名 21→19，83 个注册工作树仍保留；不计作新磁盘释放或运行目录已统一。

取消分支名为 `codex/t12-cross-modal-lifelong-server` 和
`codex/t12-audited-editors-20260926`，完整历史保留在
`refs/archive/frozen-server-20261005/<原分支末段>`。
独立 Git bundle 已通过 verify，文件清单／收据在
`/DATA/DATA1/guest3/storage_cleanup_manifests/frozen_checkouts_20261005`：

- `t12_cross_modal_20260920.bundle` SHA256：
  `002463a56287348e9b58e3452712ec319512fa45a289fb568f09bfcdcd9cc6c1`。
- `t12_git_audited_20260926.bundle` SHA256：
  `96c4a641daf7353b4f1e36cda69d7a2442430d59cd2a942fad5d10613ac2bfc8`。

另两套 sister 发布目录存在独有内容或源码修改，本批未冻结、未清理；仍原样保护。
OpenMoji 现用实验、ABO 封存树与硬件任务没有修改。旧冻结目录只供历史复核，
长期开发仍应从已发布 main 的对应正式任务入口开始。
