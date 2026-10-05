# 服务器运行目录只读复核

## 2026-10-06：七个候选的保留决策

重新核验上述七目录，均无 tracked 修改、未跟踪或忽略文件；同 UID 进程的 cwd、
启动命令及打开文件均未发现引用。对全部 83 个登记工作树的已跟踪 Python、shell、
YAML、JSON、Markdown 合并搜索，`fa_vtab_20260914` 在 25 个树的固定反馈 VTAB
命令／优化记录中被引用，其余六个名字未匹配。此搜索不覆盖未跟踪启动器、外部
工程或所有动态引用，不能据此证明完整依赖解除。

七目录各有 582–626 条名称涉及报告／测速／功耗的已跟踪路径，并非只有源码。
例如 `t07_gallery_fp32_20260912` 含 T06 的原光学组件 timing CSV、T08 的原
power_samples／timing_per_sample，以及历史 timing_recheck／historical_timing
证据。这是文件用途核查，不是这些报告指标的再次验证。

**按用户全部测速及其副本保留原位的要求，本批七目录保留为冻结历史证据例外，
不再列为待整目录删除候选。**源码日常入口仍为 main；保留旧证据目录不表示继续
维护七条开发线。没有删除文件、运行训练、读取 PT 或触碰 GPU/设备。

私有收据：`.codex_tmp/clean_candidate_all_tracked_dependency_audit_20261006.json`
与 `.codex_tmp/clean_candidate_evidence_audit_20261006.json`。首次逐项重复扫描超时，
核查原检查进程已结束后改为合并搜索取得结果；没有同时启动重复检查。

下方为 10月5日历史快照，main 身份和用途待审状态由本段及后续发布记录覆盖。

### 服务器旧 T06 读出压缩分支已退出开发线

2026-10-06，服务器 `/DATA/DATA1/guest3/.codex_worktrees/t06_readout_compress_20260912`
在原提交 `de866ff8b890b77b18a4f873f5be2d216ae4c7fa` 上冻结为 detached。
该目录是历史读出容量压缩试验，不是当前正式 Spatial／Temporal PT；本轮不选择或
修改科学模型。再次确认无 tracked 修改、同 UID cwd／命令／打开文件引用后，仅更改
HEAD 元数据并条件式取消 `codex/t06-readout-compress-20260912` 分支名。
3588 个文件的大小、mtime、类型与索引摘要、原 Git 状态前后一致，没有操作内容文件。

提交保留于 `refs/archive/frozen-server-20261006/t06-readout-compress-20260912`。
完整历史 bundle 在服务器 `storage_cleanup_manifests/t06_readout_full_history_20261006.bundle`
及本机 `.codex_tmp/storage_git_backup_20261002/t06_readout_server_full_20261006.bundle`，
两端 SHA256 `9e9ae0c5b6b2c885e6ea9d7ee8619f21ea2cd6d696e0fea5a1239cfb6b50bc9f`。
服务器 bundle verify 通过，下载内容 SHA 一致后才退出分支。
服务器分支 19→18；工作目录数量未减，没有释放空间或删除测速。
现用 OpenMoji 与根运行目录仍有进程引用，未改它们的 HEAD、源码、数据或任务。

同日旧整体压缩目录 `/DATA/DATA1/guest3/.codex_worktrees/t06_compress_all_20260912`
也在原 `630d09367219a4bddd70ed098d2cbdbe8040b617` 上冻结，取消对应
`codex/t06-compress-all-20260912` 分支名。4426 个文件的 stat、索引及 Git 状态前后
一致，未操作科学文件。历史引用为 `refs/archive/frozen-server-20261006/t06-compress-all-20260912`。
两端完整恢复包 SHA256 `20ea40156418adbc107cb0b47c0b6bd50cd053d61bcbc8849b80533ca9916c19`，
私有本机包 `.codex_tmp/storage_git_backup_20261002/t06_all_server_full_20261006.bundle`。
服务器分支 18→17，工作目录和全部旧测速仍在原位。

### T11／T12 四条历史分支统一退出（2026-10-06）

四目录在各自原 HEAD 上冻结为 detached，先验证服务器完整 Git bundle，下载本机并
核对同 SHA 后，再逐目录重复检查 cwd／命令／打开文件无占用、无 tracked 修改。
仅变更 HEAD 元数据及条件式取消分支名；目录及全部数据／测速不移动。

| 原分支末段 | 冻结 HEAD | 保留文件数 |
| --- | --- | ---: |
| t11-optical-lifelong | `604cd89e3e4a88ed0b580eaa41f88af7de2da86c` | 4608 |
| t12-physical-robust-20260926 | `197beba57440dff0ca2082fd3449635ce7d91c5a` | 4848 |
| t12-physical-robust-v2-20260927 | `7093ec46082eed2fae127ec5028d3e2e8548b592` | 31188 |
| t12-text-to-image-20260920 | `9502371917e33e1e45388010208d510c0688f9bf` | 4967 |

各恢复引用为 `refs/archive/frozen-server-20261006/<表中末段>`。
完整包：服务器 `storage_cleanup_manifests/t11_t12_frozen_history_20261006.bundle`，
本机 `.codex_tmp/storage_git_backup_20261002/t11_t12_server_frozen_full_20261006.bundle`；
SHA256 `aa063302d091af23eae898c4a434d87a66de0201bf748878afcb32e97f65151f`。
逐目录文件 stat、索引摘要和原 Git 状态前后一致，服务器根 HEAD 和原改动状态不变。
服务器分支 **17→13**，工作树仍保留；没有空间释放、重训、重评或设备操作。
私有逐目录收据 `.codex_tmp/server_t11_t12_frozen_20261006.json`。

本机另以 `check_task_registry.py --verify-local-weights --verify-local-artifacts`
复核全部14项登记，已登记本机 PT 的流式 SHA、私有报告／引用检查无错误；不加载
模型、不读取科学数据。此检查的范围仅为登记条目，不能证明所有任务资产闭包完整。

2026-10-05实查83个注册工作树全部存在，19条分支名；当前main引用
`aa78b93cbafbcd9c8a64c1ff10247468aa8429b9`，服务器根运行HEAD仍为
`3f85510285e5ffdfca28def93eef2eb082b1655c`。没有切换checkout或删除目录。

## 为什么不能按数量批量删除

10个工作树有已跟踪修改，62个有未跟踪内容，73个有忽略内容；类别重叠，不能相加。
忽略内容可能是正式数据、run或缓存，不等于垃圾。按本用户同UID的`/proc/*/cwd`
只读检查，有2个注册目录被进程作为工作目录使用；这不等于只有2个任务运行，
脚本可以从别处启动或引用其他目录，尚需命令、打开文件及下游引用审计。
未发现缺失注册目录，所以不能仅靠prune减少数量。

服务器根提交树与main的1265个路径差异分别是main独有908、根历史版本独有306、
同路径内容不同51；未提交overlay不在该比较内。不能称为1265个丢失提交，
也不能以清零该数字作为完成标准。

## 无额外文件的七个目录仍有用途待审

下面七项均为detached、无tracked修改、无untracked或ignored文件，当前没有同UID
进程cwd占用。用途说明来自其HEAD提交标题，只是定位线索，**不是完整依赖审计**。
本轮全部保留；后续必须验证必要对照、全部测速、源代码恢复包及下游引用才作取舍。

| 旧目录 | HEAD | 用途线索／保留原因 |
| --- | --- | --- |
| fa_vtab_20260914 | a2e6096fbe6ce43e6bbbc55b88f8ceb95aa4fd8b | 固定专家实验GPU UUID绑定，可能属于必要baseline |
| t03_handoff_20260915 | 3b931ccdcbd398f1c73f57d46621eccbddeff4f0 | SALICON校准Meadowlark/TUCam三步交接，需确认设备依赖 |
| t07_gallery_fp32_20260912 | c8ec81ced7f1c4d54b848ec99f747ac38c77bae5 | FP32图库训练损失对照，不自动视为无效 |
| t07_goal81_verify_20260912 | 5f3c704ad93ee8baade05ea2fc13823119c21d84 | 显式SHA权重独立复评，可能是结果证据 |
| t07_joint_best_restart_20260912 | 35055dc78d9ddd585e3d39357312f2f508cd287d | 固定已验证joint best用于成对重启 |
| t07_retail_audit_20260913 | 5a7bf527a1b4a479180097f493c5ac648057ab31 | ABO正则化、SHAPE pilot与交付审计 |
| t07_vision13_20260913 | 6c1320c2ff798321404a5e6060852033130b52b2 | 校准79.5833 best核验与有界vision13对照 |

完整私有只读收据位于 `.codex_tmp/git_runtime_scope_refresh_20261005.json`，
用途提交标题收据 `.codex_tmp/clean_tree_purpose_probe_20261005.json`；未公开数据、
进程命令或连接信息。该复核不批准删除，也不证明所有旧源码已归主线。
