# 服务器运行目录只读复核

## 2026-10-06 最新补充：两条历史交付分支退出，五套独有 overlay 保全

服务器 `codex/sister-bounded-20260927`（旧 OpenMoji 有界恢复交付目录，HEAD
`3f85510285e5ffdfca28def93eef2eb082b1655c`）和 `codex/t08-publish-retrieval`
（文搜图历史目录，HEAD `d0662a7d240340817948a3496c2cd43f4240e76d`）已固定原提交、
退出开发分支，分别保留 3609／4301 个文件；文件 stat、索引和原状态前后一致。
同 UID cwd、命令、打开文件检查无占用后才执行，根运行 checkout 不变。服务器分支
**13→11**；83 个目录继续保留，不增加磁盘释放数，不重训或操作设备。

完整双端 Git 恢复包 SHA256 `7e752530ad4b904bdd87c40d1d6752e004198887204a59ae3ddd2aa82638f4cb`，
服务器 `storage_cleanup_manifests/t08_frozen_history_20261006.bundle`，本机
`.codex_tmp/storage_git_backup_20261002/t08_server_frozen_full_20261006.bundle`。
恢复引用 `refs/archive/frozen-server-20261006/<原分支末段>`；私有操作收据
`.codex_tmp/server_t08_frozen_20261006.json`。旧有界恢复脚本的独有未跟踪来源另在
`RUNTIME_RECOVERY_IDENTITIES_20261004.json` 登记，原文件本轮没有移动。

另核查五套历史 A100／T08／Spatial 工程：共 **24 份已跟踪修改**，全部与同路径
main 不同，两份路径在 main 缺失（批量测速脚本与 Spatial 配置）。不能把这些差异当
垃圾或直接覆盖正式实现。五套已跟踪 overlay 在独立私有 Git archive 引用中保全，
工作文件、原 HEAD／状态未改变；完整历史包服务器 verify、本机 SHA 及 verify 通过，
本机已通过 Git 导入五项恢复引用。包 SHA256
`4807e4515eadd74c50981ee8a2454988c55e03314ab8054cca6ed79bd9342115`，
双端包名 `dirty_tracked_history_20261006.bundle`。用途、原提交、逐文件 SHA 和下一步
见 [剩余源码例外](REMAINING_SOURCE_EXCEPTIONS_20261006.json)。

这次仅封存已跟踪差异，不证明未跟踪源码／资产或全部依赖闭合；五个开发分支尚未
退出，也没有将其科学源码整仓合入 main。保全不是迁移完成。随后额外只读分支盘点
两次 SSH 握手失败，未重启任何实验；11 条来自刚成功的冻结事务实际计数，不冒称
已经取得额外全机器盘点。

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

### 旧 T04／T07 开发分支退出（2026-10-06，后续覆盖）

最新只读检查和操作前重复检查确认下列两项无同UID进程cwd、命令或打开文件占用，
且无tracked修改。先完成全历史Git包server verify、下载与SHA核验，再取消分支名；
不checkout其他版本、不改索引、不移动数据。各目录仍原位固定下列提交。

| 历史工程用途 | 原分支末段／冻结HEAD | 保留文件数 |
| --- | --- | ---: |
| OpenMoji 旧DC30／CCD噪声阶段，不是现用rank64运行目录 | `t04-dc30-ccdnoise` / `a61a3746d39765991048d95e40812d1028d70c93` | 34609 |
| ABO旧fresh35阶段，不是封存rank72的latestfresh35目录 | `t07-fresh35-20260929` / `bd1fcf136ac23ff2f1a4f3f9fc5c1f8bfc614239` | 4327 |

恢复引用：`refs/archive/frozen-server-20261006/<原分支末段>`；完整包SHA256
`c997416ea56a749f7834dc360d862975a71a27860fcfda387942e06ebf713fd5`。
服务器包：`storage_cleanup_manifests/old_t04_t07_frozen_history_20261006.bundle`；
本机包：`.codex_tmp/storage_git_backup_20261002/old_t04_t07_server_frozen_full_20261006.bundle`。
收据：`.codex_tmp/server_old_t04_t07_frozen_20261006.json`。
两目录全部文件stat、索引摘要及原Git状态前后一致，服务器根运行HEAD和改动未变。
分支11→9，注册工作树仍83；未删除科学文件／测速，磁盘释放为0。
Git包保存历史源码，不等于这些目录的未跟踪数据也已打包，后者仍原位保留。

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
