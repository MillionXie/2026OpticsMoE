# guest3 存储与 Git 整理（2026-10-02）

本页是10月2日历史存储快照；容量、分支、工作目录和待办不代表当前状态。
其中旧“独立整合分支／每任务worktree”建议已被根AGENTS的唯一main规则覆盖。
今天的操作只按[当前待办](REMAINING_CLEANUP_PLAN_20261003.md)，不重复旧清理动作。

最新覆盖：用户确定以服务器实际运行代码为准。54份纯代码旧ABO试错工作树已可恢复归档并移除，数据不动；服务器关键HEAD与未提交源码已分别备份、本地对齐登记。完整当前状态见 `SERVER_AUTHORITATIVE_INDEX_20261002.md`，下面首轮审计数量/待处理项是历史状态，不应覆盖最新处置。

## 当前实际占用

- `/DATA/DATA1` 是共享盘：15T，使用率99%，可用约197G。不能清理其他用户的数据。
- `/DATA/DATA1/guest3` 实占约942.4GiB。主工程约607.8GiB，并非共享盘的全部占用来源。
- 师弟采集电脑 E盘约6144GiB，空闲5634.6GiB；不是目前的数据盘瓶颈。

| guest3 目录 | 约占用 | 本轮处置 |
| --- | ---: | --- |
| `2026OpticsMoE/data` | 265GiB | 原数据保护，包括156GiB ImageNet |
| `2026OpticsMoE/experiments` | 141GiB | 实验、权重、结果保护 |
| `2026OpticsMoE/.worktrees` | 78GiB | 未删除，内含独有数据/输出，不能仅按代码副本处理 |
| `2026OpticsMoE/LightGenV2` | 72GiB | 当前及历史任务结果保护 |
| `demo_reproduction_data` | 104GiB | 原始/解压数据保护，后续可验证压缩包冗余 |
| `LightGenV2_baseline_cache_20260923` | 58GiB | 特征缓存；申请确认清理，重跑时需重新提取 |
| `t12_assets` | 56GiB | 模型、环境、数据、训练结果保护 |
| `.cache/huggingface` | 26GiB | 不是简单临时文件：保留已下载模型 |

CLEVR原压缩包和解压数据各约16GiB；SONYC raw约13GiB/audio约17GiB。未验证逐文件完整性之前不把这些目录判作可删重复数据。

根目录119份bundle/ZIP/压缩归档合计1.62GiB。同大小归档经SHA256复核没有发现完全相同文件；不能凭文件名相似删除。

## 已执行的安全整理

1. 本地16个登记工作树，6个有已跟踪修改；服务器141个登记工作树，10个有已跟踪修改，62个有非忽略的未跟踪文件/目录。
2. 全部脏工作树的已跟踪修改已备份为 staged/unstaged 二进制补丁和HEAD索引。**未跟踪文件仅列清单，没有备份其内容；因此不能据此删除工作树。**
3. 本地备份在 `.codex_tmp/storage_git_backup_20261002`；服务器备份在 `/DATA/DATA1/guest3/storage_cleanup_manifests/git_backup_20261002`。
4. 本地与服务器 `.git/info/exclude` 加入嵌套worktree/私有助手目录，本地还排除两个临时打包/表格目录。这只是本机降噪，不影响已跟踪源码，也不改变远端 `.gitignore`。服务器原exclude已备份。
5. 删除guest3自己的pip可下载缓存45个文件，共7,999,208字节（约7.63MiB）。不是实验数据，可重新下载，不提供撤销恢复副本。删除清单保存在服务器 `storage_cleanup_manifests/pip_cleanup_20261002.json`。
6. 未重置、强推、提交他人修改、删分支/worktree/权重/CCD。空间仍为99%，不能声称问题已解决。

## Git 如何理解与后续整理

主目录并不是统一最新版本：本地主目录在 `codex/t12-text-to-image-20260920`，服务器主目录在 `experiment/static-expert-caltech`。两个主目录不能直接互相覆盖或强制pull。

OpenMoji源码工作树同名分支 `codex/t04-openmoji-robust-20260928`：本地HEAD `cf825f2cf`（3处已跟踪修改），服务器HEAD `3b956503`（2处已跟踪修改）。这不是“已完全同步”。下一步需按文件审阅双方diff/提交祖先，再做独立整合分支和测试；不要全盘add或把所有任务合进当前T12分支。

最终实验复现入口继续以对应handoff指定的源码、数据与精确权重SHA为准：ABO rank72最终版不改，OpenMoji rank64最终版不改，rank48对照及原CCD保留。暂停的师姐上传不恢复。

建议目录角色：

- **代码**：每个任务一个活跃worktree；已结束且无独有输出的工作树，验证后才能归档。
- **共享数据/模型**：单一数据源，通过路径引用，不复制进worktree。
- **实验成果**：独立runs目录，保留best/last、SHA、协议、history、报告、逐样本与原CCD。
- **可重建缓存**：单独清理清单，清理前说明复现重建成本。
- **传输包**：以内容SHA去重，而不是按文件名或日期判断。

## 追加清理（用户确认后）

用户同意清理旧baseline缓存后，已核验9组冻结视觉骨干特征：LSP/OpenMoji/SALICON × CLIP/DeepSeek/YOLO11s。目录内 `.pt` 是预提取特征，而不是模型训练权重；整包特征和分片共9643文件、62,152,517,338字节（约57.89GiB）已删除。重建源码commit `a2bb0f2b30a6c02f1d2b914aa238ee45519975bf`仍存在，元数据JSON、日志、原始模型、原数据和baseline训练结果保留。

删除前核查未有用户进程打开/映射待删特征。发现3条9月23日父进程已退出的残留worker仅打开保留的日志；未结束这些进程。SSH权限分离进程、PAM辅助进程和僵尸进程不作为特征消费者检查。

共享盘可用空间约197G→255G，使用率仍显示99%。完整删除清单：服务器 `storage_cleanup_manifests/baseline_feature_cleanup_20261002.json`，本地同目录 `20261002_baseline_feature_cleanup_manifest.json`。这些缓存没有删除备份，恢复方式为重新提取特征。

## 待进一步审计

- 58GiB旧基线特征缓存：用户确认后已完成，勿重复清理。
- 压缩原始数据与解压副本：需验证完整性和下游依赖，不能现在直接删除。
- 工作树归档：需核对未跟踪和忽略文件、正在使用的进程、源码引用，并保留可恢复快照。
- Git实质同步：需要任务级审阅合并；本轮仅备份和降噪，没有宣称全仓库同步完成。

机器原始审计：`20261002_server_top.json`、`20261002_server_worktrees.json`、`20261002_git_archive_audit.json`、`20261002_local_worktrees.json`、`20261002_candidates.json`、`20261002_safe_cleanup_receipt.json`。私有补丁不上传公开仓库。
