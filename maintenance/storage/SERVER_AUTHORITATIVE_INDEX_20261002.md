# 服务器为准：代码、数据与归档索引

2026-10-02用户明确：以实际服务器运行版本为准；原实验数据、最终版本保护，中间试错可清理。此次只移除可恢复的旧代码工作树，不删除实验runs。

## 必须保留的当前入口

| 项目 | 训练服务器代码（guest3主工程下） | 当前保护点/说明 |
| --- | --- | --- |
| ABO最终rank72 | `.worktrees/t07_latestfresh35_20260929` | HEAD `d979b53ed509907a3630bd2a36c317dfb3aac965`，实际工作代码/启动脚本已备份；最终PT SHA已再核验 |
| ABO旧光电对照/鲁棒实验 | `.worktrees/t07_robust_20260926` | HEAD `62ac6c52b0bef264edabdd25d41200820ac8fc01`；原对照权重、结果均保留 |
| OpenMoji robust/rank48/rank64仿真 | `.worktrees/t04_openmoji_robust_20260928` | HEAD `3b956503a4ecae9b6c20c789ead28621e42e6eeb`；README/profiles未提交修改及6份未跟踪源码不能仅靠HEAD恢复，已另存overlay |
| T12最终及对照 | `.worktrees/t12_physical_robust_v2_20260927` | HEAD `7093ec46082eed2fae127ec5028d3e2e8548b592`，全部有效输出保留；其他T12历史/审计目录也未动 |
| T13最终及消融 | `.worktrees/t13_temporal_robust_20260927` | HEAD `493eb38375f3bd9e4e8e7edfe278468b351365f1`，代码与结果保留 |

“服务器为准”不是“guest3主目录HEAD是全部任务最新版”。当前每项仍从实际隔离工作树运行；主目录与各任务HEAD不必相同。以上是当前保留的代码身份，不把含未提交代码的版本冒称单个干净Git提交。

### 实拍推理代码在师弟Windows服务器，不是上述仿真源码的简单替换

- ABO最终部署：`E:/code/guest/2026OpticsMoE/ABO_I2I_Lab_DVP_8um/candidate_latergb_rank72_20260930`。
- ABO最终训练PT：`/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/abo200_latergb_rank72_20260930/best_epoch13_snapshot.pt`；SHA256 `25f23260864a00a0f32b5de209c27f5876166018c8d8ab0e880de81e0d2c2c22`。
- OpenMoji rank64部署：`E:/code/guest/2026OpticsMoE/OpenMoji_Robust_Rank64_SHS_20261002`；G5最终PT SHA `1fa31ec7b30a554280d9115b54f580d40b9c754f805db4ec9714ec28d22af41f`，完整记录见对应FINAL_RANK64_VERSION。
- OpenMoji rank64 G2独立采集/微调是新获授权的实验，相关代码/任务/原CCD全保留，不作为旧试错删除。
- 原始设备控制模块、前端/数据依赖所在的ABO/OpenMoji基础工程，同样保护。本轮没有删除或改动Windows部署工程。

## 已移除的中间代码副本

- 54份9月11–13日旧ABO控制实验工作树。
- 每份均核验：HEAD固定、detached、无已跟踪修改、无未跟踪文件、无忽略文件、无子模块、无进程工作目录占用。
- 只使用 `git worktree remove`，没有force。已知final/handoff/verify/best/audit等目录不纳入本轮删除名单。
- 权重、原数据、CCD、缓存（除之前单独获准清理的baseline特征）、报告和实验runs均未触碰。
- 注册工作树从141减至87。代码副本约20.53GiB，扣除恢复包后净释放约20.23GiB。共享盘可用约255G→275G；此前baseline特征清理约57.89GiB，两轮合计净释放约78.12GiB（不计随后本轮保护包少量新增）。
- 移除清单：本地 `20261002_code_archive_receipt.json`，服务器 `storage_cleanup_manifests/code_archive_20261002/receipt.json`。

## 如何恢复一个旧试验

服务器每个旧HEAD都保留在 `refs/archive/storage-20261002/<旧目录名>`，独立完整Git恢复包也已通过verify。

```sh
git -C /DATA/DATA1/guest3/2026OpticsMoE worktree add --detach \
  /DATA/DATA1/guest3/2026OpticsMoE/.worktrees/<旧目录名> \
  refs/archive/storage-20261002/<旧目录名>
```

恢复包：`/DATA/DATA1/guest3/storage_cleanup_manifests/code_archive_20261002/retired_abo_code.bundle`；SHA256 `3024035ae6d752c36fe22a36e895baca13b52ef4cc1463c75aafb81ed1403d0b`。本地备用包保存在 `.codex_tmp/storage_git_backup_20261002/retired_abo_code.bundle`；以实际本地SHA核验结果为准。

## 本地与服务器如何对齐

不覆盖本地现有目录、未提交修改或其他任务分支。服务器5个关键HEAD用同名独立引用固定：

`refs/archive/server-authoritative-20261002/<任务工作树名>`

完整提交历史包 `server_authoritative_code_20261002.bundle` 已在服务器保存，并下载本地核验SHA256：`62f742319651bcdcbf538231d4ea800c0f7ea9298fb918b50796b089735a24e0`。

未提交修改和39份未跟踪源码/启动脚本独立保存为 `server_source_overlay_20261002.tar.gz`，SHA256 `fc1e5fdc6fb9c75668cc453215410046381077b6eb1a5bedc62e6f12e437e08c`。两端都有副本。overlay还带文件清单和逐源码SHA；恢复时需要HEAD+补丁+对应未跟踪源码，不能只拿HEAD。

私有恢复包均位于本地 `.codex_tmp/storage_git_backup_20261002` 和服务器 `storage_cleanup_manifests`，没有公开上传，也未恢复师姐上传。

### 后续安全 Git 治理（2026-10-02）

已移除本地2条、服务器25条完全合入 main 且无工作树使用的历史分支名；不是删除
源码/数据或 Git 历史。每条 HEAD 保留于 `refs/archive/merged-branches-20261002/`。
本地分支48→46，服务器75→50；工作树数量及运行工程未变。
逐项收据与后续计划见 `../git_safety/README.md`。根协作规则已在本地纠偏，但尚未
发布/同步到全部工作树；主线整合仍待逐任务差异审计，不冒称三端已统一。

## 测速、功耗及baseline也保护（2026-10-03追加）

用户明确要求保留旧测量和相应baseline，不仅是最终模型。测速/能耗总表、原始
逐次计时/功率遥测、硬件与scope审计、配置/命令/源码/PT身份、逐样本结果及必要
对照均不得按“中间试错”直接删除；历史和拒绝数据标注后保留。具体见
[保护清单](MEASUREMENT_BASELINE_RETENTION.md)。未列入本地16目录清单的服务器
原始测量、任务报告和baseline同样保护；清单不代表已经完成全机器备份验证。

## 其余目录为何暂时保留

87个剩余工作树里，有最终/交接代码、命名任务分支、未提交修改、未跟踪/忽略文件或正在使用的目录；不能仅因“旧”就清理。分类原始清单：`20261002_worktree_classification.json`。后续若继续清理，先把独有源码/输出归档核验，再移除中间代码；原数据和有效实验结果仍默认保留。

不自动把本地修改反向合入服务器，不全盘add/merge，不强推，不删除Git历史或实验结果来追求表面整洁。
