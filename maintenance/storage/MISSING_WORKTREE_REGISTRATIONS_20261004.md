# 四条失效工作树登记：安全收尾

2026-10-04实际核验并清理服务器Git登记，不删除任何科学文件。

| 原登记 | 保留的提交 |
|---|---|
| source_e548cc33 | e548cc3350223e3647ae548dad8ddf050263d89d |
| source_23612d55 | 23612d55106cb8c817cf4b4e7b38ea5bc138361a |
| source_6d7c4299 | 5298f296699851950b59f49583c53eb660768a84 |
| source_22173121 | 22173121e6b5b01cba08587c36c71186cdd592bb |

原路径均在 `LightGenV2/tasks/t04_semantic_interaction/runs/smoke/` 下。
第三条名称与实际HEAD不同，恢复以表中真实HEAD为准，不猜目录名。
目录和gitfile均不存在，Linux进程cwd无占用，无锁；Git dry-run只匹配这四条。
清理前再次核验名单和HEAD一致，先保留恢复引用/管理目录备份，再执行原生Git prune。

- 注册工作树 **87→83**，没有删除现存工作目录，不以此宣称其它83份均可清理。
- 恢复引用：`refs/archive/missing-worktrees-20261004/<原登记名>`，四提交仍可达。
- 服务器私有恢复包：`storage_cleanup_manifests/missing_registration_recovery_20261004/missing_worktree_administration.tar.gz`。
- 本地私有恢复包：`.codex_tmp/t11_source_20261004/missing_worktree_administration_20261004.tar.gz`。
- 两端包SHA256：`d3edae1bcf048fef82fea0e7bf72331d31bb7c3a1db6bb01b082900244d3d65c`，619139字节。
- 原服务器HEAD仍 `3f85510285e5ffdfca28def93eef2eb082b1655c`，原16项已跟踪状态、
  原index未变化；OpenMoji现用工作树、任务进程、GPU、SLM/CCD未操作。
- 后检查dry-run为空；独立保护恢复引用已核验。

源码工具 `maintenance/git_safety/prune_missing_worktrees.py` 通过三项安全测试：
只清已列missing元信息、名单变化拒绝、现存目录即使缺gitfile也拒绝。
初次服务器检查发现旧Git不支持 `--path-format=absolute`，未发生清理；修正为
相对/绝对Git路径显式解析后重新核验。源码通过Git bundle传输，没有覆盖工作目录。

恢复元信息不代表恢复原已不存在的目录。必要时先读恢复引用和私有包，确认用途后
按真实commit恢复；本轮不新建工作树，不删除数据/PT/有效CCD/收据/报告/任何测速。
这是Git登记收尾，不是释放大量磁盘空间，不能加到此前78.12GiB科学资产清理数字。
