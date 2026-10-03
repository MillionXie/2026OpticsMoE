# T10 实际运行源码收敛收据

发布核验：本地main、GitHub main、服务器origin/main已一致至
`c9768ad267254d734e1a7a6770d29a11fa5e1627`；本批44项变更含T10源码、T03收据与总登记表。
候选18份T10 Python与实际服务器快照逐字节一致，14项登记任务入口在main无缺失，
既有89项源码/22项教师依赖SHA复核无错。服务器原HEAD/状态及本地18项原修改保持。
这不代表全部运行目录切换main；后续须在停止占用和保留独有修改后才能收敛checkout。

额外保留：本地未跟踪的T10 plan.py含光学合同校验及旧设计检查变量修正，
依赖本地未跟踪common/optical_contract.py。未覆盖、未把它误当服务器已验证runtime；
后续单独核对共享模块归属，再决定如何纳入main，不影响本批原运行代码的身份保留。

2026-10-03。本批将实际服务器 `t10_corrected_20260918` 的固定HEAD及9份源码overlay，
按精确SHA导入main；最新发布提交见本页末尾追加的核验记录。
不是整仓merge，不切原工作树、不删除数据、不启动旧GPU队列。

原HEAD `5a1178ef09be40f45b58a20a8633a6e1add2d75d`；
完整实际源码Git快照 `38afd43c1e28aeaff28e33bad58177ae09a0f156`。
服务器及本地私有Git恢复包SHA：
`ebf14f4b8c8fda9617319c263014fbde94d92483f7cf48fa9e2f3c60f0c84f01`。
恢复引用 `refs/archive/server-authoritative-20261003/t10-runtime-overlay`；
快照前后服务器HEAD、真实暂存区、工作文件及状态一致，没有把未跟踪代码当垃圾。

29份源码/配置/历史协议纳入main，README另追加当前实际入口，原正文全部保留。
18份Python编译通过，4–100专家几何检查通过；4专家/4层四架构有界CPU检查通过。
本轮没有性能重评或新训练；2层小检查不满足原参数匹配条件，改用原正式4层后通过，未改模型。

4个结果组原位登记；固定478下35份训练result的best SHA均重新核验一致，best/last保留。
数据、TEST锁定清单、逐样本结果及所有测速/功耗保留。旧status不代表任务仍在运行；
结果文件计数不等于全部jobs设计点数，复用条目及完整资产闭包仍待核对。

证据：T10_PINNED_RUNTIME、T10_RUNTIME_OVERLAY_AUDIT、T10_OVERLAY_RECOVERY、
T10_BOUNDED_CHECKS、T10_ASSET_AUDIT，文件名均带20261003。
旧plan.py完整设计检查存在变量作用域问题，已在正式入口指出，暂不当一键复现验收通过。
本轮治理检查33项通过；T03已发布，T10本批发布不把整个任务的资产收尾虚报完成。
