# T11实际源码发布与入口补漏收据

2026-10-04。此次只整理代码与身份，不重训、不重复科学评估、不操作光路、不删除数据。

## 已发布

- 精确服务器T11来源 `72180033c6edd605c7151b2faa8a599dc15910ee`：54项源码、配置、
  必要baseline与原复现协议进入main；任务README保留全部协议并追加当前身份说明。
- 第一批发布 `5ba7143f0e9176978252f74f4a77ce5d5a9528e5`，后续工具与入口补漏发布
  `ba0effba9ab867206d95c1c0292246cc8d347d70`。源码逐项SHA记录在T11的source_import。
- 未改变纯光/联合D2NN/CRC9 OEO三类模型合同，不将各自PT混用；没有拿新版覆盖旧run。
- 登记334项原资产记录、162项best/last SHA。原数据、预测、收据、时间/功率测量原位保留。

## 检查结果

- 发布Git树：14任务入口/导航/公开身份清单无缺失；私有handoffs仍另行核验，不在Git。
- 精确来源核验：143项T11/T13/T16源码、22项教师快照、251项T03/T10已审源码，共416项通过。
- 服务器原T11目录：18项CPU合同通过。
- 服务器发布树：31份Python编译、18项CPU合同通过；从Git blob直接加载，无新checkout。
- Git-tree检查器两项新增测试通过；其余33项治理测试通过。旧本地工作目录的完整registry
  测试仍会报告未迁入的T11文件，不覆盖旧工作文件来消除这个真实差异。
- 本地torch DLL问题未改动。初版Git-tree测试工具namespace错误已修正；记录保留在私有
  审计收据中，最终发布树复测成功，不把工具失败说成科学模型失败。

## 三端与保留边界

本地main、GitHub main、服务器origin/main同步到上述后续发布；服务器原工作HEAD及
已跟踪状态在fetch前后相同。引用同步不等于原运行目录已经切到main。

私有CRC9 Git恢复包SHA：
`6f4dd18eee45b599813e81e4a1fa977b4370e4df2d6df7356fca2bd3cbb86ee7`，
固定恢复引用 `refs/archive/server-authoritative-20261004/t11-crc9-runtime`。
Git恢复包需已保留的joint-D2NN历史提交，不冒称独立完整仓库备份。

补漏包含ABO/OpenMoji既有FINAL_IDENTITY、OpenMoji复现说明，以及历史A100九任务测速
README。只补入已核验记录，不更改当前OpenMoji实验、ABO封存模型或测速数值。
缺失清单见 `MISSING_ENTRY_ADDITIONS_20261004.json`，不是删除名单。

三套T11旧运行目录及报告PNG继续原位保留；新克隆若需要原报告图，应按原位置取回，
不把仅源码发布当作全部资产已搬迁。完整工程统一目标仍在进行，未宣布全面整理完成。
