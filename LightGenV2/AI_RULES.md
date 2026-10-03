# LightGenV2 AI 工作规则

本文件是 LightGenV2 的跨任务工程规则；仓库协作、分支与清理同时遵守根目录
`AGENTS.md`（2026-10-02 用户最新要求）。任务事实只写在对应任务的 `README.md`，
避免多个状态文档互相矛盾。

1. 开始工作前先读仓库根 `AGENTS.md`、本文件、`LightGenV2/README.md` 和当前任务 `README.md`。
2. 不在 `LightGenV2/` 下随意增加一级目录；新任务必须使用 `tNN_task_name`。
3. 不把原始数据、Qwen 权重、特征缓存、checkpoint、CCD 原图或 ZIP 提交 Git。
4. 不创建 `new`、`new2`、`final_final`、`backup_latest` 等目录；源码版本由 Git 管理。
5. 普通实验差异写进 config；只有计算图确实不同才增加模型实现。
6. 所有训练必须进入任务自己的 `runs/smoke|simulation|hardware/<run_id>`。
7. run 至少保存：实际配置、原始命令、Git commit、环境、指标和状态。
8. 论文结论写入 `reports/`，只引用 run ID，不复制整份 run。
9. 实验室 ZIP 只能由任务自己的 `build_lab_package.py` 生成，并附 manifest 和 SHA256。
10. `common/` 只接收已经被两个任务使用的稳定代码；任务专属结构留在任务内部。
11. 未经人工确认不得删除 run、缓存、原始采集或外部 worktree。
12. `experiments/` 当前是兼容后端和历史证据，不再把新试验散落进去。
13. 修改 LUT、ROI、曝光、朝向或相位中心后，必须生成新的硬件会话身份，不覆盖旧会话。
14. README 只在接口、正式结果或操作顺序变化时更新；每个失败小试验不要求手写文档。
15. **最终以测试通过并发布的 main commit 统一源码。**本次存量迁移先以实际服务器
    已验证运行版本为基线（包括未提交源码），不能把服务器主目录 HEAD 当成全部任务最新版。
    本地独有改动须保留并审计，不能整仓覆盖。一项代码修改只有在测试通过、形成语义清楚
    的 commit、成功 push 到 GitHub 后才算同步完成；未发布必须明确报告待同步。
16. 本地、训练服务器和实验室电脑不得用 SCP/网盘互相覆盖源码。开始任务前先
    `git fetch`，确认工作树；同步只用 `git pull --ff-only` 或 checkout 指定 commit。
    checkpoint、缓存、CCD 数据和实验室 ZIP 仍通过 manifest+SHA256 传输。
17. 禁止 `git push --force`、`git reset --hard` 和在有未保存改动时 pull。发生分叉时
    先备份、逐项审计并说明冲突；未经用户明确许可，不新建整合分支或 worktree。
    长期只维护 main，不用“谁较新就覆盖谁”的办法解决，不批量 merge 旧实验分支。
18. 训练服务器原则上执行已发布的指定 commit。已有已验证运行工作树在迁移期间
    原样保护，未提交源码须单独记录 SHA/备份，不冒称干净提交。后续修改按任务路径
    分工、同文件单写者、Git 事务串行，测试后纳入主线；不得默认另开独立分支。
19. 正式 run 和实验室 release 必须记录 Git commit；复现实验先校验 commit、config、
    checkpoint、数据 manifest 和硬件合同，再比较指标。
20. 与论文结论有关的几何、prompt、路由、融合、帧语义等合同必须写入任务 README
    或正式 profile，不依赖聊天记录；改变合同必须新建 profile，不能覆盖旧结果含义。
21. 多视频光并行必须保持“一条视频一个标签、一个输出”的身份合同。训练和正式评估
    必须使用整幅相干光场联合传播，不能先逐视频独立传播、再在软件中无损拼接来冒充
    硬件并行；任何 padding 样本必须用有效位掩码排除在损失与指标之外。
22. 光学 router 的均衡必须按每个物理槽位跨样本检查；全场汇总后看似均匀不算通过。
    正式候选必须报告路由选择随样本变化的比例，并做同一视频在所有槽位间的循环换位审计。
23. 八任务跨项目进度只汇总到根目录 `PROJECT_SCORECARD.md`，每个任务固定一行。任何性能、
    速度或功耗数字必须附可追溯证据；仿真核心时间、实验台端到端时间和估算时间必须分栏，
    不得互相替代。任务细节仍以对应任务 README 和正式 run 为事实来源。
24. 正式训练 run 的模型权重默认只保留 `best_checkpoint.pt` 和 `last_checkpoint.pt`；周期
    测试只覆盖更新 best，不生成每 5 epoch 的 phase/checkpoint PT。只有目录名和配置明确
    标为 `mask_evolution` 的分析实验可以保留周期相位 PT，且不得混入正式性能 run。

## T06 当前兼容策略

最新 36 帧实现已经在旧目录中完成验证。第一阶段由 LightGenV2 的单一入口调用该
后端，并把新 run 写回 T06 目录；不要复制一份模型源码后同时维护。待其他任务结构
稳定后，再把验证后的公共光学模块提升到 `common/`。
