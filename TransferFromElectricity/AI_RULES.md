# 工作规则

遵循 ../LightGenV2/AI_RULES.md 的源码同步、产物保存和实验记录规则。
先完整读取仓库根 AGENTS.md；其单一main和保护边界覆盖本项目旧隔离要求。
用户已批准固定专家库、Caltech101 十类检索的首轮尝试。

- 新代码只进入本项目任务目录；复用 LightGenV2，不复制已有传播后端。
- 当前任务事实集中在 tasks/t01_object_retrieval/README.md；总 README 保留设计。
- 普通实验差异使用 config；runs 按 smoke/simulation/hardware 和唯一 ID 保存。
- 不修改原始 d2nn_pack，不提交其特征或数据文件。
- 源码测试、commit、push 后服务器 checkout 指定 SHA；不传文件覆盖源码。
- 凭证不落盘；数据、基座权重和缓存引用已有路径，产物传输校验 SHA256。
- 每个正式 run 只保留 best/last checkpoint；小型导出相位为单独部署产物。
- 禁止删除用户已有 run、缓存或未提交改动。
- 用户明确要求不得占用 A100。启动前查询 GPU 型号和占用情况，显式设置 CUDA_VISIBLE_DEVICES，
  只使用非 A100 的可用设备；不能因空闲而改用 A100。
- 必须按 nvidia-smi 返回的完整 GPU UUID 绑定设备，不能把显示序号当成 CUDA 序号。
  启动前核验 UUID 对应型号，启动后核验 PyTorch 实际 device name 和进程所在 UUID。
- 长期只维护main；未经用户明确许可，不新建分支、worktree或独立工程副本。
- 已运行的历史工作目录保持原commit和overlay，不为同步而切换或覆盖。
- 多窗口按不重叠任务路径分工；共享文件单写者，Git事务串行且只提交本任务路径。
- 遇到未提交修改或分叉，保全并审计，停止有冲突的写入；不以隔离为由自动增加分支。
