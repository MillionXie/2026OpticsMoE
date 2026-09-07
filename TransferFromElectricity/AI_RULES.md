# 工作规则

遵循 ../LightGenV2/AI_RULES.md 的源码同步、产物保存和实验记录规则。
用户已批准固定专家库、Caltech101 十类检索的首轮尝试。

- 新代码只进入本项目任务目录；复用 LightGenV2，不复制已有传播后端。
- 当前任务事实集中在 tasks/t01_object_retrieval/README.md；总 README 保留设计。
- 普通实验差异使用 config；runs 按 smoke/simulation/hardware 和唯一 ID 保存。
- 不修改原始 d2nn_pack，不提交其特征或数据文件。
- 源码测试、commit、push 后服务器 checkout 指定 SHA；不传文件覆盖源码。
- 凭证不落盘；数据、基座权重和缓存引用已有路径，产物传输校验 SHA256。
- 每个正式 run 只保留 best/last checkpoint；小型导出相位为单独部署产物。
- 禁止删除用户已有 run、缓存或未提交改动。
