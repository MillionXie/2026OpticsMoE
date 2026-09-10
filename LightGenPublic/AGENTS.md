# 维护规则（先读）

1. 这里只保留可独立运行的研究审阅版；试错继续在 `LightGenV2` 对应任务进行。禁止在两个目录同时暗改同一实验。
2. 每个任务自包含：模型、训练、配置、说明、测试。禁止运行时导入 `LightGenV2`、`experiments` 或其他任务。
3. 数据留在原位置，通过 `--data` 指定；权重/processor 放任务的 `assets/`；运行输出只放任务 `runs/<run_id>/`；包只放 `releases/`。都不提交 Git。
4. 一个任务只维护 README（事实）、COMMAND（命令）和必要报告。只保存 best/last，不保存周期权重。不要删除旧数据/结果。
5. T07 必须光 router Top2、V/L 各 router+expert+global、alpha>0.4；不引入 Transformer/attention/full-Qwen；改变几何、读出或预处理须建立新实验并说明，不能冒充等价整理。
6. 默认一张 GPU，同时最多两张。训练结束/失败确认自己 PID 退出，不结束其他人的进程。
7. 测试通过后提交并同步 GitHub，报告 commit；服务器执行已同步的指定 commit。源码通过 Git，权重/ZIP 用 manifest+SHA256。
8. 发布前必须隔离目录固定权重复评，核对逐样本特征与指标；“固定权重等价”不等于“重新训练必得相同结果”。不隐瞒按 test 选模，不承诺未达到的 75%。
