# 唯一复现入口

本轮是源码架构，不是训练结果复现。导师v2来源、5展示/4训练的合同见任务README；命令见COMMAND。

已执行的本地证据：源快照22项SHA校验；条件/数据划分/传感器梯度/相位映射/共同验证恢复单测；G5合成CPU前反向冒烟。对应临时产物位于本任务 `runs/smoke/`，不提交Git。正式测试结果和服务器提交身份以最终同步报告为准。

导师v2的release.json仍保存历史逐field simulation_prediction，不能用它与最新0.80228 runtime比较；当前逐样本golden来自独立CUDA验证 `server_cuda_full_current_20260923.json`（SHA见reference/golden_first_field.json）。本地torch2.11 CPU与该torch2.6 CUDA第一场最大差0.0084343 MOS，按原导师逐预测0.01 MOS门限验证；不是bit-identical，也不是全558复评。曾以旧逐field记录和更严0.001门限检查的失败日志保留在runs，不掩盖。

训练资产预检当前缺manifest、两种Qwen-front缓存和train-only soft targets的本任务登记路径。源导师包自身也声明原始两种训练缓存缺失。未恢复身份前不启动完整训练；打包test fields不能用于训练。

后续每组报告必须记录：训练模块、原train留出validation身份、558test历史选模局限、完整命令、Python/PyTorch/GPU、commit、输入/权重SHA与硬件session。四组光学相位发生变化，需要各自采集；原0.7977实测不能冒充新组结果。
