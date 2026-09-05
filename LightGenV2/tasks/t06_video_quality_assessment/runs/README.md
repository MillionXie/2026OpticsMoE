# Runs

本目录只保留说明，实际产物被 Git 忽略：

- `smoke/<run_id>`：CPU 结构和梯度检查；
- `simulation/<run_id>`：仿真训练、测试和相位快照；
- `hardware/<run_id>`：六次光传播采集、逐阶段微调和最终实测。

请使用 T06 入口自动创建 run，不要手工建立 `new2`、`final_final` 等目录。
