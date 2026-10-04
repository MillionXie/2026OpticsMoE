# 文搜图采用版：实拍回放与读出工具

本包归入当时服务器实际运行的七份工具，来源和逐项 SHA 见仓库
`maintenance/storage/T08_PHYSICAL_TOOLS_IMPORT_20261004.json`。仅将局部导入改为
包内导入、将反向检索实现明确绑定 `reverse_runtime`，计算函数和历史科学合同不改。
命令从仓库根以 `python -m LightGenV2.tasks.t08_abo_image_text_retrieval.physical10cm.<模块>`
调用；各模块 `--help` 列出显式数据/配置/封存权重/CCD目录参数。

- `export_full_next_stage`：按全数据逐层导出，重放此前各层真实CCD；TRAIN固定每类8图。
- `cache_physical_readout`：从六层真实CCD重建末端读出前特征。
- `finetune_readout`：只改原LN+Linear读出，600 TRAIN拟合/200 TRAIN留出选模，封存后TEST一次评价。
- `evaluate_full_physical`：固定主体或指定SHA的适配PT，重放100标题和2400 TEST图。
- `audit_physical_router`：逐样本记录原真实路由。
- `export_router_pilot`、`export_full_vision_router`：历史导出和诊断对照，非新硬件验收。

这是 **2026-09-25旧文搜图合同**，保留其每图正值p99.5幅度缩放，不能冒称后来
ABO/OpenMoji的保零bounded tanh合同；不要套给其他主体或已有新版CCD。
主体 `cc977b83...`、用户采用10轮适配读出 `89e25360...` 的 .85 版本及200轮
读出 `d4d1f6c4...` 的 .99–1.00 对照分开保存，不相互覆盖、不混用测速PT。

这些是原运行工具，原输出写入/接续行为保留；仅在明确授权的新实验目录下运行，
不能用整理工作触发重采、重评或覆盖已封存报告。本次只进行了源码和有界合成CPU检查。
Windows SLM/CCD采集助手及私人连接/接力脚本尚未迁入本包，当前不能称整个硬件工程已闭包。
