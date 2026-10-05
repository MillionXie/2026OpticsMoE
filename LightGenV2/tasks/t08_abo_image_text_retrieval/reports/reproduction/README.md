# T08 双方向检索复现入口

当前采用版本及历史指标以 [任务 README](../../README.md) 为准。
本次是源码和资产身份核验，不是重新训练或固定权重精度复现。

## 2026-10-04 采用版资产核验

文搜图采用版是 10cm 主体 `cc977b83...` 加10轮末端读出 best `89e25360...`，
不是更长微调对照。服务器实际读出目录在任务 `runs/physical/alpha040_10cm_20260925/`
下的 `finetune_train800/readout_finetune10`；不能省掉 `finetune_train800` 层级。
best 第9轮、last 第10轮，384→64读出严格CPU加载通过；两份PT的视觉51项和语言51项
上游张量均与封存主体逐值一致。历史训练600FIT/200VAL，10轮历史按VAL选第9轮，
报告TEST Hit@1约.85为原有结果，本次没有重评或训练。

七份主体／读出／历史／实拍特征缓存的完整位置和SHA见仓库
`maintenance/storage/T08_ADOPTED_READOUT_IDENTITY_20261004.json`。
统一只读核验：`python maintenance/storage/check_t08_assets.py --repo-root /absolute/path/to/2026OpticsMoE`。
384维实拍读出特征缓存**不是**64维Qwen教师缓存。2026-10-05已另行找到
正确反向64维shared缓存，完整方向／提示词／CSV／身份顺序检查通过，SHA及位置见仓库
`maintenance/storage/T08_TEACHER_CACHE_IDENTITY_20261005.json`。不需要重建或改写该缓存；
当前本机snapshot的16个文件已完整核SHA，但缓存没有revision，历史生成绑定仍待核验，
见 `maintenance/storage/T08_LOCAL_SNAPSHOT_BINDING_20261005.md`。本次未从原CCD重新生成实拍读出特征，
尚不能声称完整原图→CCD→缓存复现闭环已经完成。SDK/LUT、设备回归和运行目录
切换仍待核验，保留所有baseline、旧测速与更长微调对照，禁止把不同主体的速度拼接到.85结果。


图搜文及文搜图原 baseline 技术说明、原始测速与历史日志均保护；尚未完成的 baseline 入口迁移不得当作已完整复现。

## 2026-10-05 历史冻结 backbone 对照入口恢复

三方向的原始 CLIP / DeepSeek / YOLO baseline 已从服务器实际运行提交
`c93fad700a1da4b30ceeac9af2afc7985934eb00` 原样纳入主线：
[协议](../../ABO_BACKBONE_BASELINES.md)、
[原结果与取舍](../backbone_baselines_20260923/ABO_RESULTS.md)、
[执行源码](../../abo_backbone_baselines.py)。这是历史对照，不替代 T07 封存模型
或 T08 光电模型。原运行目录源码 SHA 与提交相同，五项 CPU 合同测试通过；
本次没有重新提取特征、训练、评估全数据或测速度。

原结果页区分完全冻结的三方向 baseline、训练过的 rank-64 校准诊断及被拒绝的
高容量对照。YOLO 只原生支持图搜图；YOLO+CLIP 文本+拟合对齐不是 zero-shot YOLO。
easy100 是已登记商品的不同视图，不是未见商品泛化。历史报告里的 Ours 行是当时
引用，不能当作当前封存版指标。特征、PT、原报告与测速仍保留原位；资产及模型环境
尚未逐项闭环，不因源码恢复就宣称整套复现完成。
