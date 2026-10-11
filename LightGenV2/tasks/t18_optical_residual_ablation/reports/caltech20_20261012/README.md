# Caltech20：每类60训练组

最新人工指定20类。类别按源图数量与预固定种子选择，不按精度挑选；不是完整Caltech101。
准备计划：../../caltech20_preparation_plan.json；协议：../../CALTECH20_PROTOCOL_20261012.md。
每类60 train组/10 val组/其余test，上一轮train/val保留原角色；所有选中类别图片保留。
数据已准备：3888图/3888精确像素连接组；train1200、val200、test2488。
每类train60/val10，所有旧50类train/val保留原角色，进入新test数量0。
实际身份、20类名及支持见[data_identity.json](data_identity.json)。
新数值预检及运行PID通过后追加收据。
所有test保持未评估。原50类低精度结果保留，不当20类结果或隐藏负结果。

新train-only预检完成，真实收据见[execution_probe.json](execution_probe.json)。
2/4两rho micro1/2/4通过；8/16两rho仅micro1通过，micro2/4概率误差超过原1e-6门槛，失败保留不放宽。
Linux7项CPU检查通过。两层正式100轮使用micro4，16层只2轮整轮计时使用micro1，
不能套上一轮50类micro4速度；当前不启动其余深层正式臂、不提前test。

已真实启动caltech20_s17_L2_e100_L16_timing_4gpu_20261012，CPU父776112。
GPU0/4两层rho0/.3子776114/776115，各完整100轮；GPU1/2十六层计时子776116/776117，
各仅2轮，不当正式100轮结果。四个生产smoke均通过、67实际运行和原Gitblob匹配
aaccbeb7911eade85723964f5ef052161ca5d487，同depth初始化SHA一致。
启动收据见[production_start.json](production_start.json)，显存约926/1168/1020/926MiB。
此为启动快照，后续以current_processes/status核对，不重复启动；尚无可靠新精度或整轮ETA。
训练期间冻结67实际来源，只独立报告/CPU审计可发布；HEAD变化不代表训练源变化。
