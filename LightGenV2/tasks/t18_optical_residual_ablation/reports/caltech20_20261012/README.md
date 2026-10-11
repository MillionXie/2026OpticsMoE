# Caltech20：每类60训练组

最新人工指定20类。类别按源图数量与预固定种子选择，不按精度挑选；不是完整Caltech101。
准备计划：../../caltech20_preparation_plan.json；协议：../../CALTECH20_PROTOCOL_20261012.md。
每类60 train组/10 val组/其余test，上一轮train/val保留原角色；所有选中类别图片保留。
数据已准备：3888图/3888精确像素连接组；train1200、val200、test2488。
每类train60/val10，所有旧50类train/val保留原角色，进入新test数量0。
实际身份、20类名及支持见[data_identity.json](data_identity.json)。
新数值预检及运行PID通过后追加收据。
所有test保持未评估。原50类低精度结果保留，不当20类结果或隐藏负结果。
