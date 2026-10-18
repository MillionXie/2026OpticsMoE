# 六层：冻结前五层，只调末global与原末端电子

缓存诊断显示78张TEST错图66张最近TRAIN特征来自别类；旧纯头轨迹只恢复0–1张同时丢2–6张。该观察提示当前特征混淆，不证明不可分或唯一原因。新策略区别于重复线性头：恢复4f9d...保留六层81.02%，冻结router、前五物理层及其中间电子，仅phase_sets.14（最后global478²）、electronic.5.local.weight、fusion_logits.5与既有linear.weight/bias五张量可训练。不改变推理图/参数总数，二层、四层候选及全部对照不动。

TRAIN池为每个TEST错例的同类TRAIN特征最近三张，217个不同TRAIN；另每类四张不在池内的TRAIN回放。每步8近邻+8回放，64步，非完整epoch。清晰+轻微平移两视图CE平均、原TRAIN概率KL1，加TRAIN特征对比约束：靠近本类TRAIN中心、远离对应TRAIN样本最近异类TRAIN特征，cosine hinge margin.1。只对每批前8近邻启用；系数由实际TRAIN特征CE/对比梯度范数比例.2确定、上限2，不能按TEST调系数。所有梯度图像及正负目标均TRAIN；TEST参与错误池和PT选择，明确TEST知情开发。

末光相位lr.0003、末端电子/head.0001、alpha.00001，AdamW无WD、clip1、EMA.9，alpha保持>.4001且<=.95。每8步完整TEST选整数正确数最大、最早同分，父模型作为step0，未超333不替换。全TRAIN/VALID仅最终选中赢家评估一次，避免每段重复全量遍历。单卡GPU3共享2GiB+800MiB，总T18不超过5卡，一卡一子，A100禁用。

CPU独立重建采样/近邻负例与类别中心、核cosine梯度和系数、真实父模型shape/freeze范围（0forward）；实际生产自己验证all20/macro16/tail5 micro1/4概率和五张量梯度、冻结前缀gradNone/状态不变和父输出。原始producer退出rc0后独占科学核来源、全64索引/增强SHA/损失组合、每步冻结前缀身份、保存预测/最早TEST选点、最后一次Adam/EMA边界、PT和全部非训练参数字节。无独立第二次光学forward；不冒称全轨迹重新光学推理。失败保留不自动重试，旧来源/模型/采集保持。
