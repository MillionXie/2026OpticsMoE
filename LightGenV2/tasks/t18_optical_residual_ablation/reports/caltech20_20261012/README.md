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

## 新20类实际速度（两层尚在训练）

16层两rho各2轮计时已结束、776116/776117实际退出并释放GPU1/2。
普通轮183.13/176.86秒，含完整train评估轮227.70/222.22秒；
按每5轮train评估估均摊192.05/185.94秒，100轮单臂约5–6小时，
只此新20类micro1模式，非全suite ETA，额外端点/获选train评估另计。
真实收据见[early_throughput_20261012.json](early_throughput_20261012.json)。
2层约60轮仍学习，train约49–50%/val约38%，NLL下降；非最终获选精度，
不能宣称收敛、70%达标或只归因过拟合。等完整100轮审核后决定其它正式深层。
仍test0、67来源未改；未启动深层正式100或额外调整。

## 两层100轮完成、未达70%验证门槛

两组全部100轮，无残差按val NLL选E99，train56.33%/val41.50%，
残差按val NLL选E100，train63.33%/val43.50%；val NLL2.07703543/1.95519117。
选中train-val gap14.83/19.83pp，训练精度本身也有限；不能只归因过拟合。
末十val NLL均值比前十改善1.636%/1.989%，最低轮在末端，100轮不证明充分收敛。
总1339.63/1369.61秒，均摊13.32/13.62秒；全部test0。
保存PT/history/prediction/cache标签CPU独立审核通过：67实际和archivedGitblob、
最低NLL首次、参数11相位梯度、初始化/order/augmentation100配对、config唯rho、
ID标签argmaxsupportCM/NLL/macro均过；无完成臂重复模型推理。
真实结果、拟合审查及PNG/SVG/PDF/CSV见[L2](L2/pilot_summary.csv)及
[fit_review.json](L2/fit_review.json)。所有四个GPU子/CPU父实际退出、GPU无本项目占用。
尚无正式4/8/16结果，也没有最终test；已询问优先学习诊断还是接受长时间深层比较。
不因低分换类/追加或重测旧test，当前深层启动取决于该预算优先级回复，监控仍ACTIVE。
