# Caltech20：每类60训练组

## 最终交付：八组100轮及首次单test全部完成（2026-10-12）

2/4/8/16层两rho各100轮均已结束，全八CPU审核/锁定后每组获选EMA首次只测一次。
最高总体test accuracy为4层rho=.3的69.77%，仍未达70%；该组test macro recall50.06%。
测试类别支持不均，accuracy与macro必须同时报告；16层残差总体/宏召回均下降，
不能声称残差越深越有效。全八保存预测/权重/数据/来源/指标独立CPU审核通过，
PNG/SVG/PDF/CSV已保存并视觉检查，训练与测试进程退出，GPU0/1/2/4实际释放。
没有新增候选/扩轮/换类，旧结果保留；[完整最终表、限制与证据](final/README.md)。
以下在训/未test快照均为历史。

## 4层100轮完成，仅16层继续（2026-10-12 13:22）

4层两rho各100轮完整结束，CPU保存结果审核通过，没有完成臂模型复推。
验证balanced NLL首次最低选无残差E100、rho=.3 E99：
train76.0833%／92.00%，val45.50%／56.00%，bNLL1.92216424／1.67628584。
rho=.3末轮val57.5%不替换NLL选定的E99成绩。train-val gap30.58／36.00pp，
两组均未达70%；末十val NLL均值比前十改善2.721%／2.283%，预算末端仍在改善，
不能宣称充分收敛或只因过拟合。实际总1630.64／1648.22秒，均轮16.18／16.36秒。
[CPU审核及CSV](L4/pilot_summary.csv)、[拟合审查](L4/fit_review.json)、
[PNG](L4/learning_curves_L4.png)／[SVG](L4/learning_curves_L4.svg)／[PDF](L4/learning_curves_L4.pdf)
已保存并视觉检查。来源67文件、原Gitblob、数据/PT/配置/首次选轮/21相位梯度、
100order/增强配对及ID标签argmax/support/CM/NLL/macro核验通过。

4层子1088805／1089281已实际退出，GPU0／4无本项目占用；队列无pending或失败。
只剩GPU1／2的16层两组，实际检查约E69／68，冻结67来源仍完全匹配。
已完成2／4／8层六组，全部test0；待16层100结束后自动全八审核、锁定及各首次test。
不另问预算、不增加候选、不重训已完成结果；当前进程以实时状态核对。

## 8层100轮完成，4层自动接续（2026-10-12 12:52）

8层两rho各完整100轮，保存PT、history、预测及数据标签的CPU审核通过，没有重复模型推理。
两组均按最低验证balanced NLL首次选E99：无残差train88.75%／val48.00%，
rho=.3 train99.17%／val55.00%；验证bNLL1.92594350／1.74848565。
train-val gap40.75／44.17pp，两组均未达70%；末十轮平均val NLL相较前十轮，
无残差改善2.975%，rho=.3恶化0.104%、近似平台；两组最低轮均在末端。
存在明显泛化差距，不能仅凭获选E99声称完全收敛或证明残差因果机制。
实际总10624.47／10743.79秒，平均每轮105.82／107.00秒。
[CPU审计和真实曲线](L8/pilot_summary.csv)、[拟合审查](L8/fit_review.json)、
[PNG](L8/learning_curves_L8.png)／[SVG](L8/learning_curves_L8.svg)／[PDF](L8/learning_curves_L8.pdf)已保存并视觉检查。

旧8层子1056117／1056249已实际退出。队列自动派发4层rho0／.3，
GPU0／4新子1088805／1089281，两组fresh production门槛通过；micro4，
相同初始化47d320a52b3c3f4bc2e21a9b8103f797843ec59556f3e56049c367ac240b13a8。
概率最大误差5.96046e-7／7.45058e-7、梯度相对L2 1.33680e-5／.00565468、
最小cos .99999988／.99998397，所有相位梯度有限非零、router无残差。
四个活跃子67实际源码和原Gitblob均一致，完全等同旧L2来源。
16层仍训练，4层接续100轮；没有失败、没有新增候选、全部test0。
此为阶段快照，后续以实际status/current_processes/proc/UUID核验；全八审核锁定后才首次test。

## 当前正式六个深层臂已启动（2026-10-12）

人工指出十余小时空等后，已继续原先全深度100轮比较，不再等待预算选择。
新run `caltech20_s17_L2L4L8L16_e100_4gpu_20261012` 的CPU父1055706实际存活，
CUDA不可见；GPU1/2的16层rho0/.3子1055853/1055985、GPU0/4的8层两rho
子1056117/1056249均实际存活且fresh production数值检查通过，尚为首轮。
4层两组等待允许卡释放后自动派发。各新臂100轮，4层micro4、8/16层micro1，
复用旧L2完整100轮不重训，不复用L16计时权重；test全部0。
67实际来源与当前发布Gitblob核验，等同旧L2运行来源；新父/子记录Git52ab4d58c。
实际初始化同depthpair一致，显存约938/1170/1020/938MiB，四卡单子。
[启动收据](full_suite_production_start.json)保存真实PID/UUID/门槛/来源/队列/显存。
以上只是启动快照，以后按status/current_processes/proc核验，勿重复启动。
L16此前实测100单臂约5–6h，L8/L4以实际新轮计时更新；全八审核锁定后首次test，
队列不自动test。已更新监控去掉旧预算等待，阶段完成后继续审核/交付；普通健康进展安静。

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
不因低分换类/追加或重测旧test。以上预算等待为历史状态：2026-10-12人工指出
十余小时未推进，现直接继续原先全深度100轮比较，复用L2，fresh六个深层臂。
预登记[有界计划](../../caltech20_full_suite_plan.json)：16pair在GPU1/2、8pair在0/4，
4pair接空出的允许卡；最多四子，每臂100轮，未增加候选。原70%门槛失败继续如实报告。
队列只train/val；全8独立CPU审核、pair/selection/global锁定后首次单test，运行收据待实际启动补充。
