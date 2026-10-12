# Caltech20 common100：八组最终结果

全部2/4/8/16层×rho0/.3各100轮已完成；验证EMA balanced NLL最低首次选模，
全八CPU审核及pair/selection/global锁定后，每个获选best EMA只测试一次。
所有训练／测试进程实际退出，GPU0/1/2/4无本项目占用；没有追加候选、扩轮或换类。

|层数|rho|获选轮|train %|val %|test accuracy %|test macro recall %|test macro F1 %|test NLL|test balanced NLL|总拟合秒|均轮秒|
|---|---|---|---|---|---|---|---|---|---|---|---|
|2|0.0|99|56.33|41.50|63.59|41.26|33.65|1.664521|2.105445|1339.63|13.32|
|2|0.3|100|63.33|43.50|61.90|43.53|35.71|1.594379|2.002151|1369.61|13.62|
|4|0.0|100|76.08|45.50|66.40|41.18|34.86|1.405820|1.953469|1630.64|16.18|
|4|0.3|99|92.00|56.00|69.77|50.06|40.87|1.242104|1.783865|1648.22|16.36|
|8|0.0|99|88.75|48.00|66.40|43.77|35.68|1.357620|2.039168|10624.47|105.82|
|8|0.3|99|99.17|55.00|67.40|49.62|39.05|1.309560|1.896651|10743.79|107.00|
|16|0.0|95|82.42|51.50|62.46|42.81|33.14|1.472134|2.130773|18257.46|181.77|
|16|0.3|71|84.58|48.50|56.47|41.87|31.35|1.705248|2.177537|18402.25|183.23|

测试2488张，类别支持从10到730不等；train/val每类60/10，因此train/val accuracy等于macro recall，
test accuracy不等于macro recall。最高总体准确率69.7749%，没有达到70%，
该4层残差组宏平均召回50.0628%，不能把大类占比带来的总体准确率当均衡泛化。

同depth rho=.3相对rho0的test accuracy差（pp）：

- L2：accuracy -1.69，macro recall +2.28，balanced NLL -0.103295。
- L4：accuracy +3.38，macro recall +8.88，balanced NLL -0.169605。
- L8：accuracy +1.00，macro recall +5.85，balanced NLL -0.142517。
- L16：accuracy -5.99，macro recall -0.95，balanced NLL +0.046764。

残差并非随深度增加总有收益：2层总体accuracy下降而macro/NLL改善；16层总体与macro下降，
balanced NLL变差。4/8层观察到部分指标改善；单seed17不能证明跨种子稳定收益或光学机制因果。

所有验证70%门槛失败。选中train-val差：2层14.83/19.83pp、4层30.58/36.00pp、
8层40.75/44.17pp、16层30.92/36.08pp；尤其深层拟合训练较好但泛化差距明显。
rho0各depth末十val NLL仍改善；rho=.3的8/16层末十均值较前十恶化0.104%/0.038%、接近平台。
有限100轮不证明充分收敛；16层rho=.3按NLL选E71，不能按已见test或末轮accuracy反选。

数据按源图片数≥80的25类池、seed20261012固定选20类，3888张/组；
train1200、val200、test2488，旧Caltech50所有train/val固定原角色，进入新test数量0。
这是计数预筛的Caltech101二十类子集，沿先前验证探索的任务调整，不能冒称完整101或纯独立选任务，
也没有物体个体／摄影者独立证据。全幅RGB150保留背景，无CNN/Linear/pretrain；
九专家原ASM/OEO及rho仅expert/global，router无残差。所有depth同固定20个32窗、同seed及公共超参。

预算每组base100/extra0；L2复用已完成原run，不重训；旧L16两轮计时不作正式臂。
新六臂有界四卡队列拟合墙钟18416.87秒（约5.12小时）；这不含之前L2、最终测试及CPU审计。
4层micro4，8/16层micro1，原数值门槛未放宽；均轮数字为对应实际history均值，
不是所有depth同速，也不是硬件实验台采集速度。

CPU审核仅读保存PT/CSV/history/cache标签，没有完成臂train/val复推；
最终审核ID、标签、argmax、概率、support、完整CM、NLL、balanced NLL、macro recall/F1全部通过。
全部八组test_evaluations=1、test_used_for_selection=false，未选/raw/last不测，失败没有重试。

[八组CSV](final_summary.csv) · [每类支持/召回](per_class_support.csv) · [train/val/test完整CM](confusion_matrices.csv) · [曲线CSV](curve_data.csv)

[完整train/val曲线及单test点](learning_curves_all.png)（[SVG](learning_curves_all.svg)、[PDF](learning_curves_all.pdf)）；
[获选汇总](selected_summary.png)（[SVG](selected_summary.svg)、[PDF](selected_summary.pdf)）；
[宏平均召回汇总](macro_summary.png)（[SVG](macro_summary.svg)、[PDF](macro_summary.pdf)）。
单seed没有误差条；所有图已视觉检查。测试CM图按行归一化着色，格内为真实计数：

- L2：[PNG](test_confusion_L2.png)／[SVG](test_confusion_L2.svg)／[PDF](test_confusion_L2.pdf)。
- L4：[PNG](test_confusion_L4.png)／[SVG](test_confusion_L4.svg)／[PDF](test_confusion_L4.pdf)。
- L8：[PNG](test_confusion_L8.png)／[SVG](test_confusion_L8.svg)／[PDF](test_confusion_L8.pdf)。
- L16：[PNG](test_confusion_L16.png)／[SVG](test_confusion_L16.svg)／[PDF](test_confusion_L16.pdf)。

[最终CPU审核](final_audit.json) · [全八锁](global_selection_lock.json) · [测试父身份](test_suite_metadata.json) · [进程/GPU释放收据](gpu_release_receipt.json)。

全八global lock SHA256：746ca2f6cb9ab90281b67572e634b6d57969ffecd787d7cd9b24e0c00f27a904。
训练67文件实际内容等同原aaccbeb来源；新六臂记录训练Git52ab4d58c，L2保留原训练Git。
报告HEAD更新没有改变训练来源。旧COIL/Caltech50/Leaf/Butterfly/FishNet/Mango/L12全部保留，不重训重测。
