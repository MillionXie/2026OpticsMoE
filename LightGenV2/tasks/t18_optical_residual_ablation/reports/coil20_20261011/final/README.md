# COIL20四深度验证选模探索性评估

两层达到用户速度和70%精度目标：每轮约11秒，30轮约5分半，首次单次test为79.44%／81.94%。
完整20物体／1440张原生图，无挑类或挑图。任务是已知实体的留出视角识别，
不能冒称未见实体、跨地点独立或20个多实例语义类泛化。作者数据身份与原始许可边界见[上级报告](../README.md)。
固定相邻6视角分块加精确像素连接，train960／val120／test360，每类48／6／18；
块边界近邻、采集依赖仍存在。灰度复制RGB／bicubic150／原固定振幅编码，九专家原ASM／OEO，
无新增CNN／Linear／pretrained／电子学习head。rho只expert／global，router无残差。

|层数|rho|获选方案／epoch|train %|val %|test %|test NLL|test macro F1 %|
|---|---|---|---|---|---|---|---|
|2|0|base／30|94.79|84.17|79.44|.858299|77.16|
|2|.3|base／30|94.27|85.00|81.94|1.023200|80.62|
|4|0|base／30|98.44|100.00|91.11|.494340|90.76|
|4|.3|base／30|98.13|94.17|88.89|.889652|88.58|
|8|0|base／30|99.06|97.50|90.28|.481139|89.86|
|8|.3|base／30|97.71|88.33|78.61|1.693863|77.59|
|16|0|rawinit .1／10|85.94|70.83|60.83|1.277814|56.87|
|16|.3|rawinit .1／10|91.15|82.50|71.11|1.044680|68.52|

三划分类别均衡，因此macro recall等于accuracy、balanced NLL等于NLL；
精确train／val／test的accuracy、macro recall／F1、NLL／balanced NLL、参数、预算及PT SHA在
[final_summary.csv](final_summary.csv)，逐类支持／正确数／recall在[per_class_support.csv](per_class_support.csv)。
可训练参数2／4／8／16为449848／889696／1769392／3528784；固定buffer不计参数。

全部每轮只验证EMA balanced NLL最低选best，精确同分保留首次，不能按accuracy反选。
2／4／8各完整base30轮。16层base30原rho0选E30 train95.94%／val86.67%，
rho.3选E26 train8.65%／val9.17%，原始学习停滞负结果见[base审计](../L16/pilot_audit.json)。
按预登记仅一次成对raw初始化范围±1→±.1重启10轮，双方同seed17、初始参数、
10轮order／augmentation及其余配置一致，模型／rho／数据／类别／损失／lr没有改动。
重启审计与真实曲线在[L16_adjust](../L16_adjust/pilot_audit.json)。
两rho最佳val NLL均值base1.809468、重启.895909，选整个重启pair；
无残差原base验证更好也没有给它单独换回旧方案，test不参与方案选择。
16每臂消耗base30+重启10，获选权重训练10轮，不是累计epoch40；未选方案永不test。

残差test accuracy变化2层+2.50pp、4层−2.22pp、8层−11.67pp、16层+10.28pp。
2层残差accuracy更高但NLL更差，不能宣称所有指标优势；4／8层负结果如实保留。
16层初始化调整与额外预算使它不能和其它深度作纯深度因果比较，也不证明初始化就是唯一停滞原因。
这里没有多seed误差条或任意权重逐像素数值等价证明。

过拟合／欠拟合／未收敛必须一起看：获选trainval差2层10.63／9.27pp、4层−1.56／3.96pp、
8层1.56／9.38pp、16层15.10／8.65pp，16无残差及两层存在明显泛化差距。
预算末端验证NLL仍改善，2／4／8最后十轮均值较前十轮降低16.39%–48.72%，
16重启后五轮较前五轮降低36.36%／46.87%；所有获选best均在预算末轮。
因此不能称已充分收敛，不能仅train高就断严重过拟合，也不能低train就断容量不足。
16原残差低train／val是学习停滞；一次调整恢复学习，但10轮仍未证收敛且留出精度有限。
遵守用户速度要求和已登记上限，不再追加候选、训练轮数或看test换权重。

最后一个方案选择完成后，保存全部pair／selection／global lock，才首次加载test。
[global_selection_lock.json](global_selection_lock.json) SHA
4b0a2c00278673052f18ab988b29b97272a4e389b7d86a62fe138067cc7edc3c。
全8获选EMA各test_evaluations=1、test_used_for_selection=false，history没有逐轮test。
每臂test_started在读取test数组前独占写入，失败禁止自动重测；本次全部成功，没有重测。
原数据、best／last、train／val／test预测保留在各run，不复制PT／cache／原图进Git。
最终run：`coil20_s17_L2L4L8L16_val_selected_once_20261011`。

[CPU最终审计](final_audit.json)仅读保存结果／PT／CSV／缓存标签，未做光学推理：
全8锁、首次标记、选轮、原发布Git来源、参数、三划分ID／标签／argmax／支持／CM／recall／NLL／macro全部过。
[测试启动身份](test_suite_metadata.json)与[GPU释放收据](gpu_release_receipt.json)保存；
CPU父、两调整子和八测试子均实际退出，GPU1／2／4本项目实际释放，不杀其它项目。

真实图及数据：

- 全部候选train／val曲线及获选单个test星点：[PNG](learning_curves_all.png)、[SVG](learning_curves_all.svg)、[PDF](learning_curves_all.pdf)、[曲线CSV](curve_data.csv)。
- 获选三划分accuracy／NLL比较：[PNG](selected_summary.png)、[SVG](selected_summary.svg)、[PDF](selected_summary.pdf)。

PNG视觉检查通过，单seed无误差条、不制造逐轮test曲线。训练来源版本、各候选PT SHA和预算
详见CPU审计及全8锁，逐原metadata archived Git blob核验，不拿交付HEAD冒充训练版本。
