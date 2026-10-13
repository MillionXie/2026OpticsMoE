# 主干4层配对150轮保存结果审核（2026-10-13）

L4rho0/.3各完整150轮，原子2775881/2437562均实际退出，项目GPU2已释放。运行7源当前/原4406Gitblob一致。
新无残差CPU verify_arm核全部PT/config/data/源/初始相位/11张相位150轮有限非零梯度、Adam86E/scheduler、独立150轮order/affine/MixUp含12尾、保存预测ID/标签/argmax/CM/support/macro/NLL通过；残差既有审核与PT/history SHA仍一致。
配对初相位身份、源、150轮order/transforms完全一致，配置只差rho（含contract.rho派生项）；没有模型前向、test数组、部分锁或重复训练。

|rho|获选epoch|train|val|val宏召回|val balanced NLL|train-val差|
|---|---:|---:|---:|---:|---:|---:|
|0|147|52.8426%|48.7633%|49.3341%|2.010079976|4.0793pp|
|.3|150|57.2157%|46.6431%|48.6914%|1.869108924|10.5726pp|

残差train高4.3732pp、val低2.1201pp、val宏召回低0.6427pp，但balanced NLL低0.140971；指标不一致，不能只选有利指标宣称精度收益。残差差距更大，验证曲线末段趋平台，存在过拟合迹象；仍不能称充分收敛或单种子因果结论。末十验证NLL均值相对前十改善1.7300/1.0047%。总8463.16/8537.59s。

[曲线PNG](learning_curves.png)、[SVG](learning_curves.svg)、[PDF](learning_curves.pdf)、[完整配对审核](cpu_saved_pair_audit.json)、[汇总CSV](selected_summary.csv)、[曲线数据](curve_data.csv)、[全CM](full_confusion.csv)、[support](per_class_support.csv)。Train曲线为每5轮及端点的unaugmented评估，val每轮，空心圆只标验证NLL获选点，无test点。

仍等待2/3无残差组完成、真实CPU父/子退出及全六审核锁后各EMA首次单test；75%最终test目标未验收，不利用test修改方案。旧test已见及val类别筛选，属于探索性开发。布局/BMP/正则均变化，不能与旧run作单因素正则因果比较。

PNG实际视检、PDF用Poppler渲染并实际视检均通过；[SHA及视检记录](visual_inspection.json)。
