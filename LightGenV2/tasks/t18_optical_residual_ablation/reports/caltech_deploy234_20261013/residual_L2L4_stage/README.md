# 2/4层残差150轮阶段审核（2026-10-13）

本阶段只读已保存的train/val结果，尚未完成全部六组；未推理模型、未读取test数组、未创建任何选模锁。
L2/L4 rho.3均完成150轮，原GPU子2437706/2437562实际退出，队列自动接续无残差组。
训练7源仍为4406c010f95f60a959d732ad2200eaa34657519a，当前文件和原Gitblob逐项一致。
CPU verify_arm核PT/config/source/data/初相位身份、6/11相位每轮有限非零、Adam86步/轮、scheduler、独立全部150轮顺序/仿射/MixUp（包括12尾），保存预测ID/标签/argmax/CM/support/macro/NLL全部通过。
L4原E7接续历史完全一致，无完成轮次重播；检查没有模型前向调用。

|主干/rho|获选epoch|train|val|val宏召回|val balanced NLL|train-val差|
|---|---:|---:|---:|---:|---:|---:|
|2/.3|150|36.2245%|36.0424%|36.1298%|2.412189|0.1821pp|
|4/.3|150|57.2157%|46.6431%|48.6914%|1.869109|10.5726pp|

均因150轮上限停止；总时间2层8351.82s、4层8537.59s（包括接续前缀）。末十轮验证NLL相对前十轮均值改善0.4773/1.0047%。
4层存在泛化差距，验证NLL末段仍改善，不能称充分收敛或仅凭差距判严重过拟合。2层训练也低，拟合/优化仍不足，不断言容量上限。
4层最终test≥75%目标尚未验收，不用test修改当前方案。与旧9专家/旧编码的差异同时包含布局和训练变化，不作正则化单因素因果比较。

[真实train/val曲线](learning_curves.png)、[SVG](learning_curves.svg)、[PDF](learning_curves.pdf)、[完整审核收据](cpu_saved_audit.json)、[汇总CSV](selected_summary.csv)、[曲线CSV](curve_data.csv)、[全CM](full_confusion.csv)、[support](per_class_support.csv)。

仍待所有六组拟合、全六CPU审核和排他全锁，之后每组获选EMA首次单test与最终CPU图表交付。此前观察过旧test且类别用val筛选，结果属于探索性开发。

PNG已实际视检，PDF已用Poppler渲染单页并实际视检，坐标轴/图例/选中点/排版清晰；SHA见[视检记录](visual_inspection.json)。
