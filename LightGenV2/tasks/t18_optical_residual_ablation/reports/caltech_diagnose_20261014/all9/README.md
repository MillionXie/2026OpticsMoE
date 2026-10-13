# 全九组共同40轮消融：CPU保存审核完成

旧20类train1372/val283、main4/rho.3、seed17、macro16/micro1/尾12、EMA.95和fixed lr.002。
八个当前图变体同初始化；legacy是多因素旧9expert整套参考，不能拆成单一布局因果贡献。
九组全部40轮，val balancedNLL最低首次选EMA，无test或正式选模锁。

| 因素 | Epoch | Train accuracy | Val accuracy | Val macro recall | Val balanced NLL | Train−val gap | Val零召回类 |
|---|---:|---:|---:|---:|---:|---:|---:|
|base|40|33.75%|28.98%|34.17%|2.405106|4.77pp|7|
|unquantized|40|34.69%|30.39%|35.08%|2.385514|4.31pp|7|
|fullfield_ln|40|33.82%|32.16%|32.00%|2.482380|1.66pp|1|
|rgb_tiles|40|45.77%|44.17%|42.76%|2.207855|1.60pp|2|
|dense_router|40|46.87%|47.00%|48.17%|2.041299|-0.13pp|2|
|preserve_global|40|43.88%|45.94%|45.11%|2.124360|-2.06pp|4|
|mixup02|40|36.37%|31.80%|36.60%|2.360889|4.57pp|7|
|mixup06|40|31.20%|26.15%|29.28%|2.431283|5.05pp|9|
|legacy_reference|39|65.82%|50.88%|48.43%|1.923981|14.93pp|1|

RGB输入、连续四专家路由、保留478中间全局场分别显著改善当前共同预算的拟合/验证。
单独关闭量化仅val+1.41pp，不能将浮点不可部署模型作为修复。
MixUp.2相对基线val+2.83pp、macro+2.42pp、bNLL降低.044217；MixUp.6的accuracy/macro/NLL
都更差，并出现更多零召回。仅加大正则没有修复结构信息问题，也不代表长预算过拟合已经判定。
整面LN虽然少零召回，macro/NLL比基线差，不能只看小gap或零类数量选方案。
旧9expert参考选E39：train65.82%、val50.88%、gap14.93pp，拟合明显强于新物理图，
仍有crab验证零召回，且仍有泛化差距；不直接套入当前四expert部署合同。

所有组至少一个验证零召回类，多数预算末段仍改善，不能称充分收敛或过拟合已解决。
不存在可直接进入正式六组的合格候选。75%最终test目标尚未验收，不能把单因素增益相加预测组合。
部分因素恢复旧Faces/Motorbikes/airplanes说明类失效与结构交互，不能不断删类掩盖问题。
人工指定新20已经固定，不重筛；因旧test已见和类别排除参考旧结果，仍属开发实验。

下一步组合有证据支持的RGB输入、dense4和保留478全局场，公开作为一项复合修正，
保留ABO几何、8bit联合振幅相位、固定非线性和纯光能量读出，不加CNN/Linear或预训练。
组合交互未验证；先固定profile/门槛，再做最多三个4层rho.3正则候选，<=150轮/14400s。
2/3层尤其末端expert要独立确认；逐类/验证/物理合同支持后才正式六组，不用test选方案或无限搜索。

[全部accuracy曲线](all9_accuracy_curves.png)、[SVG](all9_accuracy_curves.svg)、
[全部NLL曲线](all9_nll_curves.png)、[SVG](all9_nll_curves.svg)、
[逐类验证召回及support](all9_validation_class_recall.png)、[SVG](all9_validation_class_recall.svg)。
黑点/虚线只表示实际train端点，蓝线每轮val，红星为val NLL选中EMA；
空心菱形是获选EMA的保存train CSV指标（legacy E39），不要误读为E40训练端点。
三PNG已实际视检，统一坐标、星标和图例可读，无test点或单seed误差条，不填造逐轮train数据。
[曲线CSV](curve_points.csv)、[选中汇总](selected_summary.csv)、[train/val各类support/recall](class_support_recall.csv)。

[全九CPU审核](cpu_audit.json)、[历史/fit SHA及保存曲线](saved_curve_evidence.json)、
[实际父子退出/GPU释放](completion_gpu_release.json)。完整40/PT/config/原Gitblob/data/相位梯度、
Adam3440步、独立orderaffineMixUp含尾12、原17/17/12 prefix逐项一致、
预测ID标签argmaxCMsupport宏NLL通过，model_forward0/test数组0/test0。
九GPU子及CPU父实际退出，项目GPU2/5/6释放，其他项目未改；121来源e4cb仍冻结。
原数据/best/last/历史/失败收据/test保留，不重复已完成诊断或旧test。
