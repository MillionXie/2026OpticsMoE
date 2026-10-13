# 3层残差150轮与部署导出审核（2026-10-13）

L3rho.3完整150轮完成，原子2437634实际退出；GPU1自动接续L2rho0。
CPU verify_arm全部PT/config/source/data/初相位身份、10相位150轮有限非零梯度、Adam86E/scheduler、独立每轮order/affine/MixUp（含12尾）、保存预测ID/标签/argmax/CM/support/macro/NLL核验通过。
没有模型前向、test数组、选模锁。运行7源当前/原4406Gitblob一致；只发布独立导出/完成工具，训练源不改。

首次最低验证balanced NLL选E150：train47.8134%/val35.6890%，train宏召回56.4583%/val宏召回41.2584%，val balanced NLL2.0373893024，差距12.1244pp。总10552.10s，150轮上限停止。末十val NLL相对前十均值仍改善1.1402%，存在泛化差距但不能称充分收敛或单凭差距判严重过拟合。

[真实train/val曲线](learning_curves.png)、[SVG](learning_curves.svg)、[PDF](learning_curves.pdf)、[CPU审核及实际权重导出收据](cpu_saved_audit.json)、[曲线CSV](curve_data.csv)、[全CM](full_confusion.csv)、[support](per_class_support.csv)。三个残差组均完成；仍等待全部无残差组后全六CPU审核/全锁及首次单test，不根据test调参。

## 独立部署导出修正与实际核验

旧导出把设备尺寸误标canonical，已修正独立工具，标准478×478与设备相位1920×1200分别保存；振幅设备1920×1080。联合amplitude/phase才能实现rho，不能只上相位图。
`export_caltech_deploy234.py`直接调用冻结physical_fields编码，记录每张振幅active与illuminated支持域的mean/p99/max、zero/saturation，绝无额外峰值拉亮；静态包保存幅度乘子、固定探测/router mask、类映射、config及源/权重/SHA合同。
Linux真实10项CPU检查全部通过：四项claim顺序/失败保留/拒绝重复与六种模型标准/设备BMP、量化联合光场、零保持、固定mask/SHA/无额外推理。
已训练L4rho.3 E150权重也已CPU静态导出核验，五次捕获的相位标准/设备尺寸正确，synthetic黑输入完全保持零。无数据集图像推理、无GPU进程、无设备打开；静态导出证明不代表实拍精度验收。
[10项检查收据](export_protocol_checks.json)。相位/振幅LUT、ROI/方向/曝光/gain及新硬件会话仍需后续实拍校准，本次未连接实验室。

PNG与PDF已实际视检；PDF用Poppler渲染单页，坐标轴/图例/获选点和排版通过，见[视检证明](visual_inspection.json)。
