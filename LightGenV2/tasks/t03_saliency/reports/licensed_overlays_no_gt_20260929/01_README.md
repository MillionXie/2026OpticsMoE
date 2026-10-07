# 预测叠加展示包（不含GT）

先看00_preview_all.png；每图目录包含原始JPG、预测叠加PNG、纯预测热图、原始预测NPY、指标和双栏预览。
本包无GT图、GT数组、fixation文件或含GT的旧预览，也未嵌套旧结果包。

叠加采用inferno色系、65%亮度原图和随预测显著程度变化的透明度；所有图统一公式。按预测峰值缩放仅用于显示，没有修改模型输出或重新计算指标。

CC/PCC、SIM、NSS、KLD、AUC、MAE为此前使用GT计算的原始指标，不代表无GT评估。只有715属于public-test，其余五张是训练样例，论文中请区分。
署名来源见ATTRIBUTION_AND_REVIEW.md；使用叠加图请注明saliency overlay added。
