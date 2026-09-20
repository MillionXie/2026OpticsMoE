# t08_abo_image_text_retrieval 复现说明入口

[Baseline????????](BASELINE_METHODS.md)?????baseline??????????????????????2026-09-09?

本目录集中保存baseline及主方法的可复现性证据；本次只建立入口，**尚未进行本任务的新一轮复现**。
当前任务结构和已有结果见 [任务README](../../README.md)。不得因为存在本文件就声称已复现。

后续每个正式结果需在这里记录：

1. baseline定义，冻结/训练的参数，预处理、输出头与主方法差异。
2. 原始数据版本、train/test清单与SHA256、标签生成方法、指标与选模口径。
3. 源码commit、完整命令、依赖环境、模型及checkpoint的SHA256和获取位置。
4. 固定权重复评与从头重训分别报告；标明样本数、seed、运行ID、结果和误差。
5. 速度/能耗的硬件、计时边界、功率积分口径；未测的不得填估计值冒充实测。

原始日志及逐样本结果留在本任务runs，文档只引用。参考 [SALICON复现说明](../../../t03_saliency/reports/reproduction/README.md)。
# 2026-09-20 纯文搜图：独立于下方图搜文的新增协议

见任务README顶部“纯文搜图可行性”。冻结Qwen3-VL-Embedding-2B，100标题查询、2400张test图图库，
24个同SKU正例/查询；无微调、无查询图片、不改变候选库。源码`1e218991a`，RTX4090 GPU4，进程已退出。
run=`runs/simulation/text_to_image_frozen_20260920`，含report.json、逐标题预测和embeddings.pt。
64D动态/白边Hit@1=65%/66%；2048D动态/白边=82%/80%。完整命令、身份SHA和环境见report.json。
这是已登记目录的纯标题检索，未来用相同标题训练不代表未见文本泛化；尚未训练本协议光学模型。
