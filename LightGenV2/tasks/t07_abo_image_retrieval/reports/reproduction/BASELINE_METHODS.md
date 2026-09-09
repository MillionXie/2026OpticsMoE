# T07｜ABO相似商品图搜图：baseline复现说明

整理日期：2026-09-09。本文对应服务器独立分支的 **ABO similarity-10冻结Qwen baseline**；不是T08的图像到标题检索。该实现尚未完整整合进当前本地任务入口，源码身份和证据见[审查说明](CHECK_20260908.md)。

## 数据与模型

数据包含10个商品类别、200个商品，每商品12张视角图像。按商品身份划分train/validation/test为120/40/40个商品，对应1440/480/480张图片；每类分别有12/4/4个商品，商品身份不跨split。gallery使用120个train商品，query使用40个test商品的全部480张图片。对任意query，与其同类别的12个gallery商品都视为相关，因此任务是**未见商品的同类别相似商品检索**，不是识别同一个商品实例。

采用Qwen3-VL-Embedding-2B，完整执行冻结的原生视觉和语言Transformer，不增加训练头、不微调。图像按该分支流程裁剪缩放至224×224，经同一processor编码。gallery和query使用相同指令：

> Represent this catalog product image for category-aware visual similarity retrieval.

取最后有效token的完整2048维向量并L2归一化。模型全部冻结，因此没有优化器、训练epoch或checkpoint选择；train在这里用于构建gallery，并非用于反向传播，validation也不参与这个冻结baseline。

## 复现流程与指标

先确认商品级划分和类别映射，再提取1440张gallery图像的embedding。每商品的12个归一化特征取均值后再次L2归一化，形成120个商品centroid。对每张test图片提取单图embedding，与120个centroid计算余弦相似度并完整排序。

分别报告Hit@1/5/10、Precision@5/10、相关商品集合Recall@5/10，以及mAP@10、NDCG@10。Hit@K衡量前K项是否至少含一个同类商品；Precision@K为前K项同类商品数除K；集合Recall@K为该数量除12。原报告的 `R@K` 实际使用Hit@K定义，复现和写作中应明确命名，不能把两种Recall口径互换。另有十个类别原型的路由准确率诊断，不用于预先限制候选库。

实现位于分支 `benchmark/t07-abo-a100-baseline`，原run源码commit为 `354c53e54b9d26b1d7835a3ba51b60929425baa8`，入口为该版本的 `LightGenV2.tasks.t07_abo_image_retrieval.baseline_a100`。使用相同数据包、Qwen版本和新run目录即可执行全量评价；大数据清单、SHA及服务器位置保存在本目录证据中，不能仅将命令复制到缺少该入口的当前checkout运行。

## 测量与适用范围

已有记录使用A100-PCIE-40GB、BF16、batch1，预热50次后按类别均衡选200条query计时。计时包含首个Vision block以后的完整Qwen、归一化、商品排名及类别原型诊断，排除gallery预计算和图像前处理；A100结果不能当作5090D结果。

当前已核验的是冻结电子baseline，不在此虚构匹配的光学/D2NN训练流程。原归档还有一处报告SHA转录错误，复现应采用[机器证据](EVIDENCE_20260908.json)中实算的正确hash。
