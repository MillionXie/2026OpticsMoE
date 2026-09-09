# T06｜LGVQ视频质量评价：baseline复现说明

整理日期：2026-09-09。主要baseline为**冻结Qwen3-VL-2B-Instruct＋五质量词评分头**。Spatial和Temporal分别训练、分别评价；不把两个质量目标或不同帧数的结果混用。

## 数据与视频输入

使用固定LGVQ清单，2250条视频训练、558条视频测试；监督为每视频一个Spatial MOS或Temporal MOS。Spatial baseline每视频取4帧，Temporal baseline分别建立4、9、16帧三个实验。抽帧位置覆盖视频时长的10%至90%：4帧使用0.10、0.37、0.63、0.90，9帧为0.10至0.90间隔0.10，16帧在同一区间等距采样。按记录的OpenCV随机定位方式解码，每帧取以短边65%为边长的中心正方形，再缩放至448×448。

视频帧与任务prompt共同经Qwen原生processor输入完整视觉和语言Transformer，主干权重全部冻结。Temporal prompt为：

> Please evaluate the temporal quality of this video and rate it using one of the following five levels: Excellent, Good, Fair, Poor, or Bad.

Spatial使用相应spatial quality指令，其他措辞以该任务配置为准。不同帧数各自提取特征并训练评分头，不能仅改变测试帧数而沿用另一帧数的头。9帧输入还须保留原生temporal patch的补帧行为。

## 五质量词评分结构

取完整Qwen最后一个有效位置的2048维隐藏表示。读出由Bad、Poor、Fair、Good、Excellent五个质量词对应的输出权重行组成，从预训练模型对应行初始化，再在任务训练集上优化，共5×2048＝10240个可训练参数，不另加bias。

只使用训练集MOS最小值和最大值，将分数范围均分为五个区间，产生五等级训练标签；另在该范围内设置五个等距分值。五路logits经softmax得到概率，以概率对五个分值加权求和，输出连续MOS。即 `预测MOS = Σ p_k s_k`。这属于监督训练的质量读出，而不是原版Qwen零样本生成评分；测试集不得用于确定分数区间或重新拟合预测映射。

## 训练、复现步骤及评价

先固定视频清单、目标MOS字段、Qwen版本和抽帧配置。完整执行冻结Qwen提取训练/测试隐藏特征；用缓存训练五行头可以节省重复骨干计算，但在线推理仍须执行完整Qwen。只在2250条训练视频上更新头，使用AdamW、50epoch、batch512、学习率0.001、weight decay 0、seed42和余弦学习率衰减，损失为五等级硬标签交叉熵。

每轮用558条test的连续MOS计算SRCC，保存最高test SRCC对应的头。最终报告SRCC、KRCC、PLCC、RMSE和MAE，直接比较预测MOS与真实MOS；当前属于**无独立validation、test参与选模**的协议，应明确披露。

统一入口为本任务 `quality_token_resolution.py`，依次执行特征提取、头训练和完整测试。Temporal配置为 `configs/baselines/qwen3vl_quality_tokens_r448.yaml`，Spatial配置为 `configs/baselines/qwen3vl_spatial_quality_tokens_4f_r448.yaml`。模型路径、manifest和新run目录由复现者设置，其他输入与标签规则保持一致。

## 其他版本与速度测量

历史另有冻结Qwen＋ `Linear(2048,1)` 标量头、以及不同帧数/分辨率的参照；这些是独立baseline，不能合并成一个无标签的大模型分数。需要重现历史Linear头时，应使用该版本原训练配置和权重，不能套用五等级CE。

已有448px五质量词5090D测量采用batch1、每种方案/帧数独立加载一次模型，**无显式warmup，全部558条test且包含首条**。时间从第一个原生Vision block计至MOS就绪，包含原生主干及评分路径；该历史实现还执行全词表投影。解码、裁剪、processor、H2D及patch embedding另列。它与根协议的warmup50热稳态测量不同，需按真实run注明；缓存头推理时间不能代替完整大模型时间。

结果来源与输入token几何见[原始baseline报告](../paper_results/qwen3vl_quality_token_baseline_r448/README.md)，版本和能耗口径限制见[审查说明](CHECK_20260908.md)。本任务不预设一个已经完成的统一matched D2NN对照。
