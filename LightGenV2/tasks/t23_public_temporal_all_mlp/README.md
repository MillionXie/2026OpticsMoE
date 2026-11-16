# LightGen-II MLP global reference

本任务基于原定开源版`LightGen2/weights/LightGen2.pt`，SHA256为
`c6593a2b801d74698e4065cd63dcc3ba42fd24c8f571d4f8f3b19bdd8e41701e`。
独立推理工程为仓库根目录`LightGen2_MLP`。

## 计算流程

视频帧经冻结Qwen patch和位置处理得到每帧49×1024特征，经过原192维适配器、
Top2 router和四个线性专家，进入vision global MLP。
vision global对每帧展平为9408维，使用`9408→32→9408`两层全连接和中间ReLU，
恢复49×192，再由原mean/max merger得到四个视觉token。
四个视觉token与原38个文本token拼接，经过language router和线性专家后，
language global使用`8064→32→4032`，恢复42×96，送入原32单元、五等级读出头。
两支MLP分别在所属分支的全部视频和帧之间共享参数。

输入沿用2250 TRAIN、558原TEST的身份、四帧采样及448×448图像处理。
原TEST用于开发选模，结果标记为开发评估。
每批16条不同视频，尾批14条真实视频及2条循环填充，指标按有效位排除填充。

## 作者侧训练与核验

训练采用母版在TRAIN上的特征及分数蒸馏，并结合TRAIN人工MOS监督。
Qwen前端、prompt特征、视觉适配器和vision router/experts由母版继承。
MLP global使用新初始化，训练与后续微调的参数范围由配置和run记录。
核验目标为SRCC在0.790–0.799、专家总体及逐槽位份额、槽位换位一致性，
以及两个global的独立贡献。

作者代码位于本任务，训练产物写入`runs/simulation`，临时核验写入`runs/smoke`。
独立推理工程使用单个PT和原558条视频帧NPZ，执行入口为`infer.py`。
结果包含逐视频预测、指标、预测散点图及输入帧预览。
