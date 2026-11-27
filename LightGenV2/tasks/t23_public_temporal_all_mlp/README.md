# LightGen-II MLP global reference

本任务基于原定开源版`LightGen-II/weights/LightGen2.pt`，SHA256为
`c6593a2b801d74698e4065cd63dcc3ba42fd24c8f571d4f8f3b19bdd8e41701e`。
独立推理工程为仓库根目录`LightGen-II-MLP`。

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
结果包含逐视频预测、指标及输入帧预览。

## 当前候选与核验结果

训练记录为`runs/simulation/global32_20261117`。两支global从新初始化开始，
前40轮训练global，随后同时微调原32单元读出头和language router末层偏置。
其余继承张量保持冻结。使用第102轮通过精度和路由均衡核验的候选。
该轮公开权重为`LightGen-II-MLP/weights/LightGen2.pt`，SHA256为
`0a6e4081eff1aabba8852b06c301bf2540319e1b0134a99a4def3130fab51c1a`。

完整视频帧NPZ经正常推理入口得到558条唯一视频结果，SRCC为0.7939403761。
对应缓存前端特征复算为0.7939367845，浮点前端计算带来约0.0000036差异。
558条原TEST用于该候选的开发选模，均属于开发评估。

| 分支 | Expert 1 | Expert 2 | Expert 3 | Expert 4 |
| --- | --- | --- | --- | --- |
| vision | 26.79% | 25.09% | 23.34% | 24.78% |
| language | 23.21% | 27.96% | 23.48% | 25.36% |

两支均出现全部六种Top2组合。16次循环槽位换位的最大逐视频MOS差为0。
每个视频的MLP仅处理自身token，参数在视频之间共享。

固定其余权重，将vision global替换为恒等映射时SRCC为0.2702833756。
将language global替换为固定相邻通道两两均值以维持42×96输出时，为0.7346179060。
每支三个随机global替换也均降低SRCC。这些记录评估所选权重对各模块的依赖。

两支MLP输出同时加入高斯特征噪声，标准差分别取固定TRAIN教师输出RMS的5%和10%。
每档10个固定种子，5%时均值为0.7931247237，最低为0.7917025257。
10%时均值为0.7921840605，最低为0.7889275817。
该实验的噪声定义及逐次结果位于`runs/smoke/validation_export_20261117/audit`。

## 推理工程与打包

`LightGen-II-MLP`包含推理入口、模型、结果绘图、单个PT、558条视频帧输入、
独立MOS参考文件及PDF说明。运行`python infer.py`完成全部558条视频推理。
原`LightGen2`发布工程及33个受保护文件的SHA核验均保持一致。

MLP说明手册沿用`LightGen-II/README.docx`的版式、章节顺序、两张Files表、
灰底命令、图注及页码。编辑用`README.docx`保存在独立工程中，发布包使用`README.pdf`。
绘图模块统一命名为`visualize.py`。作者运行`tools/write_guide.py`生成Word文档，
通过`tools/export_guide.ps1`导出PDF，并运行`tools/verify_guide.py`核对表内文件与实际目录。

作者侧运行`python -m LightGenV2.tasks.t23_public_temporal_all_mlp.build_lab_package`，
生成`releases/LightGen-II-MLP.zip`。压缩包按白名单收集，文件SHA与核验信息保存于包外。

2026-11-27按作者要求对齐非MLP版的文件结构与说明手册，输入读取并入`infer.py`，
取消散点图生成并清理旧输出。新增`refine_readout.py`在原2250条TRAIN上微调五等级读出头，
冻结其余模块，目标为558条开发评估SRCC达到0.795–0.799。
