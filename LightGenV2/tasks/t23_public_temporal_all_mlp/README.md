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

训练记录为`runs/simulation/global32_retrain_20261127`，源码提交为
`0ab0f7a0ea326e05dd13b414bb55932f3e12a77b`，配置为`configs/global32_retrain_20261127.json`。
两支global仍为32单元的两层MLP，末端仍为32单元的五等级读出头。两支global从新初始化开始，
前40轮训练global，随后同时微调读出头和language router末层偏置，其余继承张量冻结。
选定第160轮的不可变快照，完成路由、槽位、global替换、噪声及完整RGB入口核验后发布。
该快照位于`runs/smoke/mlp_release_alignment_20261127/download/selected_candidate.pt`。
训练于200轮停止；最后一轮的best/last保留用于作者追溯，不替换已核验的发布快照。
正式单PT为`LightGen-II-MLP/weights/LightGen2.pt`，SHA256为
`22429a5adbbfe5147f82b63314f2c129dc2f91f59e1027d2dc37e54322383153`。

完整视频帧NPZ经服务器正常推理入口得到558条唯一视频结果，SRCC为0.7965075506，
按两位小数报告为0.80。对应缓存前端特征复算为0.7965033720。
558条原TEST用于开发选模，属于开发评估；不宣称独立测试泛化。
初版0.7939403761的原权重及文档在本轮`runs/smoke/.../before`保留，
原训练记录仍为`runs/simulation/global32_20261117`。

| 分支 | Expert 1 | Expert 2 | Expert 3 | Expert 4 |
| --- | --- | --- | --- | --- |
| vision | 26.79% | 25.09% | 23.34% | 24.78% |
| language | 23.21% | 27.60% | 23.21% | 25.99% |

两支均出现全部六种Top2组合。16次循环槽位换位的最大逐视频MOS差为0。
每个视频的MLP仅处理自身token，参数在视频之间共享。

固定其余权重，将vision global替换为恒等映射时SRCC为0.2622559437。
将language global替换为固定相邻通道两两均值以维持42×96输出时，为0.7600851378。
每支三个随机global替换也均降低SRCC。这些记录评估所选权重对各模块的依赖。

两支MLP输出同时加入高斯特征噪声，标准差分别取固定TRAIN教师输出RMS的5%和10%。
每档10个固定种子，5%时均值为0.7952906322，最低为0.7941133455。
10%时均值为0.7943480175，最低为0.7918347617。
该实验属于特征噪声试验，不等同于光路CCD测量。逐次结果位于
`runs/smoke/mlp_release_alignment_20261127/download/candidate_validation`。

## 推理工程与打包

`LightGen-II-MLP`包含推理入口、模型、结果绘图、单个PT、558条视频帧输入、
独立MOS参考文件及PDF说明。运行`python infer.py`完成全部558条视频推理。
非MLP的`LightGen-II`工程及本轮33个受保护文件的SHA核验保持一致。

MLP说明手册沿用`LightGen-II/README.docx`的版式、章节顺序、两张Files表、
灰底命令、图注及页码。编辑用`README.docx`保存在独立工程中，发布包使用`README.pdf`。
绘图模块统一命名为`visualize.py`。作者运行`tools/write_guide.py`并明确传入留存的
`--reference`、`--reference-sha256`和作者修改版`--author`，生成Word文档，
通过`tools/export_guide.ps1`导出PDF，并运行`tools/verify_guide.py --reference ...`
核对文件表、结果、原版式及页码。只在任务内留存核验记录，不放入独立推理包。

作者侧运行`python -m LightGenV2.tasks.t23_public_temporal_all_mlp.build_lab_package`，
生成`releases/LightGen-II-MLP.zip`。压缩包按白名单收集，文件SHA与核验信息保存于包外。

2026-11-27按作者要求对齐非MLP版的文件结构与说明手册，输入读取并入`infer.py`，
取消散点图生成并清理旧输出。读出头单独微调及旧global继续微调的尝试未达到目标，
均保留作者记录；64单元global尝试在原32单元候选通过目标后停止，没有用于发布。
本轮最终发布仍采用原32单元global结构，满足0.795–0.799目标。
