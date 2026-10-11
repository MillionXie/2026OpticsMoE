# 用户换数据：完整Butterfly-10K／2、4、8、16层

2026-10-11最新人工指令覆盖FishNet继续运行和12层计划：换更多类别、四种深度、
按验证集选模，遇到过拟合／欠拟合／未收敛适当调整，最多三GPU，使用完释放。
继续比较每深度rho0／rho0.3；不预设残差或更深模型必须胜出。

## 数据身份与划分

采用作者完整发布的VLM4Bio Butterfly-10K物种分类基准，60物种、10,013张，
不是从大数据集自行挑容易的类别；所有发布类别和图像保留。来源：
[作者仓库](https://github.com/imageomics/VLM4Bio)、
[数据卡及许可](https://huggingface.co/datasets/imageomics/VLM4Bio)、
[原论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/eced4a5fbc776e81b45e2f72447f0164-Paper-Datasets_and_Benchmarks_Track.pdf)。
固定数据仓库revision `354805681278db357ffdff8749a29e2ca145d32e`；
DOI10.57967/hf/3393。逐个下载文件核对Git blob身份并保存SHA256清单。
完整逐行元数据实际60类／10,013，最大两类各2600，最小类21；
作者论文的最大类2000描述与本revision实际发布数量有差异，以保存文件为准，不裁减。

作者明确声明全部蝴蝶图像CC BY4.0；逐图许可CSV覆盖9664个文件，
另外349个未在CSV出现，由作者数据卡的全体许可声明覆盖，不能冒称其逐图映射已齐全。
保留原许可CSV、JSON和原始来源BibTeX，发表时引用原始收集者及基准作者。

按文件中的标本编号规范化正反面、白标准图及零前缀别名，再与原始像素／缓存像素
精确重复连接合组，seed20261011按类70／15／15划分。标本编号预审计6780个，
没有跨类冲突；实际像素连接后数量以准备manifest为准。作者文件名train/test词
不充当新划分。所有记录保留，任何分组标签冲突须失败审查，不能默默分开。
已知标本分组仍不保证不同家庭／brood／采集地点独立，未知别名可能残留；
长尾小类必须报告support、macro recall、macro F1、NLL及balanced NLL。

完整作者图像EXIF转正、RGB8、bicubic150；固定RGB振幅编码沿用原T18，
没有CNN、预训练特征、学习式电子头、离线增强、再分割或裁掉难图。

## 光路及训练合同

四种主路径深度2／4／8／16对应1／2／4／8周期；九专家、原OEO和ASM传播、
router没有残差，rho只作用expert/global，相干振幅混合公式不变。
可训练相位11／21／41／81张，参数449848／889696／1769392／3528784。
60个固定32×32 CCD窗，八行布局8/8/8/8/8/8/8/4，间距28、整体居中，
所有深度同一布置；窗和传播buffer不算参数。

seed17，AdamW lr0.002、有效batch16、EMA0.95、LS0.02、capture0.2、
phase smooth0.02、原几何增强。全部输入留CPU、仅当前小批量移GPU，
避免完整10K float数据占满显存。全组使用microbatch1、eval1，样本数加权累积，
每有效batch只做一次clip／AdamW／EMA（含尾batch／辅助项）。
这是新数据统一执行合同；旧12层batch2和single对比失败，不能声称已通过旧门槛。
新GPU smoke核验实际micro1合同下rho0精确一致／初始化／router／重复前向／
全部有限非零梯度，以及同小批量图的宏损失与加权累积梯度。

每轮验证EMA balanced NLL最低选best，精确同分保留首次；不按accuracy反选。
共同ReduceLROnPlateau规则factor0.5、patience8、minlr0.00002，指标只有val balanced NLL。
train精度为未增强完整train上同一EMA，每5轮和端点评估，与val口径一致。
每轮保存history、梯度、顺序／增强身份，best／last，best验证预测；不存逐轮PT。

## 有界适应与最终锁定

先各臂完整100轮，只读train／val，不测test。每深度两臂共同判断：
末十轮val balanced NLL均值比前十轮改善超过0.5%，且任一best落在最后十轮，
则两臂均从原last继续50轮；150轮仍满足则共同延至200，最多200。
条件未满足不自动延长，最后仍改善则标未收敛，不能伪装架构上限。

明显过拟合的预登记触发：任一臂末端train／val差≥10个百分点、
末十轮平均val balanced NLL较此前best恶化≥5%，且best已落后≥20轮。
允许每深度仅一次共同加强正则的从头配对训练（完整100轮，LS0.05、smooth0.04、
旋转20°／平移5px／scale0.10，其余相同），原结果和权重全部保留。
只以两臂best验证balanced NLL平均值选整个配对方案，同分保留base；
不会给rho单独挑不同方案制造配置不一致。

若两臂末端train macro recall均低于85%、train／val accuracy差小于10pp，
且末十轮val NLL均未改善0.5%，则标记低拟合停滞风险；数值、标签／编码、梯度
与完整数据核验通过后，允许一次共同优化恢复从头100轮：lr0.003、LS0.01、
smooth0.01、原增强，其余相同。它与正则重训二选一，最多一个追加配对方案，
仍按两臂平均best验证NLL选整体方案，负结果全部保留，test不参与。
低train不单独证明欠拟合；若恢复无效，报告优化／信息／容量限制未解。
其他改变需要新记录并保证两臂共同规则，不无限追加候选。
每深度最多base200＋一次恢复或正则100轮。
这些阈值是运行诊断规则，不是证明过拟合／欠拟合的统计检验；同时保留实际曲线。

所有调整和方案选择结束，核对PT／配置／数据／源身份，保存pair和selection lock，
之后才首次加载test，获选EMA每臂仅一次test，不比较raw／last、不测被淘汰方案。
test不参与层数、学习率、轮数、正则、选PT或追加试验。新test首次使用，仍是
单seed开发后的验证选模评估，不声称跨地点／家系独立或确定的深度因果效应。

## 执行与保护

只使用物理GPU1／2／4 UUID，最多三GPU且每GPU一个本项目训练／推理子进程；
Torch上限3GiB、另留800MiB，不能终止其它项目。CPU协调按深度成对调度，
完成子进程退出、确认实际释放。数据准备／CPU审计不需要GPU。

FishNet验证suite父305617和部分四层子518458已停止，GPU实际释放，
`dataset_change_pause_20261011.json`保存停机身份；六层两个完整结果、四层部分
best／last／history保留。12层smoke run保留失败与原源码，未训练、已取消。
旧FishNet test开发选模与Mango88.04%开发结果保留，不重训、不重测。
新增报告与FishNet报告分开；真实train／val曲线及最终单test点，单seed无误差条。
源码只沿main经Git同步，不改历史冻结训练源，不增分支／worktree／源码副本。
