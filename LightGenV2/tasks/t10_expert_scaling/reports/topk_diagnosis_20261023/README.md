# 固定四专家 Top-k 诊断（2026-10-23）

来源：`/DATA/DATA1/guest3/t10_fixed478_3datasets_20260921_runs`，包含复用目录。
逐一读取36项metadata、result、history及TEST报告；本轮没有修改原训练或原指标。
原训练均60轮、4个特征相位层、seed17/27/37；选择验证macro-NLL最低的EMA模型。

## 既有结果核对

以下为三个seed平均准确率，单位%。训练指标是所选EMA模型在不增强训练集上的评估，
不是在线增强训练的batch准确率。所有原history均缺少逐轮训练准确率，不能补造。

| 数据集 | k | Train | Validation | Test | 各seed最佳epoch |
|---|---:|---:|---:|---:|---|
| Kather2016 | 3 | 76.33 | 73.49 | 71.01 | 60/59/58 |
| Kather2016 | 4 | 75.65 | 71.81 | 69.73 | 60/59/59 |
| BloodMNIST | 3 | 88.31 | 87.73 | 86.27 | 58/60/57 |
| BloodMNIST | 4 | 88.85 | 88.18 | 86.67 | 58/60/60 |
| OrganCMNIST | 3 | 74.35 | 76.51 | 61.85 | 60/50/52 |
| OrganCMNIST | 4 | 75.09 | 76.39 | 61.38 | 49/52/60 |

Kather：k4训练也略低，不能仅凭TEST下降归为过拟合。两个k的验证损失在后期仍下降，
但学习率已衰减到初值的约1%，60轮训练预算值得检验。seed17第40/50/60轮：
k3验证macro-NLL 0.8910/0.8740/0.8708，k4为0.8877/0.8709/0.8677。
与此同时k4的所选模型TEST macro-NLL 0.8877优于k3的0.8982，
说明最低NLL与最高accuracy并不等价；不能用TEST准确率重新挑旧checkpoint。

Organ：k3/k4验证均值仅差0.11个百分点，TEST相差0.46个百分点；
现有三个seed不足以把它解释为确定的结构退化。更大的问题是验证到TEST约15个百分点
的泛化落差。seed17的balanced accuracy也从74.15/73.65下降到60.11/59.67，
说明不能只用类别比例不同解释；但这仍不直接证明训练过程出现经典的验证损失回升。
股骨、肾脏等类别的TEST召回率较差。灰度输入重复成三通道后进行四象限编码，
沿用的翻转、90度旋转可能不适合解剖方向相关分类，需要另做预处理对照才能归因。
官方索引互斥不等于本次已重新证实患者独立性或检查了全部近重复。

## 结构与训练检查

- Top-k在入口恰好选k个专家，所选概率重新归一化为单位总功率，振幅取平方根。
  端口score再经固定温度softmax，光学router不保证可让任意额外端口趋近零。
  因而“允许最多k个”与当前“恰好k个”不同，不能证明k4函数集合严格包含k3。
- 之后是整个版面的相干传播、逐层强度测量/归一化/ReLU/饱和OEO。
  改变入口还改变干涉与阈值响应，增加支路不是无代价地添加可选特征。
  稀疏mask也不在每个后续专家相位层重新施加。
- 训练effective batch16，microbatch2。soft均衡项在各microbatch2上独立计算后累积，
  不等同于先合并16样本概率再做均衡。原实验各k一致，本次不改损失合同。
- CPU端到端检查：k3/k4全部可训练参数梯度存在且有限，router梯度非零。
  k4 STE与精确归一化振幅路由梯度最大差约4.3e-11；未发现k4特有梯度截断。
  检查未证明所有训练阶段、所有输入均有足够梯度。

## 已启动的单seed诊断

代码提交：`09d670a151b447a790bbbf109ccb7bf1fcd98834`，本地、服务器主线及GitHub
HEAD实测相同。旧runtime的model/train/study配置与主线逐文件SHA一致。
本地HTTPS连接GitHub失败，因此通过Git增量bundle快进服务器，再从服务器push
GitHub；没有通过文件复制覆盖源码。其他窗口已有暂存内容保持原样。

`diagnose_topk.py`：Kather/Organ各k3/k4，共4项，均seed17。
从原last的live/EMA/Adam状态续训；各追加30轮，lr从2e-4余弦降到2e-5；
保留原几何、损失、增强、microbatch2与EMA。原best纳入验证选模候选。
逐轮记录训练目标、验证指标；每5轮及首轮评估无增强训练集。
续训前额外评估live/EMA的train与val，区分EMA滞后和优化问题。
新目录只保留best/last，原run不改写。

正式启动路径：`/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/`
`t10_expert_scaling/runs/simulation/topk_diagnosis_20261023`。
最多两张空闲卡，跳过GPU6，完成/错误/信号退出时检查自己的子进程释放。
根据验证macro-NLL选定checkpoint后只评一次TEST。因为研究问题来自已经查看过的
TEST表，新增结果为开发诊断，不是新独立确认性结果，也不强求k4超过k3。

派生证据：`historical_summary.json`为36项结果摘要；`seed17_convergence.png`
展示训练目标与验证曲线。运行结果随后以新run的history/result/test_result为准。
