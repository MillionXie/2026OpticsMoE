# T10 专家数量与 Top-k 扩展实验

## 2026-10-23：两任务k4三seed微调重复

Kather seed17累计120轮已完成，验证accuracy选定epoch106：
TRAIN79.15%、VAL73.94%、TEST72.74%，同seed固定k3为77.20/74.34/71.28%。
这只是一seed开发结果，不能替换旧表三seed的69.73±0.28%。

用户要求第一/第三行k4重复三seed，统一seed17/27/37、累计120轮：
原60轮→原目标extra30（lr2e-4→2e-5，验证NLL选模）→margin30
（lr5e-4→5e-5，margin0.2、权重0.2，capture权重0.005，验证accuracy选模）。
复用Kather17的两阶段及Organ17的第一阶段，剩余9个30轮阶段需要训练。
按最新授权三张GPU分别固定seed17/27/37，同卡两个数据集并行运行
`k4_repeat_schedule.py`；每任务只补剩余60轮，内部30+30阶段按源结果依赖调度，
已有同身份结果复用。run为
`runs/simulation/k4_three_seed_20261023`，每阶段退出后释放，整个队列完成/异常有释放收据。
结果自动汇总到run下`table_summary.json`，不足2seed时标准差记null，不能填0。
最新统计只包含两任务k4累计120轮的逐seed、训练/验证/TEST均值和标准差，不再输出60轮表。
这是基于已查看TEST的开发微调重复，不构成统一预算的严格top-k因果消融；
不以TEST选择权重、seed或训练停止点。

## 2026-10-23：k4分类准确率微调（单次有界开发试验）

Kather N4k4 seed17从extra30的last90出发，额外30轮、lr5e-4降至5e-5，
原CE/soft均衡/TV保持，capture权重0.02降到0.005，追加0.2倍分类margin项：
`max(0, 0.2 + max_wrong(log p) - log p_true)`。目标是检验偏重正确类别排序能否
改善accuracy，同时保留capture指标观察光学功率代价。光学结构、参数量不变。
使用验证accuracy选best，以验证NLL打破平局，候选包含原best。
不承诺k4必胜；只做这一次30轮试验，不在TEST上反复搜索或以TEST触发停止。
运行GPU0，产物为`runs/simulation/kather_k4_margin_20261023`。
比较对象为同seed17无新增负载均衡的k3（TEST71.28%、VAL74.34%），不把三seed均值混入。
涉及已观察TEST后的开发诊断，不能冒称新的独立确认性结果。

## 2026-10-23：Kather k3入口路由负载约束诊断

旧soft均衡权重0.01仍保留：原项约束microbatch2的soft概率均值，不能保证
hard top-k实际使用均衡。无额外约束续训后的Kather k3在验证集始终选择同一组三专家，
平均soft概率约[0.202,0.138,0.271,0.389]，故不能宣称已学会动态专家分工。

新增可选`--hard-balance-weight`：定义`f_e=mean_batch(mask_e)`、
`p_e=mean_batch(probability_e)`，附加`lambda*N/k*sum(stopgrad(f_e)*p_e)`。
这是使用实际选择频率的可导代理，不是对离散mask直接求导；均匀使用时基线为1。
每次optimizer step单独重算同一增强图像的router，以完整effective batch16统计，
不会把16样本全光学图保留在显存。原分类/捕获/TV/soft均衡项保持。

预先固定两个权重0.1/1.0，仅Kather N4k3 seed17，从原last60继续30轮，
与既有extra30相同数据顺序、增强、学习率和优化器/EMA状态配对。
不根据TEST选择权重；报告路由是否恢复变化及验证性能权衡，不预设均衡提高准确率。
保留每轮hard负载/soft概率/选择集合数、辅助项及分类router梯度范数，
末轮与验证选定best均保存逐样本路由记录和类别分配。
若原best仍胜出，明确报告选定模型仍可能坍塌，不能以last路由替代best的性能。
新run为`runs/simulation/kather_k3_load_balance_20261023`，
由`diagnose_topk_schedule --balance-sweep`在最多两张空闲GPU运行，跳过GPU6。
这些属于已查看TEST后的开发诊断。

## 2026-10-23：固定4专家 Top-k 收敛诊断

针对已有固定478三任务表，重新核对36份训练/history/TEST报告。均为60轮；
原选模使用验证集macro-NLL最低的EMA模型，不是验证或TEST准确率最高的轮次。
三seed的Kather k3/k4训练准确率均值为76.33/75.65%，验证73.49/71.81%，
TEST71.01/69.73%；最佳轮次分别60/59/58与60/59/59，支持进一步检查优化预算，
不足以诊断严重过拟合。Organ k3/k4训练74.35/75.09%，验证76.51/76.39%，
TEST61.85/61.38%；验证与测试的较大落差不能归因于k本身。
原history只保存训练目标及验证指标，没有逐轮训练准确率。

当前路由恰好选择k个端口并重新归一化总输入功率；相干整场传播及逐层OEO
会改变多专家组合，固定光学router也未保证可将额外专家概率任意逼近零。
因此不能假定k4模型函数集合严格包含k3，也不能预设准确率必须单调。
均衡项实际在每个microbatch=2上计算，再累积成16样本的优化batch，
不是直接对16样本聚合后计算；本次为保留对比身份，不修改此损失。

诊断入口`diagnose_topk.py`从原last模型、Adam状态和EMA续训，原best参与验证选模，
新目录保存best/last及身份，不改动旧结果。seed17下Kather/Organ的k3和k4
各追加30轮，统一学习率2e-4余弦下降至2e-5；几何、增强、损失及EMA不变。
逐轮记录验证指标，每5轮记录完整训练指标，并在续训前对比live/EMA。
仅在锁定验证macro-NLL最优checkpoint后执行一次TEST；由于问题来自已查看
的TEST结果，新增结果标记为开发诊断，不能宣称新的独立确认性性能。
`diagnose_topk_schedule.py`最多占两张空闲卡、跳过GPU6，子进程退出后写释放记录。
新run位于主工程`tasks/t10_expert_scaling/runs/simulation/topk_diagnosis_20261023`。
历史摘要与收敛图保存在`reports/topk_diagnosis_20261023/`。

## 当前入口：2026-10-03工程整理

请先读[实际源码、四组结果及保留边界](CURRENT_VERSION_20261003.md)。
实际运行工程为 `t10_corrected_20260918`，不是下文已过期的校准PID或GPU队列。
原HEAD之外9份实际运行源码已另存Git恢复快照；现有结果、baseline和全部测速保留。
以下内容完整保留作历史协议，不代表今天正在训练，也不得直接照旧命令抢占GPU。

## 2026-09-18：校准完成，扫描运行中

12/12校准完成；按三架构验证macro-NLL的预定共同深度规则，选4层、学习率0.002。
相同4层的最佳checkpoint验证准确率：MoE 73.01%、总参数匹配D2NN 63.83%、等孔径D2NN 63.43%。
MoE对应训练准确率80.92%，探测窗捕获功率仅0.2725%；尚不能仅凭准确率判定光学实现成熟。
6层MoE验证75.00%不用于与4层D2NN混合比较。本轮未读取测试集指标。
证据：`kather_calibration_s17_17um_uuid_20260917`各run的`result.json`。

`kather_spaced_scan_gpu012_s17`已进入4专家k=1/2/3训练，后接k=4及每个N的三种D2NN对照，
总计39新run+3校准复用。运行GPU0/1/2，协调器878608；以下旧的校准运行中描述仅供追溯。
中期路由检查：N4k2两专家被选择比例约96%/99%，另两专家约3%/1%；
soft概率接近均衡不能证明hard top-k负载均衡，须随最终结果报告此项，不预设专家数增加必然提升。

输入审计澄清：当前`encode_rgb`先把各原始RGB通道插值到目标边长的一半，再拼[R,G;B,mean]。
MoE通道112²，D2NN通道S/2²；因此两者原始信息带宽并未严格统一，D2NN可能保留更多细节。
不应把当前协议称为“相同112²源信息”；后续匹配带宽对照需新profile，不能静默修改本轮结果。

路由数据保存：每个MoE结果的`result.json`和逐轮`history.json`均保存验证集的`route_load`（hard top-k选择比例）、
`route_probability`（soft概率均值）、`distinct_selected_sets`以及验证指标；`audit_routes.py`可将所有run导出为JSON/CSV，
并计算死专家数、最大/最小hard负载和soft概率熵。soft概率均衡不等于hard选择均衡，论文必须同时报告两者。
当前损失中的`router_balance_weight=0.01`只约束batch内soft概率均值，不能直接阻止hard专家坍塌；增加该权重可能降低分类准确率，
因此后续只对固定配置做无约束/当前约束/增强约束的成对消融，不覆盖已有结果。

OEO核查：当前实现为`intensity -> mean normalization -> spatial LayerNorm -> ReLU -> t/(1+t) -> unit-power`。
LayerNorm输出严格包含正负值，ReLU会把负半轴置零；这不是数值bug，但会造成半数左右的归一化响应没有梯度通过，需单独做激活消融。
图文/音文工程的`relu_softsign`实现是`softsign(ReLU(z))`，只把正半轴映射改成平滑有界函数，并没有恢复负半轴。
因此它不能被表述为“完全取消ReLU截断”。当前正式扫描仍保持原OEO以保证可比；后续新增profile比较原`ReLU+t/(1+t)`、`ReLU+Softsign`，
必要时再设计物理上非负的平滑替代，所有版本同时应用于MoE和D2NN。

架构冻结前的关键检查：MoE输入先编码为224×224单幅振幅；224×224路由相位经传播后读取49个24×24端口，
得到每个样本的soft概率，再做top-k选择和功率归一化，复制到专家阵列。每个特征循环依次执行专家相位+传播+OEO、
全局相位+传播+OEO；最后传播到8个分类探测窗。没有CNN、Qwen或电子残差分类支路。D2NN使用同一RGB编码函数但目标边长不同，
当前仍存在源信息带宽不完全一致的问题，必须在正式结论中单独标注。

10×10专家阵列本身没有版面限制，边长为2510逻辑像素。此前代码把路由器错误地固定为7×7=49个端口，
这是实现缺陷而非架构限制，已在最新代码中改为按N自动生成端口：N=100时为10×10、312×312路由面；N≤49仍保持224×224旧布局。
路由器相位参数、传播网格和D2NN参数匹配均随路由面更新，100专家仍需单独记录显存与耗时。

输入公平性修正：D2NN不应把224×224源场限制在大相位面的中央，否则外围可训练相位没有输入覆盖。
正式协议采用各架构的全孔径输入：MoE每个224×224专家接收同一224×224 RGB编码源并复制到阵列，D2NN将同一RGB图像编码后铺满自己的连续相位面。
两者都不额外使用电子特征或残差支路；公平性通过相同原图、波长、采样、传播距离、层数、训练预算和参数匹配控制。

确认性实验使用seed=17/27/37，至少覆盖Kather2016和DeepWeeds；N=100使用top-k=[1,10,25,50,75,100]，
其余N沿用既定稀疏—中间—密集取值。当前正在运行的旧seed17队列不修改，新的动态路由代码用于后续新profile。

DeepWeeds仍未训练。新增CPU候选审计入口：
`python -m LightGenV2.tasks.t10_expert_scaling.audit_deepweeds --data <deepweeds_fold0> --out <task/runs/smoke/audit_run>`。
输出跨split dHash候选、接近拍摄时间的分组统计与候选图；候选并不等于已确认同一植株。
审计不会自动把training_ready改为true，也不更改官方划分。

DeepWeeds全量CPU审计已完成：54对跨split dHash≤4候选；样例可见近乎相同的连续拍摄画面。
证据run：`runs/smoke/deepweeds_scene_audit_20260918`，源码c90bd143。
后续采用独立`deepweeds_date_camera_grouped_s17_v1`：按拍摄日期+相机分组，再合并54对候选涉及的组；
StratifiedGroupKFold(5, shuffle=True, random_state=17)，固定fold0测试、fold1验证、其余训练，
不搜索seed/fold，不用模型表现挑划分。组是场景代理而非真实植株身份，不声称完全生物学独立。
`prepare_deepweeds_grouped.py --data <official-cache> --audit <audit-run> --out <separate-cache>`
验证原缓存hash、标签、图像ID完整覆盖、组互斥、候选不再跨split、各类非空后生成新缓存及清单；
官方缓存保持training_ready=false，只有新分组缓存允许训练。

## 最新执行约定：GPU0/1/2、4专家全k、专家+global匹配

用户追加要求已执行。当前实际训练数据只有**Kather2016，5000张、8类组织图像**，
train3496/val752/test752；test仍不参与选模。DeepWeeds17509张9类只完成下载和精确去重，尚未训练。

本任务后续固定GPU0/1/2，不再排GPU6。GPU6原子进程863246已退出，未完成run保留在原校准目录
`interrupted_gpu6/`；0、1原进程849900、828346被新协调器接管，继续原训练，GPU2重跑被中断的一组。
协调器PID878607，扫描等待PID878608；实际进程以status.json为准。
新执行工作树`/DATA/DATA1/guest3/t10_gpu012_20260917`，源码`affa905f`。
新扫描`runs/simulation/kather_spaced_scan_gpu012_s17`，39个新run+3个校准复用，共42个比较点。
校准结束自动接续，GPU UUID绑定，完成后检查子进程退出和显存释放。

Top-k：4→[1,2,3,4]，9→[1,3,5,9]，16→[1,4,8,16]，25→[1,6,12,25]，
36→[1,9,18,36]，49→[1,12,24,49]。以下旧章节的32/33组、GPU6与k=1/2/4均属历史约定。

每个N增加`d2nn_expert_global`：匹配J×(N×224²+G²)，不计Router；
原`d2nn_total_parameter`匹配上述参数加224² Router，仍保留；`d2nn_same_aperture`为等孔径对照。
新增版本沿用同N的共同深度及总参数D2NN已选学习率，不额外使用test调参。
边长取最近可容纳偶数，实际相位参数匹配误差<0.5%，写入metadata.geometry。
CPU队列检查通过：39个唯一新配置，N4覆盖全部k，六个N均有新增D2NN版本，不启动额外GPU。

## 当前队列调整（2026-09-17）

校准仍在运行，已完成首组3/12；三张物理GPU 0/1/6均有训练进程。
原队列没有自动衔接扫描，现已补齐：独立工作树`/DATA/DATA1/guest3/t10_scan_20260917`，
源码`f025f467`，协调进程启动PID814553。它在CPU上等待校准完成及释放记录，
随后按validation macro-NLL选择共同深度/各架构学习率，并自动使用同三张卡。
扫描run为`runs/simulation/kather_spaced_scan_s17_20260917`，共32次新训练加3个校准复用结果。
CPU模拟队列检查已通过（唯一配置、选模、复用、无GPU启动）；不是新性能结果。

最新top-k取值替代下文原设计表及`reports/design`旧矩阵：
4→[1,2,4]；9→[1,3,5,9]；16→[1,4,8,16]；25→[1,6,12,25]；
36→[1,9,18,36]；49→[1,12,24,49]。
实际执行清单由`schedule.py --mode scan`生成，包含23个MoE点及12个D2NN点，
其中9专家dense MoE和两种D2NN复用校准已选结果。输入/相位尺寸、损失、60epoch与seed17不变。
所有子进程在单次训练结束退出，整个队列完成/异常时写释放检查。

DeepWeeds图像已经下载并核验存档hash，17509张256² RGB，官方划分无跨split精确重复；
场景/近重复审计尚未完成，`training_ready=false`，尚未启动该数据训练。
不能将Kather扫描队列描述为两个任务全部已经自动排完。

当前状态：**已通过CUDA结构/梯度检查，seed17的Kather验证校准正在三张GPU上训练。**
运行：`runs/simulation/kather_calibration_s17_17um_uuid_20260917`，训练源码`752d2995`。
GPU按物理UUID绑定0/1/6；最多三张，子进程逐run退出，队列结束/异常时检查释放。
当前12个配置用于N=9的共同深度与学习率校准，不是完整专家数矩阵的最终成绩。
DeepWeeds仍在数据获取/审计阶段；原始PBC未启动。详细证据见[复现入口](reports/reproduction/README.md)。
本轮研究仿真性能，不以 SLM 版面限制 N。用户最新指定输入逻辑像素17 μm、相位设备8 μm、专家间隔30逻辑px；
专家采用正式 T01/T02/T08 的 **224×224 px**，不沿用病理 demo 的 146×146。
尺寸依据：T08 `reports/optical_router_moe_20260907/main/config.yaml` 的
`expert_size=224, expert_pitch=254`。仿真计算网格为17 μm；8 μm相位SLM通过保持物理尺寸的最近邻映射承载权重，
依据同配置的 `hardware.phase_mask.mapping=physical_pitch_nearest`。224逻辑像素映射为476设备像素，
并非独立学习476²个权重。新协议ID为 `raw_rgb_fullfield_oeo_17um_phase8um_v2`，此前全8 μm设计未启动训练。

推荐两个主任务：Kather2016 组织分类和 DeepWeeds 杂草分类；原始 PBC 血细胞分类为第三项扩展。
三者的数据许可均核查为 CC BY 4.0，依据见 [数据登记](dataset/registry.json)。
它们是三个应用域的分类任务；联合训练用于研究跨域专家分工，不等于验证了分类、检索、回归等不同任务形式。

| 专家 N | 排布 | 单专家 | 有效面边长 G | 17 μm 对应边长 | Top-k |
|---:|---|---|---:|---:|---|
| 4 | 2×2 | 224×224 | 478 | 8.126 mm | 1、2、4 |
| 9 | 3×3 | 224×224 | 732 | 12.444 mm | 1、2、3、4、5、9 |
| 16 | 4×4 | 224×224 | 986 | 16.762 mm | 1、2、4、8、16 |
| 25 | 5×5 | 224×224 | 1240 | 21.080 mm | 1、2、4、7、13、25 |
| 36 | 6×6 | 224×224 | 1494 | 25.398 mm | 1、2、4、9、18、36 |
| 49 | 7×7 | 224×224 | 1748 | 29.716 mm | 1、2、4、13、25、49 |

`G = sqrt(N)×224 + (sqrt(N)−1)×30`。传播画布暂定 `G+40`，FFT 的数值补零另外计算；
这些尺寸都不作硬件可部署性承诺。固定 k 比较与固定 k/N 比较必须同时看。

## 架构和比较

**此前病理 MoE 有 global phase。** 六层是三个“专家层 + 全局层”循环，而不是每个专家六层后直接求平均。
专家输出为复光场；全局相位与传播使不同区域混合，最后由类别探测窗口的能量读出分类。
旧实现入口是理想图像复制分光；局部专家传播和全局传播并非同一计算域，不能直接等同于完整自由空间仿真。

本轮模型采用以下新仿真合同，正在验证前后向与数值传播：

```text
RGB → 固定空间编码 [B,1,224,224]
    → 光学 Router（224² phase → ASM → 固定区域积分 → Top-k）
    → 总入口功率为1的理想分光，投送到 N 个224²区域
    → [N块专家 phase → 整场 ASM → OEO
       → G² global phase → 整场 ASM → OEO] × 3
    → 整场传播 → 固定类别探测窗口 → 能量归一化 → 类别

D2NN+OEO：同一输入编码 → 覆盖其有效相位面的上采样
    → [dense phase → 整场 ASM → 同定义 OEO] × 6
    → 同一探测器和判决
```

默认两者**每个特征衍射层后都有 OEO**，无可训练电子旁路、Qwen 或 CNN。OEO 中有电子归一化和非线性，
不能写成全光无源系统。Router 额外有一次光学测量，也单独计入参数与成本。
六层是当前起始选择，不声称在224专家新合同下已证明最优；仅在固定 N=9 做小规模验证选深度，
不展开 N×k×层数大网格。

三组主对照：MoE+OEO、**总相位参数量匹配**的 D2NN+OEO、**有效孔径相同**的 D2NN+OEO。
另外固定 global 面为1748²，在 N=4/16/49 上补充对照，区分专家数量与全局层扩容的作用。
不沿用“只匹配激活专家参数”却声称总参数相等的口径。

## 实验顺序

1. 先完成数据分组/去重、数值传播收敛、张量覆盖、功率守恒、Top-1 路由梯度检查。
2. seed17，两个主任务先跑 N=4/9/16 的 k=1/4/N 及两种 D2NN，共 **28 个配置**；小规模校准另计。
3. 协议锁定后补齐 N=4–49、固定 k 和固定比例扫描，两个任务共 **88 个配置（包含前述28个）**。
   D2NN 不随 k 重复训练。完整矩阵一致时才可复用 pilot，否则重训。
4. 固定 global 的容量控制共18个配置；在 N=9/25 上进一步比较两任务联合训练与单任务训练，
   检查专家是否按任务/类别分工。路由消融、联合训练、噪声检查另计，详见方案。
5. 本轮初扫仍只有 seed17。稳定结论建议后续对预先选定的代表点补3个配对种子；当前未安排这些训练。

不是要求结果随 N 或 k 单调增加，也不预设 MoE 必须胜出。要回答的是：性能何时饱和、最佳稀疏度是否变化、
收益来自专家分工还是孔径/参数增加，以及大 N 是否出现训练不足或过拟合。

## 文件和入口

- [详细实验协议](docs/PROTOCOL.md)：输入、OEO、路由、损失、公平性与数据划分。
- [设计配置](configs/study.json)：机器可读设计参数；**不是已有训练入口可加载的配置**。
- [训练入口](train.py)读取设计配置并记录实际参数；[三卡队列](schedule.py)当前提供calibration阶段。
- [主实验矩阵](reports/design/main_matrix.csv)、[几何/参数表](reports/design/geometry.csv)。
- [复现入口](reports/reproduction/README.md)：本任务唯一证据入口，当前无新性能数值。

从仓库根目录运行（仅标准库，不使用GPU）：

```bash
python LightGenV2/tasks/t10_expert_scaling/plan.py
```

脚本校验专家尺寸/间隔、路由探测区域不重叠、k 合法、入口功率、D2NN 参数匹配误差，
并生成计划 CSV/JSON；不会自动启动实验。今后训练写入本任务 `runs/smoke/` 或 `runs/simulation/`。
