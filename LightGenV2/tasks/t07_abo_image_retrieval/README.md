# T07 ABO 商品图搜图：独立光电工程

当前入口为 `python run.py`，只使用本文件夹的 `standalone/`，**不导入T01或旧experiments，也不加载完整Qwen模型**。
日常步骤看 [COMMAND.md](COMMAND.md)；维护、导出与历史数值证据看 [复现入口](reports/reproduction/README.md)（源码仓库内）。

## 给老师检查的结构

- 固定图片224×224 + 固定英文检索指令（见standalone/data.py）；无类别/商品标题输入。
- 冻结Qwen patch Conv3d、视觉位置表、主merger，以及固定prompt用到的token embedding行。
  原始tokenizer和processor保留；未知prompt/token会报错，不能冒充支持任意语言输入。
- V：196×1024 → Linear/LN到192 → 两级光电融合 → 投影回1024并加输入跳连 → merger得到49×2048。
- 图像特征填入文本模板的image token位置；L：S×2048 → 192 → 两级光电融合。
- 每个模态：光Router一次CCD选Top-2；四个专家中的两个一起传播/CCD；融合后电子重编码、再加载全局相位/CCD。
  V/L共六次捕获，不是三层专家再额外加Router。Router能量标准化、softmax、Top-2仍为电子运算。
- 电子残差：V两组3×3 depthwise二维卷积，L两组核长5因果一维卷积；每组含192→384→192通道MLP。
  无attention、Transformer、VGG或额外图像分支。当前未采用失败试验中的扩核/增强读出头。
- 同尺度融合：En=E/rms(E)，On=O/rms(O)，M=(1-alpha)En+alpha On，F=rms(E)M/rms(M)。
  尺度统计停止梯度，alpha范围[0.01,0.95]，best约0.087～0.104；不是性能贡献比例。
- 最后有效token mean/max拼成384维 → LN → Linear64 → L2。头仅25,408个参数。
- CCD明确保留 `frame mean → clip12 → log1p → pool224 → row LayerNorm → ReLU → Linear192`；
  **不是纯线性CCD读出**，不得在部署时擅自去掉并继续引用当前指标。
- 专家224×224，2×2排列、间隔30，有效场478×478，FFT画布518×518；532nm、10cm、逻辑17um。
  8um相位SLM需物理映射，不可直接把224逻辑像素当硬件像素。相位为2πsigmoid(raw)，当前从best续训，不是全零初始化。
- 独立模型保留原相位、融合及电子权重。只删除无输出作用的L隐藏状态重构投影、原始TF模块、LM头、
  未用的词表行、旧优化器和历史候选。输入前端约2915万冻结参数，可训练约278万。

## 当前结果及边界

固定best源SHA256：`3674981c3499555077c7eadc3072a11e07583675000ec4c012616a96185867f5`（epoch10）。
旧版A100为70.0000%；旧版4090为70.2083%。独立版最终隔离目录4090完整复评为70.2083%、去光67.2917%，
测试embedding对旧版平均余弦0.99999756；存在微小混合精度差，不宣称逐位相同。最终实跑结果见包内reference和[验收记录](ACCEPTANCE.md)。
独立assets约85.6MB，包含processor、模型、仅训练样本的教师目标；无完整2B模型。

ABO similarity10：10类200商品、每商品12视角；train/val/test按商品120/40/40，
即1440/480/480图片，val不使用。480测试图查120训练商品中心，**同类别算相关**，不是同商品实例检索。
每个query有12个正候选，报告Hit@1（旧称R@1）、mAP等，不混用positive Recall。
先逐视角L2归一化，再按商品均值/L2；不按真实类别预筛选候选。

训练包括监督对比、训练教师向量/类别中心、Router均衡、相位DC与工作点约束；
训练中20%～30%相干未调制分量、截断偏置高斯CCD噪声、相位旁路；位移=0、k空间约束关闭。
评估是确定性理想仿真，**不是实际光路或固定20%零级光测试**。
EMA、每5epoch test选best，仅best/last；成绩为test-selected，不能声称独立无偏测试。
冻结完整Qwen baseline历史95.2083%，当前仍明显落后。完整baseline需另行加载大模型，
不混入此精简学生运行入口。

## 文件怎么找

| 文件 | 内容 |
| --- | --- |
| run.py | 独立命令入口 |
| standalone/frontend.py | 冻结Qwen必要前端，无模型/Transformer类 |
| standalone/model.py | 电子残差、融合、图文组装、检索头 |
| standalone/optics.py | Router、相位、传播、CCD及实测注入接口 |
| standalone/data.py | 固定数据协议、图库和指标 |
| standalone/cli.py | 评估、去光、独立微调、报告和相位图 |
| standalone/export.py | 一次性CPU选取原权重；交付接收人不需要执行 |
| assets/ | 运行资源（在交付包内；不进入Git） |
| reference/ | 交付版本实跑报告（在交付包内） |

源码中的 `legacy_run.py`、旧configs/refinement/retrieval_contract只保留历史审计，不进入独立ZIP。
旧后端被其他任务共用，不做全仓删除。历史证据仍在Git和runs；本次不删除数据/旧权重。
`build_lab_package.py`白名单打包当前代码、assets、ABO数据和reference；名字沿用公共规则，
**此次是仿真/微调包，不是包含SLM/CCD厂商SDK的实验室控制包**。实测注入接口不能冒充已验收的自动采集流程。

GPU默认一张，同一助手最多两张；退出后确认自己的PID已释放，不杀他人进程。

## 当前训练优化试验

`--profile teacher_curriculum` 保持上述推理图不变：训练集教师特征预热→光电联合→无蒸馏收尾，
batch40、跨商品正样本、关系蒸馏递减、EMA、四个alpha固定为当前best数值。
这是任务内预热，不是额外外部数据集预训练；不宣称已获得提升。详情及命令见COMMAND第5节。
正式独立包与70.2083%证据不覆盖；新候选通过全量正常/去光评估后再考虑替换。

上述小集curriculum已完成30epoch，未提升（新epoch最高70.00%、末轮67.9167%），保留原best。
下一方案是`standalone.broad_transfer`：从较大原始ABO中限量选商品类型，在与目标商品/图片做去重的
预训练池上联合训练相位与电子，再迁移当前10类。训练辅助类别头不进入推理，alpha固定、无完整Qwen/教师。
代码/参数及完整命令见COMMAND第6节；目标75%，尚未获得新性能结论。
预训练池已审核并选出128类型、6144商品、12288图；正式run为`runs/simulation/broad_transfer_20260910/`，
阶段结果分别在`artifacts/pretrain`和`artifacts/adapt`。详细数据排除/训练身份见复现入口；原正式包不覆盖。

该轮已完成：预训练后目标Hit@1为62.0833%，微调新epoch最高67.50%（10/15轮），末轮67.2917%；
最终选择epoch -1，即原70.2083%保底，不代表预训练带来了提升。

### 严格高alpha候选（尚非正式70.21%权重）

`standalone.broad_transfer --mode adapt --profile high_alpha` 使用原最好相位/电子参数热启动，
四个融合系数改为`0.4001+0.3999*sigmoid(raw)`，初始0.45；浮点饱和也不会低于或等于0.4。
只在高alpha候选中选优，绝不拿旧低alpha成绩当本版本结果。计划60epoch×64steps，训练batch40。
前5epoch冻结主要电子残差与输入/输出adapter，训练相位、Router、光学编解码、融合系数及读出；
随后联合训练，分组学习率+余弦衰减+EMA。相位峰值LR0.004，Router0.0005，无教师loss。
轻量裁剪、翻转、小角度旋转、亮度/对比度、低概率轻模糊；25% batch加入轻量CCD噪声及20%～30%DC。
V/L融合前光特征另加训练用分类监督，其辅助头不进入推理、不替代最终检索。
每5epoch全量test选EMA best（明确test-selected），仅best.pt/last.pt。
最终输出同权重去光、相位像素打乱、单种轻量CCD噪声测试、相位更新量和Router分布。
推理仍为原六次光捕获、Top2、无TF/attention；Vision外层输入跳连仍在，alpha不是全网能量/性能贡献比例。
固定ROI、正常精度相位、k滤波/像素位移关闭，CCD解码仍含已有log1p等非线性（未新增）。
命令见COMMAND第7节；超过75%是优化目标，不是已测成绩。旧ZIP不会自动被替换。

该60epoch试验已完成（source `3eb20ae2`）：best为55轮，Hit@1 **67.9167%**，同权重去光64.1667%
（下降3.75个百分点），相位像素打乱52.7083%，单种轻量CCD噪声67.0833%。alpha约0.432～0.441。
证据：`runs/simulation/high_alpha_aug_20260910/artifacts/final_report.json`，不是低alpha的70.21%版本。

### 高alpha检索对齐续训

`--profile high_alpha_retrieval` 仅从严格高alpha的best续训，推理图/参数量/几何完全不变。
训练图库由1440张**训练图**生成120个商品中心，每epoch刷新；每个query排除其自身商品的全部视图，
同类别其他商品作为正候选，其他类别为负候选。损失为多正例检索NLL+最难负商品margin，
降低辅助分类CE/光特征分类权重；保留轻量跨商品监督对比、Router均衡及原高alpha噪声。
训练图库是无梯度的缓存，不加入推理、不使用test/val图，不按真实测试标签筛候选。
先25epoch光电联合调整，最后5epoch仅训练已有读出头（光学与电子残差冻结）；减轻裁剪/旋转等增强。
每5epoch分别评估live和EMA，按同一test指标选best；保留高alpha起点，明确test-selected。
仅best.pt/last.pt；训练日志另记排除自身商品后的训练图库Hit@1，以区别辅助分类正确率。
参数覆盖集中在`standalone/retrieval_training.json`，基础合同继承`high_alpha.json`。见COMMAND第8节。
这是针对训练目标与实际检索不一致的尝试；不声称已解决所有语义表征差距，未达到75%前不作为正式提升。

该30epoch续训已经完成（source `e59fc45f`），选择epoch15 EMA：Hit@1 **68.9583%**，同权重去光
64.5833%（下降4.375个百分点），相位打乱55.2083%，轻微CCD噪声67.7083%。四个alpha约0.430～0.440。
训练PID2135641已退出，所用GPU已释放。仍未达到75%，不将低alpha70.21%混入此候选。

组会数据分析及老师审阅用独立代码位于仓库根 `LightGenPublic/tasks/t07_abo_image_retrieval/`。
该目录是独立仿真/续训审阅快照，不覆盖本目录的优化代码或旧实验室包。清理没有删除历史run/数据；
数据审计脚本为本任务 `analysis/audit_for_meeting.py`，只读CPU分析，保持原始测试口径。

### 完整输入 / SAM / 全场语言读出（2026-09-11）

在本目录继续优化，**不覆盖已交付LightGenPublic审阅包**。三组都从同一68.9583%高alpha权重开始：
`preserve_adam`：等比缩放、白色补边到224²，关闭裁剪和旋转增强，AdamW适配；
`preserve_sam`：同上加SAM，rho=0.03、前三轮线性热身；
`preserve_fullfield_sam`：在SAM基础上把L端expert/global CCD从“pool224后截前77行”改为整场pool到77×224。
V端读出不改，输入图像/光学ROI/专家大小/17um/10cm均不改，无新增网络参数、TF、attention或教师。
相位/Router/电子一起更新，alpha仍严格>0.4，原噪声、20%～30%未调制分量和专家均衡保留。

SAM是Sharpness-Aware Minimization，不是Segment Anything：两次反传共用同批图与相同随机噪声/dropout，
先暂时扰动活动参数，再精确恢复原值，用第二次梯度做AdamW更新；异常时也恢复，冻结参数不扰动。
参考[原论文](https://arxiv.org/abs/2010.01412)。这增加训练计算，不增加推理网络；不保证涨分。

三组使用相同的30epoch×64steps、batch40与适配学习率，末5轮仅读出，EMA/live每5轮按test选优。
第一组相较旧68.96%同时包含输入修改和新适配训练，不能把差值全归给裁切；SAM两组对照的配置更严格匹配。
新输入/读出模式写入checkpoint metadata，图库构建、训练、评估统一读取；旧权重默认仍为原中心裁切/前行读出。
选择只在各自新合同内进行，不能拿旧68.96%不同预处理的结果充作保底。仅best/last。
`generalization_queue` 单卡串行、每组一个独立进程，失败停止；状态/指标汇总到queue的status.json，完整命令见COMMAND第9节。
本节是实验计划/实现合同，未完成测试前不声称已超过0.7。新权重必须用本版T07加载，不直接放入旧审阅ZIP。
