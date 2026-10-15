# 新人工授权：只微调二层残差，目标仿真测试82%

用户已接受三、四层候选，明确只继续调二层、尽量测试82。旧185计算结束及禁止恢复的限制只对旧run有效；本次新增独立run，不重开旧worker/claim。三/四层、无残差100轮控制以及全部旧来源/PT/数据/历史/失败/claims保护。

run caltech10_L2_rho03_refit82_s17_20261015，189源和root metadata实际Git 97e80cc572e63664eb93843bf06cd85780463657。CPU controller 2738360、非A100 GPU2子2739422真实已启动，只有一卡一子、A1000，无failure。首次只读E331 val74.5223%、NLL1.217741，属于训练进度而非测试或新最终分数；本轮尚未完成350段科学/新测试，不套旧78.9962为新结果。

## 证据与新的训练合同

旧二层完整325、选E232：train76.7411/val75.1592/test78.9962/macro75.0158，训练拟合也弱、last LR2e-5已经平台；以较弱正则和恢复学习率改善拟合作为假设，不声称已证明唯一原因/保证82。完整恢复已审LAST E325 raw/EMA/Adam15600/scheduler325，prefix325不重播，不假称bestE232有完整优化器。

新阶段从E326起MixUp.3->.1、标签平滑.08->.02、相位平滑权重.05->.01，capture.2/EMA.95/原augmentation不变；LR一次恢复.0005，并明确重置ReduceLROnPlateau的best/bad/cooldown统计，保留global scheduler epoch和Adam全部动量/累计step、EMA，不冒称调度器完全没重置。后续保存段不重复重启LR。

原二层E-G/4个224专家/光router/global478/10窗口/input100/8bit/479364phase与6参数张量保持，无电子分类器。原十类/角色/缓存/划分保持，train761/val157/test1833，48Adam每轮、有效16/micro8/尾9。无残差冻结100选E98，test77.6868522，不再训练或测试；三、四层不调整。

每25新轮保存、独立科学通过后检查选中模型一次test；没有总epoch cap，82为目标不是保证。实际accuracy>=.82才达标（81.9967不能靠显示四舍五入算82）。或验证NLL无.0002改善75新轮且LR最低时平台暂停并报告未达。每段1800拟合秒故障保护，不在失败后自动重试/放宽门槛。

## 实际新准入与启动身份

- 新CPU proof实际一次passed，SHA f9273231187036dcdaa4207fedfa1b83c56395dddfaeff3680da9303b64a10a4；E325 raw/EMA/Adam/scheduler/source/新配置和独立审核编译核，0models/forward/test。
- 新GPU2 numeric proof实际一次passed，SHA 5aceb101ed3969fba89d722b10938c8fab2d796ac7b417b88e0f94a16b85ed5a；真实恢复E325 raw，all10/macro16/tail9、微批1vs8概率/capture误差0、最大gradient relative L2 1.901924378e-07，实际6张量全部finite非零，optimizer0/test0。底层通用compare的旧键名all11不是11张量证明，实际6已单独核。
- 此actual生产子自己的新恢复态gate也真实passed，参数479364/张量6，最大gradrel 1.901924378e-07，不能拿独立numeric收据充数。
- 189源码current/original一致、已Git发布并同步服务器。根launch_claim/started与rootmetadata/claim、第一段metadata/生产out/log均实际存在，不能重入cpu/numeric/start或旧helper。

各段实际producer rc0/exit后自己的独占科学验证prefix/189source/config/rawEMA/shapes/6grad/48E Adam/orderaffineMix(.2至100/.3至325/.1以后)、E326一次LR/statistics重启、保存IDlabelCM/NLL与wholeprefix首次最低验证EMA balancedNLL。自己的audit/completed SHA匹配才测selected模型。新参数digest最多一次，旧已测参数复用；继承旧二层7个已测selected模型，新增与总次数真实单列。测试反馈用于停止，明确test-informed开发，不称独立未见泛化或同预算因果残差优势。测试支持极不均，必须同时报告训练/验证/测试/macro/全逐类、实际与选中轮次。

本次只有一个新配方，不自动换类/另开候选/电子重试/更多深度/native/formal/lab/A100。源码只Git；rawJSON/PT/逐样本预测/完整日志只私有。普通健康静默，新真实失败或审核后的目标/平台终态通知一次；保留t18/每日09 ACTIVE。
