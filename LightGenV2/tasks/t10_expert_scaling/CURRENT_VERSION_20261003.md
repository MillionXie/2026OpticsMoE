# T10 当前实际工程与结果入口

2026-10-03工程整理；只核验现有结果，不训练、不启动历史队列、不重新评估测试集。

## 源码身份

实际服务器工程：`/DATA/DATA1/guest3/t10_corrected_20260918`。
原HEAD：`5a1178ef09be40f45b58a20a8633a6e1add2d75d`；它不包含全部实际运行源码。
1份修改的 train.py 和8份未跟踪源码已保留于Git恢复快照：
`38afd43c1e28aeaff28e33bad58177ae09a0f156`，
`refs/archive/server-authoritative-20261003/t10-runtime-overlay`。
没有切换服务器HEAD、暂存用户文件或改变原工程。

main按此快照追加29份源码/配置/协议/设计文件；不整分支合并。
精确清单见仓库 `maintenance/storage/T10_PINNED_RUNTIME_20261003.json`。
设计CSV是原计划文件，不是新实验结果；CPU自检不冒充仿真准确率。

## 四组结果的位置与区别

以下都在 `/DATA/DATA1/guest3/`，原位保留，没有搬进Git：

| 原结果目录 | 本轮观察到的训练result.json / test_result.json | 用途 |
| --- | ---: | --- |
| t10_corrected_20260918_runs | 18 / 14 | 修正动态路由后的 Kather / DeepWeeds 扩展 |
| t10_adrenal_20260919_runs | 30 / 0 | 病理相关专家数/路由对照；不能当已测TEST |
| t10_topk32_20260920_runs | 42 / 42 | 专家数与Top-k扩展、必要D2NN对照 |
| t10_fixed478_3datasets_20260921_runs | 35 / 28 | 固定478光场三数据集、多seed对照；还涉及复用条目 |

文件数不是完整设计点数。2026-10-05已逐一核对固定478的45项jobs：8项Kather复用
旧目录（其中k3还有一层别名），所以按任务读取为43份训练结果、36份TEST结果，
而直接目录扫描仍是上表35／28。43份训练报告和36份TEST报告均匹配当前best SHA；
36份TEST对应锁定checkpoint均匹配，28份用本组原锁、8份用原full-grid锁，未重评。
全部45项best/last均存在，但两项organ D2NN seed27／37缺训练完成报告；九项
同孔径D2NN（3数据集×3seed）缺TEST报告，不能把完整矩阵说成已完成。
旧status仍写“training/43完成”，两项原PID已不存在，未发现相关T10进程。
只保留并标明历史状态，不自动重启。逐项私有收据及公开摘要见仓库
`maintenance/storage/T10_RESULT_REUSE_AUDIT_20261005.json`。
这项审计也说明旧topk32／corrected结果目录仍被复用，不能作为冗余副本删除。

固定478目录35个原训练结果的best SHA均与记录匹配，best/last都保留。
已有逐TEST报告、数据SHA、测试锁定清单也原位保护。
资产逐项证据：`maintenance/storage/T10_ASSET_AUDIT_20261003.json`。
本轮不从TEST表中挑最高值宣布最终版本，也不把不同孔径/带宽/seed混成一个数字。

## 本轮验证与边界

18份实际Python源码编译通过；4/9/16/25/36/49/100专家几何检查通过。
4专家、4层的MoE及三种D2NN在CPU上概率归一化、有限梯度、确定性复载检查通过；
MoE Top-1 router分类梯度非零。这是有界结构检查，不是新性能或测速结果。
尝试2层最小检查时参数匹配几何不成立，随后用原正式4层检查通过，未改科学代码。

旧plan.py的完整设计校验入口存在循环变量作用域问题，原版本仍保留在上述源码快照。
2026-10-05单独审查并采用本地已有修复：49端口参考检查不再读取未定义变量，另验证
532nm／17μm／10cm物理合同。六项标准库CPU测试通过；所有专家数的几何、路由区域、
k值及主／pilot／固定global矩阵与原运行源码相同。没有重新生成或覆盖历史表，
没有训练、模型或数据改动；这只是设计校验入口修复，不是完整实验复现证明。
原／新源码SHA及发布链记录在 `maintenance/storage/TASK_SOURCE_EVOLUTION_20261005.json`。

需要完整重训时必须先核对数据/预算/当前GPU授权，不照抄历史GPU0/1/2或A100命令。
全部原数据、划分、best/last、逐样本结果、路由统计及所有测速/功耗继续保护。
当前状态：实际源码已收敛、固定478资产部分核验；完整结果矩阵与数据闭包仍待收尾。

本地独有的三个历史Top-k报告转换工具已保留原版并归主线，当前只向明确的新目录
写派生报告，不原位修改证据包。入口、16项锁定/40项全网格的区别与CPU测试范围见
[历史绘图工具](docs/HISTORICAL_HANDOFF_TOOLS.md)。这不是补跑缺失结果或更换服务器训练源码。

2026-10-06补充：固定478三份实际NPZ逐字节SHA与36份既有TEST报告的数据字段一致；
NPY头部确认Kather train3496/val752/test752，Blood train11959/val1712，
Organ train12975/val2392，RGB uint8。后两份训练缓存不含TEST，TEST另从原始归档读取；
不能把训练缓存SHA一致当成完整TEST来源闭包。原始测试归档及有序ID仍待核验。
本轮仅流式哈希和NPY头部读取，没有解码图片、加载模型或复评。
证据：仓库`maintenance/storage/T10_DATA_ARCHIVE_BINDING_20261006.json`。

同日继续核验原TEST来源：Blood3421、Organ8216的原归档SHA、manifest及有序ID
与各12份现有报告完全对应；Kather752有序ID与12份报告对应，但旧报告没有单列
test_source_sha256字段，不能补造历史字段。三组ID均唯一、与manifest的TRAIN/VAL ID无交集。
Blood/Organ ID由原评估协议的dataset/split/索引构造，不证明图像近重复或患者独立性。
只读取标签和ID，没有读取图像数组；缺失的九项TEST报告仍缺失，没有替代评估。
证据：仓库`maintenance/storage/T10_TEST_SOURCE_BINDING_20261006.json`。

已有36份TEST报告的预测NPZ全部存在：有序ID、标签、分类分数尺寸和有限性、
argmax预测均对应；从已有预测计数所得accuracy与原报告在1e-6内一致。
逐预测文件SHA及复用真实目录已登记于
`maintenance/storage/T10_PREDICTION_BINDING_20261006.json`。
这不是重新执行模型，也不补齐缺失九项TEST或证明患者/近重复独立性。
