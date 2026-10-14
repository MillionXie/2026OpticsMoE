# 有界CPU输入细节与保存误判诊断

独立工具`diagnose_resisc20_saved_input.py`来源61b21e392，实际一次CPU执行。新run/cpu_saved_input_diagnosis_20261014/checks.json和独占claim/log存在，禁止重跑/覆盖；144/bbdaf当前及原Gitblob、最终saved audit SHA、train cache/manifest与保存预测CM核通过。
没有模型构造/前向、val图像/test数组/校准拟合/GPU；只读取train缓存和已保存train/val预测，不改变输入合同、标签映射、选模或训练配置。
NPZ实际materialize11198个train图，seed20261014每类16张抽样合计320张进行输入算子统计；不是全数据或val/test图像统计。

实际编码路径为uint8RGB150 ->bicubicAA100/clamp ->affine100(本无增强检查theta=None) ->bilinearAA50 ->R/G/B/mean四tile拼成100 ->bilinear224 ->RMS.25。
独立重建该无增强编码与真实固定输入函数误差0，RMS.25检查通过。专家/router224、global478相位网络没有被缩小，但每个颜色通道进入编码前的图像细节只有50x50采样。

精确50通道再插值重建至150，与原train RGB150相比：均方误差均值.00355547；相对于逐通道去均值像素方差的误差均值24.7463%（中位22.3195%，p10/p90=10.7801/41.2167%）；相邻像素差平方能量保留均值8.2958%（中位8.0142%，p10/p90=3.8542/12.3873%）。
这是重采样像素/局部梯度度量，不能换算“丢失多少分类信息/准确率”、光照功率/吞吐或证明它是唯一原因；受插值平滑影响，原始256图的额外细节损失未在此320缓存样本统计里测量。
输入量化统计只针对学习相位乘子前的tanh(abs/.5)输入标量，非训练后完整各层M场/BMP；没有推理或实拍。

全保存预测核argmax/ID/labels/完整CM（val只已有CSVlabel，不读val图像或新test）：train11198两rho预测不同2403例，只有rho0正确465、只有rho.3正确868，两个都错4355；val1399预测不同311，只有rho0正确78、只有rho.3正确74，两个都错669。
对应最终train残差净增3.5989pp、val净差-.2859pp；观察不同错误样本不等于自动ensemble/重新标签映射/稳定跨seed收益。
完整逐类train/val最高错误去向和support保存于cpu_checks.json，低train的机场/篮球场等不能只从总体小gap推泛化解决。

该证据支持把输入细节作为下一项有界控制因素，尚未证明改编码会提高成绩。原完整网格保持，后续新GPU方案须明确范围预算授权，合格门槛/无test选模/部署匹配不放宽。
JSON产物经SHA传输；源码仅Git；旧所有audit/PT/data/history/图保留，本诊断不重跑完整科学审核。

本轮自动审批审核也拒绝了聚合Markdown的指定GitHub发布，理由为新实验payload/目的地授权不明确。原始JSON及聚合文档当前仅本地/服务器或本地保存，UI已询问是否允许仅聚合结论与方案发布；未得到明确回复前不再尝试外传。

2026-10-14用户现已明确答复“允许发布聚合报告和方案”，对应指定GitHub仓库MillionXie/2026OpticsMoE main。此前自动审批拒绝记录保留；本次仅发布聚合Markdown，不包含逐样本ID、完整误判JSON或传输包。新GPU训练仍未批准。
