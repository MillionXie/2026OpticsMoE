# RESISC十五类完整网络配对最终结果

两组均按获批轮数或时间预算正常结束，CPU父146495与GPU2残差子147208/GPU5无残差子147277实际退出rc0，本项目GPU全部释放，A100继续禁止。保存科学CPU审核passed、四份最终PDF实际逐页视检passed，但两组均未达到验证准入门槛。没有test评估，75%目标未验收；没有自动延长预算、重试或启动更多候选。

## 验证选中结果

每组按全历史首次最低验证EMA balanced NLL选权重，未按最高accuracy改选；训练指标来自同一selected EMA保存预测。

| 组别 | 实际轮数 | NLL选中轮 | 同选中EMA训练准确率 | 验证准确率 | 验证宏召回 | 验证balanced NLL | 差距pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15 classes rho0.3 | 120 | 120 | 61.7052% | 51.4776% | 51.4465% | 1.551710 | 10.2276 |
| 15 classes rho0 | 78 | 78 | 53.3341% | 47.3785% | 47.3485% | 1.699253 | 5.9557 |

两组均failval65%、macro60%及各类recall20%；train/val无零类、实际末20轮所有类zero fraction0、gap15pp、全部四路非零及输入相关路由通过仍不足以准入。残差组机场train32.08%/val18.84%；无残差组机场26.88%/15.94%、铁路20.00%/12.86%、环岛14.64%/5.71%。保留较弱类别后，部分train也仍弱，拟合/输入/优化/增强正则原因尚未分离，不宣布唯一原因或充分收敛。

## 时间预算及可比较范围

残差组120轮、maximum_epochs先到，最后history5844.89秒，含selected终点评估总5860.26秒（约97.67分钟）。无残差组78轮、time_budget先到，最后history7274.73秒，含终点评估总7310.64秒（约121.84分钟）。原规则在完整epoch开始前检查7200秒，完成已开始的一轮与必需终点评估后退出，因此不称精确秒级硬上限；没有追加下一轮预算。
两卡普通轮约46/86秒、全train评估轮约60/121秒，速度不同导致实际120/78轮，最终51.48%与47.38%的差距不能直接归因于残差。共同E78验证48.6177%/47.3785%只为single-seed同轮描述，不证明稳定残差收益。

十五类是用户明确要求的新开发任务，固定randomseed20261014从原20类选15，没有按既有成绩删除最弱类或使用test，保留机场/篮球场。原样本划分、ID顺序与全部原20缓存保持；新train8398/val1049，test1053仅原metadata保留未读取数组。类别改变，不能把这次51.48%直接作为原20类75目标验收，也不能和20类49.61%当受控提升幅度。

## 真正保存科学审核

run：resisc15_s17_mainL4_input100_rho_pair120_micro8_gpu25_20261014。159运行闭包冻结4574b3e84b4426c42724b2bc0a144da04718d93e，current及original Git blobs真实一致。报告HEAD不是训练身份。
新audit_resisc15.py在cpu_report/audit.json真正passed，exclusive saved_cpu_audit_dispatch_20261014.json与correct SHA completedreceipt存在；auditSHA：eb6d1bf0f77ed131b270aaf193b6bd5557fc6ee10d9ccd8f591c06d4ad12016e.

独立核验原cache/新subsetmanifest/15名单及labelmap、config/PT/source/init/shapes/全部11phase finite nonzero、全部actual independentorder-affine-MixUp含tail14、Adam残差63000步/无残差40950步、scheduler120/78、数学LR/early-stop/time开始边界、保存预测ID与标签/argmax/full15CM/support/macroNLL/firstminNLL；共同78轮order及增强hash相符、同phaseinit。CPU构造2个模型核shape和seed，forward/testarrays/test均0；科学passed不同于eligible。
原expert/router224/global478、908552相位、四主干E/G/E/G加额外router、5capture/OEO/dense4/8bit jointM、input100及loss/增强/macro16/micro8保持；每轮525Adam更新尾14，只15个原坐标64x64检测窗口及15路归一化，不移动窗口、不学习电子head或缩网络。

## 图表与保留

[实际完整曲线](accuracy_macro_nll_curves.pdf)、[选中逐类召回](selected_class_recall.pdf)、[残差完整验证CM](L4_15_rho03_validation_cm.pdf)、[无残差完整验证CM](L4_15_rho0_validation_cm.pdf)，同时保留PNG/SVG/聚合CSV。只由已审核保存结果绘制，未重训或复推，Poppler逐PDF真实120dpi渲染并view_image全部四页实际检查，[聚合视检记录](visual_review.md)。
原数据、cache、best/last、history、grad、旧20/容量/正则/phase诊断、failed/partial/stopclaim及全部审核/PDFmarker/传输收据均保留。逐样本ID与完整机器审核/传输/visual JSON仅本地服务器，不发布；图和CSV仅聚合指标。

## 后续边界

本阶段有界精度准入失败，没有eligible，不启动selectedtrainednative、depth2/3、正式六组或test，不扩大预算/添加候选/进一步删类或按accuracy改epoch。当前保持GPU释放、A100禁止，允许有界saved CPU机制诊断及具体方案准备；新GPU训练需明确范围授权。严格门槛和真正native/逐类/depth/formal6/science/全durable once-test前置不放宽。持续T18与人工09晨报保留，按用户约定22:30统一汇报。
