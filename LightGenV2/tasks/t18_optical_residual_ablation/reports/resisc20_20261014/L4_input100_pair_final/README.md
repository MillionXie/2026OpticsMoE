# RESISC固定20类输入100配对最终结果

两组input100完整执行各自获批的有界预算后正常退出，保存科学CPU审核通过，但均未达到验证准入门槛。全部本任务GPU进程已退出；A100继续禁止使用，按人工要求停止的input50控制只保留26轮证据。测试评估次数为0，75%目标尚未验收。

## 验证选中结果

每组按全历史首次最低验证EMA balanced NLL选择，未按最高accuracy改选；训练数值来自同一选中EMA的保存预测。

| 输入与残差 | 实际轮数 | NLL选中轮 | 同选中EMA训练准确率 | 验证准确率 | 验证宏召回 | 验证balanced NLL | train-val差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Input100 rho0.3 | 114 | 111 | 57.1799% | 49.6069% | 49.5839% | 1.756604 | 7.5730pp |
| Input100 rho0 | 58 | 58 | 46.6869% | 41.1723% | 41.1511% | 1.957741 | 5.5146pp |

两组均未通过验证accuracy65%、macro60%及每类recall20%；train/val无零召回、末20轮无持续零类、gap15pp、全部四路非零和输入相关路由通过，仍不足以准入。
残差组机场train26.88%/val17.39%、篮球场36.96%/17.14%；无残差组机场19.89%/11.59%、篮球场19.64%/8.57%，另田径场val10%、停车场17.14%、环岛11.43%、湿地10%。部分类别连train也低，输入、拟合及优化原因尚未分离，不能把小gap或无零类称为泛化已解决。

## 停止时间与比较范围

两组均由time_budget停止。残差组114轮，最后history时间7262.19秒、含必需终点评估总7282.06秒；无残差组58轮，最后history7205.50秒、总7254.06秒。原规则在完整epoch开始前检查7200秒，完成已开始的一轮及必需终点评估后退出，因而不是精确秒级硬中断，没有启动下一轮或扩大预算。独立数学LR、early-stop与epoch开始时间边界审核通过。
GPU2残差组普通轮约60秒，GPU5无残差组约113秒、每5轮全train评估更久；相同时间额度导致实际更新轮数不同。不能将最终49.61%和41.17%的差距直接归因于残差。共同完成E58的验证值为rho.3 43.4596%、rho0 41.1723%，只是单seed同轮描述，未证明跨seed稳定收益。

A100上的input50 rho.3按人工要求在26完整轮后主动停止，best/last/history/grad与durable stop claim/receipt保留，未恢复或迁移。其E26 val39.8856%；input100 rho.3同E26 val39.6712%，rho0同E26 val37.9557%。A/B只能做已共同完成26轮的输入因素观察，不能声称完整预算input50对100已证明改善。A没有最终selected train预测，不生成该指标或完整臂准入。

## 完整保存科学审核

运行目录：resisc20_s17_mainL4_input50input100_rho_controls120_micro8_gpu256_20261014。训练闭包151来源d5158a53e4fc1a5a42fff54b25b8c06353f2c1e5的当前文件与original Git blob一致；报告HEAD不改变训练身份。
独立audit_resisc20_input_pair.py在151之外，真实cpu_remaining_pair_report/audit.json与专属exclusive dispatch及completed receipt均已成功一次。科学审核SHA：adf635d3f4c5876f43a4d2bf3aad15f1e4e953899fd5a07eeff600505ffb4b73.

B/C各完整source/data/cache/config/PT/同seed初相位/所有独立order-affine-MixUp尾14/Adam700乘实际E/scheduler/全部11phase finite nonzero/保存预测ID与标签、argmax、CM、support、macro和NLL通过；共同58轮的order及增强哈希相符。A只immutable26前缀核验，共同控制前缀26；A eligible=false、complete_budget=false。CPU构造3个模型核shape和seed，无模型forward、无test arrays或test评估。producer input_study_failed的唯一failure仍是人工停止A rc-15，B/C实际rc0；没有数值失败或自动重试。

原11198train/1399val/1403test、固定20类与划分、908552相位完整网络、expert/router224/global478、macro16/micro8/700updates尾14、增强与损失、光图及8bit出口均保持。输入100仅改变通道重采样细节，不改相位网格；rho.3为70%调制场加30%参考场振幅，非功率百分比。

## 图表与原始记录

[真实完整曲线](accuracy_macro_nll_curves.pdf)、[逐类召回](selected_class_recall.pdf)、[残差组完整验证CM](L4_tile100_rho03_validation_cm.pdf)、[无残差组完整验证CM](L4_tile100_rho0_validation_cm.pdf)、[人工停止控制的共同26轮前缀](common_control_prefix.pdf)均有PNG/SVG/CSV版本，全部经过Poppler真实渲染及五页实际视检，[聚合视检记录](visual_review.md)。
只从已审核保存结果绘图，未重训或复推。原始PT、data、history、失败与停止收据、审核、PDF marker及传输收据均保留；逐样本ID、完整机器审核和传输JSON仅在本地及服务器。

## 后续边界

没有eligible，不启动selected trained native、独立2/3、正式六组或一次test，不自动补训A、扩大预算、增加候选、放宽门槛或按accuracy换epoch。当前保持GPU释放，A100禁止使用；后续仅允许新的有界saved CPU机制诊断和具体修正方案准备，新GPU训练需另有明确范围授权。T18持续监控与人工要求的每天09点晨报保留，目标未成功。
