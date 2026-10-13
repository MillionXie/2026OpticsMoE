# 2／3／4层光路方案：实际启动与部署合同

用户确认主干2／3／4层，router另算，最终4层残差test总体准确率≥75%为验收目标。
75%尚未达到或验收，不保证；当前属于原筛选20类、旧test已见的开发实验。

实际新CPU父2437439，GPU2/1/4三子2437562/2437634/2437706，分别主干4/3/2 rho=.3；
三组rho0随后自动接续，最多三张项目卡。当前允许0/1/2/4，A100 GPU6排除；
人工作出3GiB→2GiB的显存执行调整，仍留800MiB余量，未修改模型/训练参数。
4层从已提交E7 raw/EMA/Adam/scheduler完整恢复，E8接续，未重播完成轮次。
父CUDA不可见，项目实际UUID/进程/源码及original Gitblob均已核对。

六模型真实CPU保存BMP回放概率误差全部0、零保持/捕获次数/无电子分类器/残差联合编码通过。
实际micro4概率差1.82167e-5未过1e-6门槛，失败保留；采用micro1、macro16累积更新，不放宽门槛。
六组真实GPU micro1预检通过，概率差0、梯度相对L2<=1.53e-6、mincos>=.99999982；
三生产子各独立再检查通过（含恢复raw模型）。真实部分E11/E8/E17的CPU状态审核通过：
6/10/11张相位梯度均有限非零，Adam86步/轮、调度轮数正确，E7历史前缀逐项一致。
四项Linux CPU协议检查通过：拒绝提前/重复锁，失败test数组加载仍保留claim，拒绝重复test。

按目前共享负载普通轮约50–70秒、每5轮额外train评估，六组均跑满150约还需4–6小时，
暂估今晚22:30至午夜前后完整审核；估计会随共享负载变化，验证早停可能提前。
初期精度不能称已解决过拟合或75%成功。max150/min50，MixUp.4、LS.08、仿射增强，
验证NLL降LR/早停，首次最低EMA NLL选模；报告真实总体/macro/NLL/support与trainvalgap。

[完整布局、归一化、非线性、联合BMP与物理验收边界](../../CALTECH_DEPLOY234_PROTOCOL_20261013.md)。
[布局图](optical_layout.png)、[六模型BMP证明](bmp_checks.json)、[真实数值门槛](numerical_preflight.json)、
[CPU接续/梯度证据](partial_cpu_receipt.json)、[实际启动与源清单](production_start.json)。

ABO参考是封存rank72仿真R@1 .83375、实拍 .81125；不是本任务分类成绩。
仿真与BMP共用保零tanh(abs/.5)，量化进入训练；相干rho通过联合振幅/相位加载，
固定CCD非线性用于模拟与实拍。原始权重/CCD/其他任务/实验室工程不改，实验室不连接。
离线一致性已验，实拍性能和当前光路LUT/ROI/方向/曝光仍需后续现场验收。

全六fit完成且实际父/子退出后，独立caltech_deploy234_finalize.py依次lock、首次suite；
检查现有全锁及suite/test markers，不重复、不删除、不自动retry。全锁及耐久claim
先于test数组，test不调方案。audit_caltech_deploy234.py仅核保存结果，导曲线、单test点、
全CM/support/summary PNG/SVG/PDF/CSV，实际视检后发布最终结果和静态相位部署包。

## 已完成2/4层残差阶段

两组150轮保存结果CPU审计和真实曲线已完成，test0、无选模锁；[阶段数据、PDF/CSV和解释](residual_L2L4_stage/README.md)。六组拟合与最终测试仍在等待完整屏障。
