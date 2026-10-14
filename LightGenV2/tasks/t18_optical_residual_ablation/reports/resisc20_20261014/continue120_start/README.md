# RESISC完整网络120轮有界接续：实际启动收据

用户明确批准继续至最多120轮，答复对应此前“每组最多新增2小时”的具体问题。
原完整四层rho0/.3两组从第30轮raw/EMA/Adam/scheduler恢复，第31轮开始；原bestE29、完整历史与固定验证NLL选模保留。
新run：`resisc20_s17_mainL4_rho0rho03_fullres_continue120_micro8_gpu256_20261014`。
运行144文件来源冻结`bbdaf0a852aaf9001b0c82d74dcdd592ea9865ad`；原141/75be、136/513及所有旧run/PT/history/数据不改。

实际2026-10-14北京时间14:28:50 CPU父3804063启动；GPU6 rho.3子3804643、GPU2 rho0子3804652。
CPU准入检查通过：实际Linux导入/预算绑定、仅两预算字段变化、错误case和错误旧run审核均创建out前拒绝；无模型构造/前向/test/GPU。
生产两子对真实恢复raw、20类及尾14的micro1/8数值检查通过：概率/capture误差0，梯度相对误差最大6.98e-7，11相位梯度均有限非零。
恢复收据核raw/EMA与原E30 SHA一致，optimizer/EMA/scheduler均未重置，第一新轮31。

首完整E31实测：rho0 val36.7405%、macro36.7195%、balancedNLL2.110765、56.99秒；rho.3 val40.6719%、macro40.6480%、balancedNLL1.998983、61.23秒。
该单轮波动不能当已选最终EMA、稳定残差收益或准确率解决。两组仍实际运行，fail0/test0，只GPU2/6各一个项目子。
每轮完整11198train/1399val、macro16/micro8、700Adam更新、尾14以及908552相位参数/原光图保持；每5轮完整train评估仍保留。

新增预算明确披露：全局最多120轮（含已经完成30），每组新增7200拟合秒（两组合计最多4新增GPU小时），或者既有early-stop先到即停。
只预算字段变化，原网络/数据/类别/split/MixUp.6/LS.08/TV.005/capture.05/EMA.95/增强/LR规则不变；无第三candidate/自动重试/扩预算/test。
引擎在完整epoch开始前检查时间，已开始一轮及必需获选端点评估完成后退出，时间界限的epoch粒度已在协议披露。
不保证65%验证门槛或75%最终test达标。当前仅扩大有界拟合预算，不自动正式六组或独立2/3。

原门槛val65/macro60/每类20/nozero/末20zero不过半/gap15/all4/inputvariation/perclassrouting/获选训练BMP不放宽。
全部两fit完成、真实父/子退出后，独占CPU `audit_resisc20_continue120.py` 验144当前原Gitblob、所有E1-30字节/恢复、全部实际E的order/affine/MixUp/尾14、Adam700E/scheduler/11grad、数据与保存预测ID/标签/CM/support/macroNLL。
私有审核调度器`--run continue120 --action audit-ready`已准备但尚未执行；只有无audit/claim/failure及实际屏障才执行，失败保留，不自动重复。
相应完整结果绘图及获选native/独立深度尚未完成，不能拿旧30轮四图或旧freshproof代替。

本目录只包含真实JSON产物及逐SHA传输清单，不传输源码/PT；所有源经Git发布。
