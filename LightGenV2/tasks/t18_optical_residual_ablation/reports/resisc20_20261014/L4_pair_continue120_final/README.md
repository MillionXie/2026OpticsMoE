# RESISC原完整网络120轮：已完成，精度准入失败

两组各global120完整结束，CPU父3804063及GPU2 rho0子3804652/GPU6 rho.3子3804643实际退出rc0，所有项目GPU2/5/6释放，failures0/test0。
原E1-21单图、E22-30提速及E31-120人工新增有界预算的全部历史保留。原best/last/data/test/failed/run不覆盖。
144训练来源冻结bbdaf0a852aaf9001b0c82d74dcdd592ea9865ad，当前文件与原Gitblob一致；原141/75be、136/513、135/126/130不变。报告HEAD不代表运行源。

完整saved CPU科学审核及budget_continuation_cpu_checks通过：真实E30父审核/migration/dispatchSHA、E1-30 history/grad/order/affineMixUp字节、raw/EMA恢复/no AdamEMA scheduler reset/首新E31、仅两预算配置差异；全部120独立order/affine/MixUp含尾14、Adam84000/scheduler120、11相位梯度有限非零、同seed初态与配对共同增强、PT/config/cache/manifest/保存预测ID标签argmax完整CMsupportmacroNLL。
审核无模型前向、test数组或test推理；CPU形状/init验证允许模型构造。独占claim和completedreceipt存在，禁止重复auditor/dispatcher。科学审核通过与性能合格分开。

按预定firstminimum validation EMA balancedNLL，两组均选E120：

| rho | Train accuracy % | Val accuracy % | Val macro % | Val bNLL | Gap pp |
|---|---:|---:|---:|---:|---:|
| 0 | 53.3577 | 46.8906 | 46.8623 | 1.854189 | 6.4671 |
| .3 | 56.9566 | 46.6047 | 46.5776 | 1.790458 | 10.3519 |

均fail selectedval65/macro60/eachrecall20，all_pair_eligible=false；nozero/last20zero<=half/gap15/all4nonzero及输入变化通过，不等于泛化解决。
每类机场train12.90/10.93%、val7.25/8.70%；篮球场train21.61/28.93%、val8.57/15.71%；rho0环岛val11.43%、rho.3田径场val17.14%。这些类别连train也弱，支持拟合/输入/优化仍有问题，但不能由本表证明唯一原因。
未按某一最高accuracy轮反选：E108rho.3val47.82%是当时暂选，后来E120NLL更低而accuracy46.60%；严格保留原选模规则。

与原30选中EMA比较，同一完整网络继续训练后val37.96->46.89%、40.89->46.60%，改善8.93/5.72pp仍远低门槛。
当前rho.3选中accuracy比rho0低0.2859pp，valNLL较好；单seed不能声称稳定残差准确率收益或Caltech受控恢复。
训练53-57%仍不高，部分trainval差扩大；输入信息/增强/优化/容量未分离。late_validation_trends保存末10与前10NLL，只是趋势，不证明充分收敛或无限延长必达65/75。

每轮完整11198train/1399val、原908552phase、macro16/micro8/700updates/tail14和光图/8bit部署合同保持。实际新增fit时间rho0 5647.63秒、rho.3 5898.89秒，均<7200；全局120上限先到，无预算扩展/第三候选/自动retry。
保持原数据/类别/split，test未评估，不以test选方案；精确像素分组非地理/全部近重复独立，RESISC任务与旧Caltech不同。

4PDF与PNG/SVG/CSV仅从已审核保存数据生成，逐SHA产物传输，Poppler每PDF一页真实120dpi渲染、view_image全部四页实际视检passed。
曲线包含每轮val/稀疏train/firstminNLLval星/同EMAtrain菱形；完整valCM/逐类support/recall/routing/门槛CSV，不造test点/误差棒。
真实图与PDFmarker/helper已执行一次，原旧所有图保留，不重绘/重复marker/重训/复推模型。visual_review不是仅renderreceipt。

本120有界实验性能失败，目标任务尚未成功；没有合格profile，未启动条件2/3/selectedtrainednative/正式六组/selection/suite/testclaim，75%最终test仍未验收。
不得放宽门槛/更换accuracy选模/更多删类/自动加预算/重启旧队列。下一步只继续有界CPU输入/保存结果拟合机制诊断和具体修正方案准备，新的GPU搜索需新的明确范围授权。
最终网络大小保持；未来真实selectedtrainednative与独立2/3末E几何逐类准入、全正式6saved科学审核/actualexit/durable全锁后首次单test的要求不变。monitor与09:00晨报保留。
