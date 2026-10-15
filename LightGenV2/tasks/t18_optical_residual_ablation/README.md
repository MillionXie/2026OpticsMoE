## 仅残差接续已核验快照：三层差7.20pp达目标，二/四层继续（2026-10-15）

| 深度 | 残差实际轮数/选中E | 残差训练/验证/测试 | 无残差100轮训练/验证/测试 | 总体test差 | test macro差 | 目标及状态 |
|---|---:|---:|---:|---:|---:|---|
| 2 | 175 / 172 | 75.4271/71.9745/78.5597% | 67.9369/64.9682/77.6869% | +0.8729pp | +1.7591pp | 4pp：未达，接续中 |
| 3 | 125 / 123 | 90.9330/81.5287/87.4523% | 79.8949/75.1592/80.2510% | +7.2013pp | +9.3718pp | 6pp：已达并停训 |
| 4 | 175 / 165 | 99.3430/86.6242/92.8532% | 84.7569/78.3439/83.9062% | +8.9471pp | +9.2405pp | 10pp：未达，接续中 |

双方已先完整100；无残差固定，不再训练。残差额外轮数单列、无总epoch上限。三层125选123已达6pp且worker实际退出；二/四层正在后续保存段，表为最新已核验175快照，未达4/10目标不称成功。测试知情开发，不独立泛化、不称同预算因果优势；四层过拟合仍在，二层正则暂未有效提高测试。全部收据/来源及局限在[接续快照](reports/caltech10_equal100_20261015/residual_progress/README.md)。

## 新双方100轮已完整科学/测试通过，仅残差接续已自动放行（2026-10-15）

| 深度 | 有/无实际轮数 | 有/无选中E | 有残差训练/验证/测试 | 无残差训练/验证/测试 | 总体test差 | macro差 | 目标 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 100/100 | 100/100 | 96.5834/85.9873/92.1440% | 84.7569/78.3439/83.9062% | +8.2379pp | +5.7975pp | 10pp |
| 3 | 100/100 | 100/100 | 89.4875/80.8917/87.1249% | 79.8949/75.1592/80.2510% | +6.8740pp | +6.5242pp | 6pp |
| 2 | 100/100 | 98/98 | 73.1932/71.3376/78.7234% | 67.9369/64.9682/77.6869% | +1.0366pp | +0.1850pp | 4pp |

六臂实际100，二层选E98不是提前停止。无残差现冻结，只有残差完整恢复E100后继续、无总epoch上限，2/3/4层总体test目标4/6/10pp。额外残差预算单列，非新同预算因果优势。后续测试反馈控制停止，明确test-informed开发。181 metadata实际训练Git d61142f91605bd33fbbd5180541c4d488aeeb5ac，181首次发布e87的源SHA一致；185接续b03fcdfa，报告HEAD不同。[全部训练/验证/测试、macro、逐类和审核身份](reports/caltech10_equal100_20261015/baseline_final/README.md)。t18/09 ACTIVE，旧150已结束、不能冒充100。

## 最新人工纠正：双方100，无残差冻结，残差单独不限总轮数接续（2026-10-15）

旧双方150属于助手执行偏差，只作为历史，不能冒充100。新基线181首次发布e87caff8、metadata实际d61142f9（源SHA一致）实际启动CPU2419295、四张非A1000/2/4/5上先4臂100轮，二层排队；每臂4800Adam，不early，保存lastE100。新接续185源b03fcdfa控制器2446199已实际启动等待基线真正全通过，之后仅残差完整恢复E100继续，中度MixUp.3，每25轮保存/审核并继续，不设总epoch cap。无残差冻结不追加；二/三/四层总体test差目标4/6/10pp，达不到如实说，不保证成功。

新结果比较“无残差100”与“残差100+额外微调”，不是同预算因果消融。测试反馈用于停止开发，明确test-informed，不称独立未见泛化。原类别/数据/全光模型不变，不削弱对照，不换有利epoch。所有结果必须并列train/val/test/macro、实际轮数/选中轮次。

旧测试远高验证已重新核查：同模型但样本类别占比大异；测试91.27%来自三个大类。逐图ID/label/像素SHA核后跨角色精确重复为0，但不证明同对象/近重复独立。macro测试仍更高，不能只归类别不均衡，完整分析和全部训练精度在[100轮纠正与数据核查](reports/caltech10_equal100_20261015/correction/README.md)。t18/09 ACTIVE，旧源码/PT/失败/claims/图继续保护。下方150轮完成后的禁止继续是历史，由这次明确人工授权覆盖，仅这两个新run。

## 新十类四/三/二层与一次正则已全部完整科学/冻结测试通过（2026-10-15）

本轮CPU2314182与全部GPU真退出、rootstate complete_science_and_frozen_tests_passed，全部producer原始rc0、各独占科学和每个冻结测试completedSHA正确，GPU0/2/4/5项目0/A1000。四层原配对各150选149测试后一次正则50未改善NLL、所选参数相同故复用原test不重测；三层/二层各freshrho.3/0完整150、同valfirstminimum冻结各test一次。177训练来源084e7be07be831d627bbafa98f87a378a2cf97a1=current/original，报告HEAD不同。

| 深度 | 有/无选中E | 有/无验证 | 有残差test | 无残差test | test差 | macro差 |
|---|---:|---:|---:|---:|---:|---:|
| 4层 | 149/149 | 86.62/78.98% | 92.7987% | 88.3797% | +4.4190pp | +3.4698pp |
| 3层 | 149/149 | 82.17/75.80% | 88.8707% | 82.7605% | +6.1102pp | +10.4842pp |
| 2层 | 146/149 | 71.34/68.79% | 79.0507% | 79.4326% | -0.3819pp | -1.1844pp |

本轮四/三/二层都未达到测试10个百分点目标。本表测试为固定十类验证知情开发、先前Caltech test已观察，支持极不均，macro及逐类见[完整最终报告](reports/caltech10_followup_20261015/final/README.md)。所有结果按原选模、不用test换模型/类别、不扩大差距或削弱对照，不追加任何自动实验或重测，旧source/PT/data/history/claims保持，t18/09 ACTIVE保留。此完成事实覆盖下方仍在训/待审/条件可执行旧文字。

## 最新：四层正则无提升并复用原测试；三层、二层四卡已实际在训（2026-10-15）

原冻结十类四层各一次仿真test92.7987/88.3797差+4.4190pp、macro89.7096/86.2398差+3.4698，未10；用户已本turn获告，详见[测试与逐类](reports/caltech10_gap_20261015/paired_once_test/README.md)。一次MixUp.4正则接续已完整E151-200、GPU子2315182actualrc0退出，新增fit238.30716秒，自己独占SCI SHA4e2ae5722d27c63266714a1de4c0a49b67feac3e8a414a8ab9e818fc3daa0c9a passed。新50最低valNLL.5828911649未胜父E149 .5708938839，按规则仍选E149、model参数digest实际相同，真实regularize_test_reused复用原test、无第二次forward；不声称缓解过拟合成功。[完整负结果](reports/caltech10_followup_20261015/regularize_final/README.md)。

按人工顺序已自动开始三/二层fresh同150四臂：L3rho.3 GPU0子2323784，L3rho0 GPU2子2323855，L2rho.3 GPU4子2323921，L2rho0 GPU5子2323987；CPU campaign2314182、177实际训练来源084e7be07be831d627bbafa98f87a378a2cf97a1=current/original，四own gate已真passed，global4一卡1/A1000，无fail，非临时成绩最终验收。各150/7200Adam/3600fit秒/noearly，own独占保存science通过后各一次val冻结测试。source/PT/data/claims保持，不手动重审/另起队列/重试/扩预算或test选模，10pp不保证。[四卡实际启动与验收合同](reports/caltech10_followup_20261015/depth_pairs_start/README.md)。

本节覆盖下方正则仍在训/浅层pending等旧文字；本新授权继续有效仅这四臂与各一次科学/测试，不扩范围。t18/09 ACTIVE保留，普通健康静默，真完成/新失败通知。

## 新人工授权：冻结十类四层一次配对测试完成，继续残差正则与三/二层（2026-10-15）

当前冻结E149有/无残差模型已各一次仿真test、真实rc0退出并保存预测独立核验passed：1833张有残差92.7987%(1701)、无残差88.3797%(1620)，测试差+4.4190pp；macro89.7096/86.2398差+3.4698pp，未达到10pp目标。原验证+7.6433不能当test差。类别开发/此前Caltech test观察过，支持不均须看macro与逐类。详见[真实冻结测试报告](reports/caltech10_gap_20261015/paired_once_test/README.md)。不得重这两个模型测试或用test换类别/epoch/反调参。

人工明确允许继续缓解有残差过拟合、完成后跑三层与二层、最多四张非A100。这覆盖旧禁止新训练仅此有界followup：caltech10_regularize_then_L3L2_s17_20261015，177来源084e7be07be831d627bbafa98f87a378a2cf97a1；先残差full E150 raw/EMA/Adam7200/scheduler/LR.001恢复，E151-200仅MixUp.2->.4，无额外手动LR减半/重置、1800新增fit秒、无early、firstmin全prefix+50验证EMA balancedNLL选；旧已测权重若仍被选则复用已测结果不再forward，新权重冻好最多一次测试并复用固定对照。额外残差预算不是新同预算因果优势。

随后fresh三层E-G-E和二层E-G，各rho.3/0同150完整无early、各3600fit秒/7200Adam，原10类/角色/坐标/input100/4专家/8bit保持，不用test选模。各配对自己的独占科学passed才各一次冻测。新CPU恢复态+depth2/3黑十路归一8synthetic forward准入SHA17e3db1d52460bb4c35a093bf958230a8051d314d3b97847c84681e52ca452bb、GPU0五种depth/rho/恢复态all10/macro16/tail9数值与梯度准入SHA79629de6758c972b3eb1376d856f3abde357c0fcce09b2b565b45fe8a4691a43实际一次passed/optimizer0/test0，不能重入。GPU0/2/4/5全球最多4一卡一子、Torch2GiB+800MiB、A100禁止、不杀外国任务，总新增fit<=4.5GPU小时。失败保留停下游不retry/放宽/扩预算，不无限追10pp。t18与09 ACTIVE保留，普通健康静默、新真实失败/完成主动报。

此新范围覆盖下方两条优化已结束之后不再实验等旧文字；旧sources/PT/data/history/claims/失败/markers继续保护。全部原始JSON/逐样本ID/PT/完整日志私有，源码仅Git，报告HEAD不是实际训练来源。

## 本轮两条优化均全完成：新Caltech十类验证86.62%，配对残差差+7.64pp（2026-10-15）

新固定十类双方完整150且独占科学第一次passed，CPU campaign/queue与两GPU子actualexit/原始rc0，所有T18GPU项目0/A1000/test0。按firstminimum验证EMA balancedNLL均选E149：有残差train99.3430/val86.6242/macro86.3258/NLL.5708938839；无残差train90.2760/val78.9809/macro78.9135/NLL.8056351399，净多12/157正确，验证差+7.6433pp、macro+7.4123pp。真实同150预算/单seed17、无电子分类器，未测试；验证知情筛类新开发任务，不是原20或完整101达到80、也不称完全未见泛化。训练验证gap12.7188/11.2951仍在，不能称无过拟合。

固定Faces/Leopards/Motorbikes/airplanes/bonsai/grand_piano/revolver/schooner/trilobite/yin_yang，未更换；新169训练Git259cdcfdec6edc9b8ca29f9c7ddf18fb3668cffa=current/original。scienceSHA617825b1c2d628b7a2cefdd13874d4c4ca3e468ed16338def5561b3d723051db/completed实际一致。完整真实曲线/逐类与来源见[新十类最终报告](reports/caltech10_gap_20261015/final/README.md)。

原20双方同200也已全结束/自己SCIpassed，68.1979/62.8975、差5.3004pp，未80；不能与新十类跨任务拼gap。原20完整200曲线和逐类见[原20续训最终报告](reports/caltech20_continue200_20261015/final/README.md)。已按新人工三张非A100并行完成两个有界方案，现不再运行训练/审核/准备/图/markers/重试/换类/新候选/扩预算或test。旧全部源码/PT/data/history/claims/失败保持。t18和09 ACTIVE保留，正常完成/已通知结果静默；本节覆盖下方仍在训/待审核/条件可执行旧文字。

## 原20同200轮已完成；新十类150轮仍在训（2026-10-15）

原20双方真实完成200并独占科学passed/CPU与两GPU退出：有残差选E193训练94.5335/验证68.1979/macro65.7161%，无残差选E200训练77.7697/验证62.8975/macro60.2210%，差+5.3004pp（原100轮+7.0671）。训练-验证差26.3356/14.8722pp，双方继续提高而对照追上，原20未80、过拟合仍明显；不是test。完整200轮曲线与逐类在[新最终报告](reports/caltech20_continue200_20261015/final/README.md)，科学SHA3d381ac398a03e8b7a4ede3476049c238cbc4ed12148deff8874e9ddec28525d。

新固定十类fresh150配对已两组实际在训：GPU4残差2221711、GPU2无残差2229841；原20已释放GPU5，当前两卡/最多三卡，A1000。最新残差E120暂86.6242%，对照E62暂71.9745%，不同轮次不得拼最终差距。两组各150、同firstminimum规则和自己的独占科学通过后才验收；当前不是测试80或最终80成功。训练来源259cdcf不同报告HEAD。此事实覆盖下方A仍在训/B对照pending等旧状态，禁止重起A或B/重审/扩预算/自动test；t18/09 ACTIVE保留。

## 最新人工允许三张非A100继续优化：原20接续200 + 一次新十类开发（2026-10-15）

用户要求真实曲线、分析收敛/过拟合、继续冲80并保残差优势，必要时可挑类别。原100轮[真实曲线PNG/SVG/逐点CSV与诊断](reports/caltech20_current4_20261015/curve_review/README.md)已实际生成、view_image QA并在本chat展示。末段验证NLL继续降、LR一直.002，但残差train-val gap扩大，所以既未充分收敛也有泛化差，不能保证加轮数到80。此新人工范围覆盖下方旧完成后的禁止继续，仅下面两个有界方案，不自动test或无限候选。

**A：原20同预算继续。** caltech20_current4_s17_pair_continue200_gpu25_20261015，新160训练来源c266b7973b105be6caa0dcdb95cfe2fe7cba2abc。双方E100 raw/EMA/Adam8600/scheduler100/LR.002完整恢复，prefix100不重播/重置，保持配置接续E101-200，无early、各3600新增fitsec上限。新CPU状态准入真passed/0forward，双方生产恢复态gate真passed。CPU campaign2199729/queue2200431，GPU2残差2201158/GPU5对照2201161实际在训；最近E168/E165仅临时观察，不跨E拼gain。[实际启动和独占科学条件](reports/caltech20_continue200_20261015/start/README.md)。

**B：新十类开发配对。** caltech10_gap_s17_current4_pair150_gpu425_20261015，新169来源259cdcfdec6edc9b8ca29f9c7ddf18fb3668cffa。一次验证规则兼顾自身recall和非负残差差距，固定Faces/Leopards/Motorbikes/airplanes/bonsai/grand_piano/revolver/schooner/trilobite/yin_yang；原坐标窗口/划分保持，train761/val157，新任务不能当原20/完整101达到80，不把挑旧行均值当训练成绩。fresh双方150无early、各3600fitsec上限、每E48Adam尾9。metadata/新CPU native14forward黑10归一/新GPU4两rho数值门槛均真一次passed，0test。CPU campaign2220246/queue2220984、GPU4残差2221711实际在训，自己的生产gate真passed；对照pending待允许卡空槽自动启动。与A合计GPU2/5/4三子一卡一子、A1000，不抢杀外国任务。[固定名单、实际准入/启动及边界](reports/caltech10_gap_20261015/start/README.md)。

两个新source身份均真实current/original，别与旧RESISC160/169混淆。各自campaign负责完整约定轮数及实际rc0退出后的唯一独占保存科学，自己的audit/completedSHA和父退出才完成，不手动再审/启动或扩预算。每条真完成后报告同预算train/val/macro/NLL/逐类正负结果，类别开发与原20分开。旧全部source/data/PT/history/失败/claims/markers保持；rawJSON/逐样本ID/PT/完整日志私有。t18/09 ACTIVE保留，普通健康静默。

**下方旧已完成100轮、未授权继续等文字均为历史，由本次人工范围覆盖；旧审核/测试/图不能重做。**

## Caltech筛选20类当前四层已完整100轮并独立科学通过：验证差+7.0671pp（2026-10-15）

两组均100轮/maximum_epochs、无early，GPUproducer原始rc0并实际退出，CPU campaign/queue全部退出；GPU2/5项目0/A1000/test0。自己的独占保存科学第一次passed，audit SHA588b5a7046f4f7e4ca6349459d9515874c153a6c31e2a943e48203bed54508a0与completed真实匹配，额外独立审核助手SHA/Git也实际核一致。按首次最低验证EMA balancedNLL均选E100。

当前四专家/四主层纯光：有残差训练78.0612%、验证62.1908%、macro59.7451%、bNLL1.6403502369；无残差训练61.3703%、验证55.1237%、macro51.8116%、bNLL1.9969947965。**验证残差领先7.0671个百分点（176/283对156/283，净多20正确），macro领先7.9335pp**。这是本轮同100预算真实配对差距，不套旧九专家测试成绩。本轮没有测试集结果，残差验证未80；双方原严格开发精度门槛均未通过，不能把科学通过冒称精度验收成功。

末十验证NLL仍改善3.23%/2.08%，但train-val差15.87/6.25pp，残差拟合更强、泛化差也更大；不保证充分收敛或已解决过拟合。两卡并行拟合约14分钟，全部来源/权重/历史/失败/claims保留。训练来源仍763e6a242a3fe27a513044dc6a6282a0ef574b7d，报告HEAD不同训练身份。详见[最终完整指标、逐类、预算和审核身份](reports/caltech20_current4_20261015/final/README.md)。

此次一对授权已结束，不自动补训/新候选/测试/电子支路/改类/额外深度/A100/native/formal6/lab/图作者。t18与09晨报ACTIVE保留，完成消息一次，普通完成状态静默。

**下方启动/仍在训/待审核等状态保留作历史，以本节最终事实为准。**

## 最新人工选择差距优先，Caltech筛选20类当前四层配对已实际启动（2026-10-15）

用户明确切回筛选20类Caltech，要求尽快训练当前四层；已按当前四专家/四主层的rho.3与rho0 fresh同100轮、无early一次新配对实际启动。此新人工范围覆盖下方旧173等完成后“不再训练”的文字仅本轮；旧实验/数据/PT/失败/claims/markers继续保护，不重做。

run `caltech20_screened_current4_s17_pair100_input100_mixup02_gpu25_20261015`，实际训练来源763e6a242a3fe27a513044dc6a6282a0ef574b7d，155登记来源current/original一致；额外审核助手Git/SHA身份单独记录，不改冻结运行来源。CPU campaign2104530/queue2105395，GPU2残差2106051/GPU5无残差2106196实际运行，active2/pending0/fail0/A1000/test0，PID须实时核。新CPU native14forward/20黑图归一化与真实ID支持、新GPU2全20/macro16/tail12数值检查和两生产own gate都已真实一次passed，禁止重入准入/启动。

原20名单与train1372/val283/test2064原划分复用，不重选类或读取test。当前全908552光相位/四专家/E-G-E-G+router/RGB100/8bit/固定20窗口，无电子可训练分类支路。相同seed17/init/order/affineMix及公共MixUp.2/LS.08/TV.05/capture.2/EMA.95/lr.002，仅rho不同。完整100各8600Adam，每组10800拟合秒安全限、不扩预算/retry，只GPU2/5各1。

最近残差E10 val28.62%、无残差E9 val14.13%仅早期健康，不能跨E拼最终差距。初期约9秒/轮，预计100轮约15–20分钟，非保证。双方实际100/退出后campaign自己独占保存科学，CPU0forward/test；真正通过并全退出才主动汇报同预算train/val/macro/NLL/逐类差距。旧九专家Caltech优势只历史参考，不保证迁移；本轮不自动测试/更多候选/电子分支/正式6/native/lab/A100。详见[真实启动、数据、准入和审核身份](reports/caltech20_current4_20261015/start/README.md)与[本轮合同](CALTECH20_CURRENT4_PROTOCOL_20261015.md)。t18与09晨报ACTIVE保留，普通健康静默。

**以下旧完成结果和停止范围均保留作历史，本轮最新状态以本节为准。**

## 原十类一致接续已完整140轮并通过独立核验：验证78.7143%（2026-10-15）

原十类纯光有光学残差E101–140已全部完成，CPU queue/GPU producer实际退出、producer rc0，自己的独占保存科学首次passed，audit SHA `139891ea639703b8c7e02d23a95db04df626570d582bbb1f5889cabd3071bb7d` 与completed真实一致。GPU2/5项目释放、A1000/test0。按首次最低验证EMA balancedNLL选E139（不是139轮提前停止）：训练89.8750%、验证/macro78.7143%、NLL0.7545565714，训练/验证差11.1607个百分点。相较父E100的78.1429%仅提高+0.5714个百分点，仍未80；本次权重没有新测试。

无残差历史前100轮76.2857%和旧120轮选E113的77.1429%仅固定参考，数值差分别+2.4286/+1.5714个百分点；残差本次额外40轮，不能冒称新同预算消融优势。原同100轮差仍+1.8571pp。原配置继续训练改善了NLL，但准确率增益有限、泛化差仍大，不声称过拟合已解决。

[最终结果、真实耗时、科学身份和逐类汇总](reports/resisc10_restore_20261015/final/README.md)。173训练来源仍6ddd95fa90691714ef9d0e5ba96a78f38af41e05，current/original核一致；全部旧数据/PT/source/history/claims/失败/图保持。此一次授权已结束，不自动继续训练、测试、换类或电子重试。t18与09晨报ACTIVE保留，普通已完成状态静默。本次完成通知后不重复推送。

**下方启动/仍在训/待审核快照均为历史，最新事实以本节及最终报告为准。**

## 最新人工要求恢复原十类继续优化；真实曲线诊断（2026-10-15）

用户已明确要求查看本轮有/无残差训练曲线、分析收敛/过拟合，并换回原bridge/tennis_court等十类继续优化。[真实曲线、逐轮聚合CSV与诊断](reports/resisc10_reselect_20261015/curve_review/README.md)已生成2PNG/SVG并全部实际view_image QA；无新PDF。曲线显示两组后期验证NLL仍缓降，残差训练提升更快且train-val差更大；铁路站train本身也低，不能只归过拟合。

本次新授权仅一次恢复原十类残差E100完整状态接续E101-140（最多1800新增fit秒），保持MixUp.2、原Adam/EMA/scheduler和恢复LR.0005，不重复上次MixUp.4+额外LR减半设置。新173源为原169+四文件，不改旧源/数据/PT/claims；只GPU2一子，无early/无限retry/A100/test/电子支路/正式6/lab。原无残差是固定历史参考，新的残差额外40轮不是新的同预算消融；不靠削弱对照扩大差距。

目前已核实E100父last SHA856e377e5892e27540f8910c3ed1bc9b1121e870f1c4146e136ee289dc99d12b、history SHA86a20ca5da70638ea99dc2b52bb9d109c64afb70d7bd4aa2ff8eb45c293775f0、独占SCI4dbbe3c948fa93ad8895573dc405a1ad2c265217b3a99c09e9cba146357e0e01。新173 CPU状态准入已真实一次passed（E100/Adam35000/scheduler100/LR.0005/MixUp.2/编译，0models/forward/images/test），checksSHA b56e988d155c97f012c27101566590ad984eaa3668dc85096f9d1414337b1f11。已实际一次启动CPU父1816946/GPU2子1817647，新恢复态producer gate真实passed/all11finite非零/maxgradrel8.4891e-7，173current/original一致，实际训练来源6ddd95fa90691714ef9d0e5ba96a78f38af41e05。仅一张GPU2、GPU5项目0/A1000/test0，尚未完成140或新科学；不重CPUproof/start。详见[实际接续启动](reports/resisc10_restore_20261015/start/README.md)。详见[新一致接续合同](RESISC10_RESTORE_STEADY40_PROTOCOL_20261015.md)。目标80不能保证，按真实验证选模，不自动新test或更多候选。

**以下旧“不再继续优化”等范围被这次人工新授权覆盖，仅本173一次；旧所有实验/审核/测试/图仍不可重做。**

## 最新替换十类已完整100轮且科学通过：73.00%／74.14%（2026-10-15）

本轮已真实全完成，两组都100轮、maximum_epochs停止，无提前终止；CPU campaign/queue和两GPU producer实际退出，producer rc0，GPU2/5项目释放、A1000/test0。自己的独占保存科学首次passed，audit SHA aea29618e70b4f865b36f2498d35bd2cf3bcbc25f898cb47576ccf6faddea247 与completed实际一致，不重审。

事先固定firstminimum验证EMA balancedNLL选中残差E95/无残差E100：训练83.8929%/80.8750%，验证**73.0000%/74.1429%，差−1.1429个百分点**，NLL.8365082536/.8755160900，gap10.8929/6.7321pp。选E95不等于95轮停止，有残差实际继续完成了100。若只看同一E100保存历史，73.5714%/74.1429%，差−.5714pp，也没有残差accuracy优势。**换类没有达到预期，80未达，不自动继续搜索。**

新铁路站/梯田的残差选中验证召回45.71%/61.43%，仍偏低。[完整训练耗时、两种比较口径、全部逐类和真实科学收据](reports/resisc10_reselect_20261015/L4_pair_final/README.md)。两组本轮拟合约54–55分钟，未读取本轮test，无电子分类支路，没有新PDF/native/formal输出。

实际164训练来源仍f405eaf2214f9da0ada739fb5cb661747f1b8933，current/original真实一致；报告HEAD不同运行来源。原160/165/167/169/171及全部数据/PT/history/失败/claims/旧图markers保持。保留t18和09晨报ACTIVE，后续此已完成状态/旧失败不重复通知。

**以下“仍在训/待审核/首轮/新一轮准备”等内容保留为历史，当前以本节最终结果为准。旧77.86是旧名单test，不当本轮成绩。**

## 最新人工确认换入铁路站和梯田，新的十类同100轮配对（2026-10-15）

用户已明确选择保持10类，换出bridge/tennis_court，加入railway_station/terrace；新名单固定desert,harbor,intersection,island,lake,meadow,railway_station,terrace,thermal_power_station,wetland，原坐标indices[3,7,8,9,10,11,14,17,18,19]。这是新人工授权的一次重新训练，覆盖旧不得新训练/换类文字仅此范围，不自动测试或搜索其他名单。两组fresh seed17，有/无光学残差均100轮，不early；其余网络/划分/输入100/MixUp.2/350Adam每轮等保持，仍纯光无电子分类支路。每组10800拟合秒安全上限，最多合计6fitGPU小时；若意外未到100不得宣称完成或自动延长。只GPU2/5各一，A100禁止。

已复核原20输入100实际有残差114、无残差58，双方均time_budget而非验证early，独立保存SCI SHA adf635d3f4c5876f43a4d2bf3aad15f1e4e953899fd5a07eeff600505ffb4b73。残差E100铁路站51.43%、梯田41.43%、森林28.57%；不能据旧选中轮次成绩断言剩余类别一定弱，也不能保证换类增益。名单由用户根据旧验证分析确定；之前10类测试已见，此后不声称重叠测试仍未经开发观察。

新run resisc10_reselect_station_terrace_s17_pair100_mixup02_gpu25_20261015；164源闭包=原160+新runner/profile/protocol/audit四文件，旧160/165/167/169/171等不改。新的metadata子集/CPU黑图及native/GPU两rho数值门槛和实际producer门槛须各自真实通过，新的CPUcampaign只在两生产子rc0及100完成后独占一次保存科学。新prepare/CPU14forward及native/GPU2两rho数值检查已实际一次通过，164源运行来源f405eaf2214f9da0ada739fb5cb661747f1b8933，current/original一致。已一次启动CPU campaign父1500151/queue1500964/GPU2残差1501738/GPU5无残差1501804，两个producer own gate真实passed。首E1验证24.8571%/15.5714%仅普通早期值，active2/pending0/fail0/test0/A1000，不当最终差距。详见[新十类实际准入与启动](reports/resisc10_reselect_20261015/start/README.md)；不重任何新prepare/CPU/numeric/start或旧审核图。详见[本轮新合同](RESISC10_RESELECT_PROTOCOL_20261015.md)。

**下方已完成77.86与所有“不再训练/不改类别”等文字保留为旧阶段历史，由本节新人工范围覆盖；旧数据、PT、claims、失败及图markers继续保护。**

## 最新本轮已全完成：纯光残差一次测试77.8571%（2026-10-15）

按最新人工要求完成原十类有残差至100轮，随后残差单独再20轮缓解过拟合，两个GPU生产子actualrc0、CPU父实际退出，两个阶段自己的独占保存科学均一次passed。前100轮首次最低验证NLL仍选E100：有残差78.1429%、无残差76.2857%，**同100轮验证差+1.8571个百分点**。无残差不追加训练、E100权重未保留，不能声称等100配对测试增益。

残差E101-120的MixUp.4和LR减半没有改善最低验证NLL，最终按原选模规则仍冻结E100。随后171工具只执行一次700张测试，**77.8571%（545/700），macro同值，bNLL.7911064238**，独立保存预测审核passed。未达到80%，没有电子分类支路、不用test反调参或挑类别。[最终报告与逐类验证/测试分栏](reports/resisc10_equal100_20261015/final_once_test/README.md)。

169训练来源bbbc212f5af9a509e7bd7c25b3300dfaaa83e45d保持；complete100 SCI4dbbe3c948fa93ad8895573dc405a1ad2c265217b3a99c09e9cba146357e0e01，regularize20 SCIea6b6dd39c9a12ed180417d934dc805ef849bf7784b610e8d5250029b41a92f5。最终selectedE100 PT69eab238d82fb212e7f850203ec27d79bf51082b861c132e073d8f8efec1f1c8；test result2d9969957e3c5c86dd1a69a8a2335f9a2e4235de99147fb202a9933f2e177ca7、saved审核4d92bb9ccbf40f1762f16cd98c7d512f499d670ecde2d3e763cf4a112af4d17c及completed SHA实际一致。CPU父1290324及两生产子1291210/1307787、测试子1394225实际已退出；GPU项目0/A1000，test evaluations1。

本轮新增授权全部结束，不自动start/retry/audit/test/换类或继续冲80；原160/165/167及所有旧数据/PT/失败/claims/图markers保持。原8PDF已交付不重做。t18和每天09晨报保持ACTIVE，普通已完成状态与旧失败静默，只有真正新增需处理状态通知。

**下方旧快照保留，不代表现在仍在训、尚未test、可再次测试或继续调参；最新事实以本节及最终报告为准。**

## 最新统一100轮已完成，残差单独缓解过拟合进行中（2026-10-15）

用户新要求覆盖旧E92提前停止和接受后不接续范围，仅这一169新run及最终一次171工具测试。原十类名单、网络、划分和旧160/165/167全部保护，无残差不追加训练、不重试电子后备。

有残差E93-100已完整恢复E92的模型/EMA/Adam/scheduler，补足100轮且自己的独占科学首次通过：audit SHA 4dbbe3c948fa93ad8895573dc405a1ad2c265217b3a99c09e9cba146357e0e01，选E100 PT126abf82add6f5f0951e071acffb6c498a9fe288a5f629282542dc4f249a7d6b。训练87.6607%、验证78.1429%、NLL.778557385。无残差原保存历史前100轮首次最低NLL为E100，训练83.3393%、验证76.2857%。**统一100轮验证差+1.8571个百分点**，不能继续用旧E11377.1429来表示本口径。无残差E100权重不存在，不冒称可做等100成对test。

新169运行来源bbbc212f5af9a509e7bd7c25b3300dfaaa83e45d，CPU父1290324、complete100子1291210已actualrc0退出。CPU父随后一次启动GPU2 regularize20子1307787，最近只读E109验证78.2857%、NLL.790302175，仍普通训练、尚无阶段最终科学或测试结果。E101-120只残差MixUp.4、恢复LR减半，Adam/EMA/scheduler保持，每阶段新增拟合1800秒上限、不early、不无限重试。临时准确率不能代替firstminNLL最终选模。详细见[接续合同](RESISC10_CONTINUE100_PROTOCOL_20261015.md)。

[统一100轮逐类、旧20同轮证据与类别建议](reports/resisc10_equal100_20261015/class_analysis/README.md)现已更新。当前正差较明显的是网球场+8.57、交叉路口+5.71、港口+4.29，但网球场残差自身仅64.29%；岛屿+1.43、湿地+2.86，草地/火电站持平，桥梁/沙漠/湖泊各-1.43。早先E92/E113建议保留为历史，不当当前结果。未自动更换类别或新增四/六分类实验。

用户澄清最终主要关注测试集；所有训练/选模/类别建议仍只用验证。171一次残差测试工具闭包已发布abc6b96be52cd1ddf487f06fbc8feec640a87625，在169之外；CPU编译/错run和当时父未完成的拒绝准入已真实一次passed，models/forward/test数组0，不重做。两阶段实际全退出、各自独占science正确completed SHA、来源和firstminPT冻结后才创建一次final_once_test；目前test/out/claim/marker不存在，未读test。无残差E100 PT缺失，不能以E113冒充测试对照；[一次测试合同](RESISC10_FINAL_TEST_PROTOCOL_20261015.md)。

**以下实验总览和运行快照保留作历史，旧“全部拟合结束/不接续/未授权test/仍未交付”等文字由本节新人工范围覆盖。原8PDF已于08:50交付，不重作者、marker、render或传输。**

## 当前实验总览与用户选择（2026-10-15）

用户已表示接受**十分类、MixUp.2优化后的纯光有残差候选：验证准确率78.1429%**。它没有可训练电子分类支路；对应同一优化方案的纯光无残差对照为77.1429%，选中结果差为**+1.0000个百分点**。当前采用这一纯光候选，490参数电子后备不采用。80%目标没有实际达到；此处接受的是仿真开发验证结果，不等于测试集、选中训练权重原生回放或实拍验收。全部获批拟合和审核尝试已结束，不自动重训、重审或追加候选。08:50完整报告与8份已实际QA的PDF已经交付，09:00晨报也已发送。

### 三个容易混淆的名称

- **纯光有残差（rho.3）**：调制光场与未调制参考光场按70/30场振幅相干相加。残差发生在光路中，可训练参数仍只有光学相位，没有电子Linear/CNN分类支路；最终由固定检测窗口能量读出。固定OEO/归一化属于既有合同。
- **纯光无残差（rho0）**：保留相同网络和训练设置，去掉上述参考光场。它也没有电子分类支路。
- **小电后备（hybrid）**：冻结已审核纯光有残差权重，另在CPU上训练一个490参数的图像粗网格线性分类器，最终概率=0.8光学概率+0.2电子概率。这是混合模型，系数是输出概率融合权重，不是光能量或算力占比，也不是rho.3。

### 10月14日至15日实际做过的实验

以下全部为验证准确率；差距统一定义为“有光学残差减无光学残差”，单位为百分点。每组都按首次最低验证EMA balanced NLL选权重，未改选最高准确率轮。

| 阶段 | 做了什么 | 有光学残差 | 无光学残差 | 差距 | 电子分类支路 | 当前结论 |
|---|---|---:|---:|---:|---|---|
| RESISC20初训 | 固定20类，原输入，30轮配对 | 40.8863% | 37.9557% | +2.9307 | 无 | 未达到开发精度门槛 |
| 同一20类接续 | 从原30轮接续到总120轮，不是另一个独立种子 | 46.6047% | 46.8906% | -0.2859 | 无 | 未达到开发精度门槛 |
| RESISC20输入对照 | 保持网络，将输入通道重采样50改100 | 49.6069% | 41.1723% | +8.4346 | 无 | 轮数114/58不同，不能把全部差距归因残差；仍低精度 |
| RESISC15 | 按人工要求，从原20类固定随机选15类 | 51.4776% | 47.3785% | +4.0991 | 无 | 轮数120/78不同，仍低精度 |
| RESISC10原始配对 | 按人工要求，固定原20类验证表现前10类 | 75.8571% | 76.1429% | -0.2857 | 无 | 科学审核及四图实际QA通过，保留作对照 |
| RESISC10纯光优化 | 同十类、网络和划分，只将训练MixUp强度.6改.2 | **78.1429%** | **77.1429%** | **+1.0000** | **无** | **科学审核及四图实际QA通过；用户接受的当前候选** |

原十类有残差75.8571%到优化后78.1429%的**+2.2857个百分点**是训练方案优化前后变化，不是有/无残差消融增益。不能将新有残差与旧无残差交叉配对来扩大差距。十类名单已经参考此前验证结果选择，因此属于validation-informed开发任务；不能把20/15/10类的准确率变化当作同一个任务上的受控提升。此次没有测试集评估。

优化配对实际完成92/120轮，各自选中E92/E113。补充同一E92观察为78.1429%/75.5714%，差**+2.5714个百分点**；这只是单种子同轮观察，不能替换最终各自按NLL选中的78.1429%/77.1429%及其+1.0000差距。

更早的Caltech原因排查包括：9组输入/路由/全局光场等消融；3组MixUp强度对照（验证62.07%至64.37%）；3组相位网格容量对照（原网格62.84%、128网格58.24%、64网格48.66%）。均未解决原任务门槛，缩网格未被采用为当前网络。还完成CPU输入细节诊断、微批提速与数值一致性检查；这些属于原因/工具检查，不是额外高分模型。A100上的20类旧输入控制按人工要求在E26停止，未恢复。当前十类网络仍为908552个光学相位参数、四层主干及额外光学router。

### 小电后备为什么不采用

只做了一次**有光学残差父模型+490参数电子支路**的CPU拟合，固定光alpha=.8，生产记录79.00%。其首次独立审核在NumPy与Torch输入池化特征一致性检查失败，因此79.00%尚未验收，表面比纯光78.1429%高.8571个百分点也不是有效已核验增益。没有训练“无光学残差+同一电子支路”对照，所以不能声称混合模型的有/无残差差距；没有混合结果图。失败、权重和全部原始记录保留，不放宽门槛或自动重试。

### 证据入口

[20类30轮](reports/resisc20_20261014/L4_pair_microbatch_final/README.md)、[20类120轮](reports/resisc20_20261014/L4_pair_continue120_final/README.md)、[20类输入100](reports/resisc20_20261014/L4_input100_pair_final/README.md)、[15类配对](reports/resisc15_20261014/L4_pair_final/README.md)、[10类原始配对](reports/resisc10_20261015/L4_pair_final/README.md)、[10类纯光优化](reports/resisc10_optimize_20261015/L4_pair_final/README.md)、[电子后备审核失败](reports/resisc10_small_electrical_20261015/audit_failure/README.md)、[9组Caltech消融](reports/caltech_diagnose_20261014/all9/README.md)、[3组正则对照](reports/caltech_repair234_20261014/L4_regularization_reviewed/README.md)、[3组容量对照](reports/caltech_repair234_20261014/capacity_final/README.md)。

**以下保留历史运行快照；其中“仍在训/尚未审核/尚未交付/冲80”等旧文字不代表当前状态，以本总览为准。**

## 最新490CPUhead拟合完成 独立科学首次失败 混合未验收 2026-10-15

一次获批CPUhead子947019已actualexit/fit_result完整，167拟合来源0d15f4137，35轮选E34、生产记录train86.7321/val79.00%、alpha光.8、电490。真实第一次独占NumPy审核在独立RGB池化对Torch特征误差<2e-6断言失败，audit/completed不存在，全部claim/log/failure保留。混合数字仅未经审核生产记录，不是80或有效混合通过。没有hybrid图/marker，不改167、不放宽阈值或自动重试/新候选。[真实审核失败与状态](reports/resisc10_small_electrical_20261015/audit_failure/README.md)。

165纯光真实science+四图QA完成，残差有效验证78.1429%、无残差77.1429%；原160有效75.8571/76.1429%保留。当前全部获批拟合已结束，80目标未达，GPU项目已释放、A100/test/native/formal未授权。新失败提前通知一次，普通巡检不重复，08:50统一报告与09晨报保留ACTIVE。覆盖下文所有head仍在训、未执行审核的旧状态，不重审任何已完成科学/图。

## 最新165优化已全完成科学与四图QA 条件490CPUhead已启动 2026-10-15

165/6c230残差92轮选E92 train86.8036/val78.1429%；无残差120轮选E113 train84.8571/val77.1429%。原规则early/maximum_epochs先到停止，父及两子actualrc0、fail0/test0/GPU释放。独占新science SHA ddf43abb366dc0f26a2dec9b0102c48dbc2cb81ced3df0d1a5002e098ab9d346及所有严格逐类准入passed，四PDF真作者/marker/SHA传输/Poppler/全部四页actualQA完成，不重做。仍未80、未test/native/formal；[完整165结果](reports/resisc10_optimize_20261015/L4_pair_final/README.md)。

165所选残差<80后已立即一次启动获批167 CPUhead子947019，光alpha固定.8、电.2概率融合、冻结E92光EMA，仅490电参数，明确hybrid非纯光。只真实对应train/val原RGB与保存光概率，不构造/forward光模型或新增GPU/A100/test。自己的actualexit+fit_result/status+独占NumPy科学和条件三PDFactualQA完成之前不宣称80或正式通过；[真实head启动与界限](reports/resisc10_small_electrical_20261015/start/README.md)。

原160与165的科学/图已完成不得重审或重图；当前需推进的是167 CPUhead自己的有界完成与审核。08:50统一成绩/未完成状态，03:30取消，09晨报保留。覆盖下文所有165在训、science/图待执行以及head尚未启动旧文字。

条件490CPU小电后备的独立保存图工具已准备，真实Linux拒绝准入passed、models/forward/images/test/GPU/图/marker0，167及当前165源/PID保持；head尚未拟合，未来专属三PDF需要自己的真实hybrid科学和actualQA，不能套纯光出口。[条件后备三图准入与边界](reports/resisc10_small_electrical_20261015/plot_preparation/README.md)。

## 最新原十类已完成且准入通过 一次MixUp.2优化已真实启动 2026-10-15

原160/e380两组120/119轮、firstminNLL均选E113：残差train83.50/val75.8571%，无残差train81.4286/val76.1429%。实际全部rc0退出，独占genuine science SHA c7c7af78dc0fc84d79dd0552dda6fc42c8f82c1e53e1bdbba9c243745b4ab5af、逐类/路由/预算门槛均passed，test0；4PDF真实作者/SHA传输/Poppler120dpi和全部四页实际view_image QA完成。80目标未达，不算原20验收。[完整保存结果与实际QA](reports/resisc10_20261015/L4_pair_final/README.md)。

按最新人工授权立即新fresh纯光两rho优化，仅MixUp.6->.2作为拟合假设，其他10类/网络/划分/input100/训练设置保持，每组原120或7200秒、只GPU2/5各一。CPU/native和actualGPU2数值准入已真实一次passed，03:12:31启动CPU父767046/GPU2子768304/GPU5子768385；165冻结运行来源6c2309ba048f2fa12f13390ca6fbc62954dc2c74与current/original一致。A100禁止，test0，无第三候选/自动retry/延长旧run。[真实启动、证据假设和边界](reports/resisc10_optimize_20261015/start/README.md)。

原160审核/图/marker已经完成不可重跑。新165自己全完成屏障后一次独占科学/四图实际QA；若所选残差仍<80才一次已授权490参数CPUhead概率融合光alpha.8。08:50统一汇报，03:30取消，09晨报保留；覆盖下文所有原十类在训、未审核/未图和03:30旧文字。当前新165尚未完成科学/最终成绩，167head未拟合。

## 最新人工目标80与九点前汇报 小电残差条件许可 2026-10-15

165纯光优化专属保存图工具已独立准备，Linux真实CPU导入/编译及错run、缺真正优化科学审查时创建out前拒绝passed；模型/forward/images/test/GPU/图/PDFmarker0，当前160源及GPU PID集合保持，新优化尚未实际fit。[真实165图准入与未来四PDF](reports/resisc10_optimize_20261015/plot_preparation/README.md)。

用户取消03:30汇报，要求今天09点前给结果，并要求有残差组冲80%；若纯光不足，允许极少量侧边电残差，光alpha必须>.6。单次汇报已经改为08:50，同聊天给实际成绩/完成或未完成状态；09晨报保留。80为目标不能保证，当前都是验证数值，test仍未授权。
当前160/e380纯光两组继续原预算不改源码；实际全部退出和saved science后，若所选有残差验证不足80，执行一次新fresh配对MixUp.6->.2纯光优化，其他因素/10类/网络/划分/输入保持，每组max120或7200拟合秒、两组最多4新增fitGPU小时，仅2/5各一。若已到80，优先真实性/逐类/native验收，不盲搜。
若这一纯光优化仍不足80，后备固定光alpha=.8、电=.2概率混合，区别于rho.3的相干70/30场振幅。保留科学审核所选光EMA冻结，电支路固定RGB100的4x4均值池化48features+Linear48x10/bias仅490可训练参数，无CNN/hidden/pretrain/大型电子替代。仅CPU拟合已审查train/val保存光概率和对应固定输入特征，max120或7200CPU秒，firstminval EMA balancedNLL选；明确hybrid非纯光，不读取test/新GPU第三子或使用A100。
新纯光优化五文件165source与条件入口、490参数小电支路/独立numpy保存审查已准备；只完成工具数学CPU准入，尚未执行实际新profile门槛/拟合/科学审查，不称80成功。
夜间新工具CPU准入已真实一次passed（source167，检查SHA51eb72fc4b0b27db0048dab7710cb2afc4f4f68232ed207a5d81b20dea0c81d3）：Linux导入/条件pure-parent未完成提前拒绝、固定alpha.8和拒绝.6/490参数/独立NumPy池化与概率/两个电参数梯度finite非零均通过。仅12synthetic输入、2电forward、实际dataset/test/GPU/opticalmodel/opticalforward0，当前160及GPU PID集合保持；不是已训练hybrid、80成绩或native。新smoke/claim及私有helper已经一次完成不得重跑，raw JSON仅本地服务器。
需要真实新claims/config/source/parentSHA/490params/alpha>.6/IDs/CM/NLL/更新步数核，不套旧11phase训练门槛或当前纯光native出口。原全部160/159/151和旧数据/PT/run/失败/claims/markers保留，源码只Git，所有新步骤独占out不重跑、不无限扩搜索。详情见[夜间优化与电残差合同](RESISC10_NIGHT_OPTIMIZATION_20261015.md)。

## 最新人工要求当前十类完成后继续针对性优化 2026-10-15

用户明确问首批是否等于全部epoch完成，并要求完成训练后继续优化。首批仅指最初E5/E4；当前十类还在训练，不能误称全阶段结束。先按原160/e380冻结合同完成现有两组，实际全退出后一次saved science和曲线/薄弱类分析，再执行一轮有界明确新profile的针对性优化；不提前重叠启动或改现用160文件。
后续优化保持同十类、固定网络、原划分/input100/仅GPU2和5，沿用每组max120或7200拟合秒、一次两rho配对最多4新增fitGPU小时的界限，不无限搜索或延长旧run；新准入/来源/新run须真实核验。若当前验证目标已满足，优先验收与正式方案准备。该人工新要求覆盖下方当前十类协议里完成后无自动优化的旧范围，不授权A100、test、正式六组或现场校准。
今天北京时间03:30依然汇报当前阶段明确成绩，不能因后续优化未完成而推迟；普通健康静默，09晨报保留。

## 最新人工要求原二十类表现前十类 已真实启动 2026-10-15

用户时间紧迫，明确要求从之前20类选表现最好前10类继续。按原20 input100 rho.3 selectedE111验证逐类recall排序，固定草地/沙漠/岛屿/湖泊/桥梁/港口/交叉路口/网球场/湿地/火电站，属于validation-informed开发子任务，不再随机选类或读test。
原cache/划分保持，train5600/val700/test700 metadataonly；350Adam每轮尾16，原908552phase/expert224/global478及固定网络光图/input100不变，只10个原坐标读出窗口与10路归一化。160新训练闭包冻结e380a206b5b032d5d6bf1a855846a73de556564f，原159/151和全部旧依赖保留。
最终CPU14forward/native回放/黑图概率sum1及actualGPU2数值准入、两producer own gate真实passed。01:05:37 CPU父585499/GPU2残差子586248/GPU5无残差子586314实际启动，仅2/5各一，A100禁止；PID为历史以实时proc核。首E5/E4验证53.29%/46.29%，轮次不同不作rho收益或最终75结论。
每组max120或7200拟合秒、两组最多4fitGPU小时，原early先到停止，无第三候选/retry/延长/test。今天北京时间03:30统一汇报明确成绩/成功或未达标或未完成状态，普通健康静默，09晨报保留。
[前十名单依据与真实准入/启动/预算](reports/resisc10_20261015/start/README.md)；[独立10类四图工具准入](reports/resisc10_20261015/plot_preparation/README.md)。科学审查尚未做，自己完成屏障成立后一次新专属audit，旧15/20审核/图不重复。

## 最新十五类配对已结束 验证准入未通过 2026-10-14

残差组120轮选E120 train61.7052%/val51.4776%/macro51.4465%；无残差78轮选E78 train53.3341%/val47.3785%/macro47.3485%。分别轮数上限和时间预算先到，父146495/两子147208、147277实际rc0退出，本项目GPU全部释放，A100仍禁止。
真正新保存科学CPU审核passed（SHAeb6d1bf0f77ed131b270aaf193b6bd5557fc6ee10d9ccd8f591c06d4ad12016e），独占claim/completedreceipt齐全；159原4574来源保持。两组仍failval65/macro60/each20，test0，无eligible，不启动native/depth2或3/正式六组/test或扩预算。
不同卡速度使实际120/78轮不同，最终差不是纯残差因果；共同E78 val48.62%/47.38%仅single-seed描述。十五类是明确改动后的任务，不能冒称原20类目标成功。
四PDF/PNG/SVG/聚合CSV真实生成，Poppler120dpi逐页渲染及全部四页view_image视检passed，所有旧data/PT/source/run/history/failure/claims/marker保留。按人工约定22:30统一汇报，持续monitor/09晨报仍ACTIVE。
[全部实际结果、门槛、预算和图表](reports/resisc15_20261014/L4_pair_final/README.md)。

## 十五类人工要求与历史真实启动 2026-10-14

用户明确要求选15类继续尝试，已固定随机seed20261014选15类、保留原划分。train8398/val1049、macro16/micro8/525updates每轮尾14，原908552相位和expert224/global478不变，只15个原坐标固定检测窗口。
159源码闭包冻结4574b3e84，CPU/native及actualGPU2数值检查与GPU2/5两生产门槛均passed、概率/capture0、gradrelative<6.65e-7、all11finite非零。父146495、两子147208/147277实际健康，A100项目0，仅两卡各1；PID只历史以实际proc核。
各fresh max120或7200秒沿用有界规则，两组总最多4新增拟合GPU小时，无额外候选/test或扩预算。15类任务成绩不得直接验收原20类目标；原全部数据/PT/source/run/收据保留。
独立15类保存图工具已准备，Linux缺真正科学审核时在创建out前拒绝，未产生图/PDF。[工具准入和未来四图](reports/resisc15_20261014/plot_preparation/README.md)。
[15类名单和实际启动](reports/resisc15_20261014/start/README.md)；[四层精度变化已知证据与未分离原因](reports/resisc15_20261014/ACCURACY_CHANGE_EVIDENCE_20261014.md)。首轮不作最终精度结论，75目标未验收，监控与09晨报继续。

## 二十类输入100配对已结束 精度准入未通过 2026-10-14

获批两组均正常退出且项目GPU全部释放，A100继续禁止。rho.3实际114轮/选E111 train57.18% val49.61% macro49.58%；rho0实际58轮/选E58 train46.69% val41.17% macro41.15%，均按原epoch开始时间检查触发7200秒预算停止。
专属剩余pair保存科学CPU审核实际passed，B/C完整记录及A26停止前缀均核验；test0、没有eligible，不启动native/2或3/正式六组/test或自动扩预算。
两卡速度不同使完整轮数不同，最终accuracy差不能直接归因于残差；共同E58 val43.46%/41.17%仅单seed描述，A/B输入因素比较仅共同26轮。
五PDF及PNG/SVG/聚合CSV由保存结果生成，Poppler真实渲染和全部五页view_image视检passed；原151/d515训练源及全部数据/PT/历史/停止与审核收据保留。
[完整结果与五图](reports/resisc20_20261014/L4_input100_pair_final/README.md)。无新增GPU预算，75目标未验收，监控及每天09点晨报继续。

## 历史资源调整 暂时仅GPU2和5 A100已主动停止并释放 2026-10-14

用户要求“暂时只能使用两张GPU，A100的先停掉，我有另外用处”，覆盖下方所有三卡/A100可用的历史范围。
只对核实command/run/CUDAUUID的GPU6 A50rho.3子3940641发一次SIGTERM并实际退出，nvidia确认本任务A100进程已释放，best/last/history/gradient全部保留；完整26轮，不自动恢复。
CPU父无pending，GPU2 B100rho.3/GPU5 C100rho0继续各原120/7200预算，不改151/d515冻结源或增加候选。
18:06:59保存CPU只读：B E30train45.72/val41.39%；C E15train36.70/val33.81%；A已停E26val39.89%。轮次不同不作残差收益比较，test0。
producer的input_study_failed/rc-15记录指人工停止A，不是另外两组数值失败；原三完整CPU审查/5PDF入口不适用于当前partial，需独立剩余pair审查，不放宽旧门槛。
[实际释放、存档、两卡进度与后续审核边界](reports/resisc20_20261014/GPU2ONLY_STOP_A100_20261014.md)。
剩余input100配对的独立保存CPU审核已准备，Linux提前拒绝及停止A26独立order/增强/MixUp/LR检查passed，但完整科学审核未执行。[真实准入与新屏障](reports/resisc20_20261014/remaining_pair_audit_preparation/README.md)。
相应“两完整组＋停止前缀”独立保存图工具已准备，Linux缺新科学审核时创建out前拒绝passed，未生成图/PDF。[未来5PDF口径与真实准入](reports/resisc20_20261014/remaining_pair_plot_preparation/README.md)。

## 人工批准输入细节三组对照，完整网络已实际启动（历史启动，2026-10-14）

用户在普通话解释后明确要求“你赶紧试一试吧，我的时间很紧迫”，已批准具体三组范围，不再等待同一预算批准。
三组fresh L4：原50通道输入+rho.3控制、100通道输入+rho.3、100通道输入+rho0；网络仍908552相位、expert/router224/global478。
CPU原输入有/无增强复现exact，新输入独立构建及9项native磁盘回放概率误差0；实际GPU6三组20类+尾14门槛passed，梯度相对误差<=6.35e-7，初始phase SHA相同。
151运行来源d5158a53e已冻结，CPU父3940020及GPU6/2/5三子3940641/3940707/3940779实际启动（PID仅历史，后续以实际proc/log核）。
每组最多120轮或7200拟合秒，原11198train/1399val、micro8/macro16/700updates保持；总最多6拟合GPU小时，不加第四候选/自动扩预算/test。
[新三组实际检查、启动及预算](reports/resisc20_20261014/input_detail_start/README.md)。
三输入专属保存图工具已独立准备，Linux缺完整科学审查时在创建输出前真实拒绝；CPU模型/推理/test/GPU/PDFmarker/图表均0，原151源保持。[准入收据与未来5PDF条件](reports/resisc20_20261014/input_plot_preparation/README.md)。
旧144/141/136/135/126/130与全部旧输出保持，原120结论仍不合格，不能当新输入结果；正式六组与75目标尚未验收。

## 新CPU输入诊断已完成，聚合报告发布获批准（历史决策记录，2026-10-14）

不构造/推理模型、不读test、不占GPU的一次有界CPU诊断完成，来源61b21e392；20类各16train样本共320。
原编码150->100->50通道细节重建独立误差0；50重建150的相邻像素差平方能量保留平均8.30%，中心像素方差相对误差24.75%，只描述重采样，不证明分类原因。
[聚合CPU报告](reports/resisc20_20261014/saved_input_diagnosis/README.md)；[已获人工批准的三组原网络输入50/100方案](reports/resisc20_20261014/INPUT_DETAIL_PROPOSAL_20261014.md)。
原专家224/global478/908552phase不变；新50rho.3控制、100rho.3、100rho0各最多120轮或7200拟合秒，最新批准与实际启动以上文为准。
自动审批审核拒绝此新CPU诊断明细JSON以及聚合报告的GitHub发布，原因是指定实验payload/目的地授权不明确；原始收据本地服务器保存，不变相导出。
用户明确答复“允许发布聚合报告和方案”，聚合Markdown可发布至MillionXie/2026OpticsMoE main，逐样本ID/完整JSON仍本地服务器保留。新训练问题最初答复为不理解，解释后另获明确训练批准；两项授权分别记录。
原120有界失败完整报告已发布；所有源144冻结、旧data/PT/test/run/markers保护。当前训练以最上方新151运行记录为准。

## RESISC120轮全部完成：CPU审核通过，精度准入失败（2026-10-14）

两rho各120完整结束，父/全部GPU子实际退出，项目2/5/6释放；完整保存科学CPU与预算接续审核passed，test0。
firstminvalEMA NLL选E120：rho0train53.36/val46.89/macro46.86/gap6.47pp；rho.3train56.96/val46.60/macro46.58/gap10.35pp。
比原30val37.96/40.89提高，但两组仍failval65/macro60/各类20；机场/篮球场等train与val都弱，不认为75/充分收敛或唯一原因已解决。
144当前/originalbbdaf源保持，原所有data/PT/history/proof/run保留。4PDF/PNGSVGCSV已真实Poppler渲染逐view_image视检通过。
[全部实际结果、曲线、逐类CM与审核](reports/resisc20_20261014/L4_pair_continue120_final/README.md)。
没有eligible，不启动正式6/条件2或3/native/test，不自动扩预算或换epoch选模；下一步仅有界CPU机制诊断与具体修正方案准备。

## 人工批准RESISC继续至120轮，完整网络已实际接续（2026-10-14）

原30轮已审核完成后，用户明确批准两组继续至总计最多120轮，每组新增最多2小时，先预算/既有early-stop到即停。
新144文件入口来源bbdaf0a85，CPU父3804063、GPU6 rho.3子3804643/GPU2 rho0子3804652实际启动；恢复原E30 raw/EMA/Adam/scheduler，无重置/重播。
两生产20类+尾14数值检查passed，首完整E31实测rho0 val36.74%/56.99秒、rho.3 val40.67%/61.23秒；尚未选最终权重或达精度门槛。
全11198train/1399val、原908552网络、micro8/macro16/700steps及数据/光图/门槛不变；旧141/136及完成报告全保留，test0。
[真实启动、恢复、预算与首轮收据](reports/resisc20_20261014/continue120_start/README.md)。
后续需实际全部退出后新有界接续CPU审核；无合格不强行2/3/native/正式六组，不无限追加。

## RESISC完整30轮已结束：提速成功，精度准入未通过（2026-10-14）

两rho全30轮，原E1-21及micro8接续E22-30保持；父/两子实际退出，项目2/5/6显存释放，完整CPU+执行接续审核passed，test0。
同按firstminvalEMA balancedNLL选E29：rho0train41.45/val37.96/macro37.93/gap3.50pp；rho.3train45.24/val40.89/macro40.86/gap4.35pp。
两组failval65/macro60/eachclass20，低训练拟合而非此前Caltech高train大gap；小gap不合格，不认为75/泛化已解决。
每轮全11198train/1399val、macro16/700steps/tail14保持，micro1改8后普通完整轮57-61秒，约快7倍；最终网格未缩小。
141接续运行源75be/原136513不变，所有PT/history/失败收据保留，4PDF真Poppler渲染逐view_image视检通过。
[完整结果、速度、恢复、逐类CM和审核](reports/resisc20_20261014/L4_pair_microbatch_final/README.md)。
原30轮预算已完，不自动更多训练/换类/扩预算/提前test；正式6和独立2/3/native仍未准入。

## 用户要求提速：RESISC原完整网络micro8已实际接续（2026-10-14）

完整11198train/1399val，macro16/每轮700Adam更新/tail14不变；实际micro1改8，模型/数据/总30轮与14400秒预算保持。
直接批量数值门槛失败已保留；逐图保持原固定窗口归约、批量光传播后2/4/8全通过，micro8概率误差0、梯度相对<=8.27e-7、显存约1GiB。
旧两组完整E21状态保留并主动停止，raw/EMA/Adam/scheduler及全部已完成历史接续，不重训/不重播E1-21，丢弃可能未完成一轮已披露。
新run resisc20_s17_mainL4_rho0rho03_fullres_microbatch_continue30_gpu256_20261014，父3740234，GPU6/2子3740747/3740819，141源冻结75be74b96。
两生产恢复门槛passed，首完整E22普通58.66/61.77秒，约比原7分钟快7倍；原136/513所有文件保持。
[真实提速、恢复、预算与后续CPU审核收据](reports/resisc20_20261014/speed_continuation/README.md)。

## 相位网格诊断已完成，缩网格未解决验证门槛（2026-10-14）

full/128/64各150轮，父及三子实际退出，完整shape-aware CPU审核passed，test0。
按各自firstmin valEMA NLL选E147/E146/E150，train86.92/67.98/56.98%、val62.84/58.24/48.66%，gap24.08/9.74/8.32pp。
三组无eligible：粗网格缩小差距，同时训练拟合与验证精度都下降，不能据此当作修复或唯一原因。
最终网络仍保留原分辨率；新RESISC完整网格配对继续原30轮/14400秒预算，不扩网格搜索/提前test。
135运行来源仍cb7冻结；五PDF已真实Poppler渲染并逐页view_image视检通过。
[真实结果、逐类门槛、曲线和完整验证CM](reports/caltech_repair234_20261014/capacity_final/README.md)。

## 容量／RESISC保存结果图表工具已准备（2026-10-14）

独立绘图入口支持三网格诊断及新数据两rho，不修改135/136冻结训练来源。
Linux实际CPU导入和两项缺完整审核时提前拒绝已通过；尚未生成图或PDF、无推理/test/GPU动作。
每项实验实际全部退出并完成相应CPU审核后，才绘制真实曲线、逐类support和完整验证CM。
[工具准入检查和未来绘图条件](reports/saved_validation_plot_preparation_20261014/README.md)。

## 人工选择RESISC45固定20类，允许共享三卡最多六个GPU子（2026-10-14）

用户明确选择RESISC45固定20类，并允许总最多6个项目GPU子进程、物理2/5/6每卡最多2个。
原网格对照只用于结论；新数据实验保留原分辨率/网络/光路，先L4rho0/.3同预算配对。
原始45类各700张，训练前seed20261014固定20个名字，不按成绩筛类；14000图按精确像素连通组80/10/其余划分。
实际14000图片全部保留、13994精确像素组，train11198/val1399/test1403，无每类100上限，SHA已冻结。
新原网格四层两rho10:39:43实际启动：CPU父3579436，GPU6rho.3子3579968/GPU2rho0子3580043；
CPU/实际GPU门槛及两生产检查passed，136运行来源冻结513f0e884，项目共五GPU子、每卡不超过二。
新初轮各最多30轮/14400秒，首完整epoch指标/耗时待实测，test不选模，未到正式六组。
当前三容量训练的135源和预算仍完全冻结，不修改其模型或配置；新数据源码/输出独立。
[新数据、固定原网络及共享并发协议](RESISC20_PROTOCOL_20261014.md)。
[实际数据、门槛与启动收据](reports/resisc20_20261014/README.md)。

## 人工已批准三组相位容量原因诊断，保留最终网络大小（2026-10-14）

用户在解释后明确允许尽快尝试full/128/64三组，强调最终网络大小已定、不偏好缩小；
同时要求核查训练样本为何少，若当前数据不合适则换适合的数据集。此授权覆盖下文“仍待审批”。
独立入口已完成并实际09:29启动，CPU父3509387，GPU6full3509913/GPU2main1283509980/GPU5main643510049；
三生产检查passed，目前已保存E9实际CPU检查。135运行源冻结cb7bfb3e2，原126/130完全不变。
只约束可学习相位网格，专家224/global478实际孔径、4层主干及额外router不变。
三新组共用mainraw0初始物理相位，routerseed17；新原分辨率对照必需，旧随机初相位不能作对应控制。
同物理展开尺寸的平滑项、原数据/图/量化/门槛及150轮/14400秒预算保持，不自动采用缩小网络。
实际CPU初始光场/native及GPU自由参数梯度门槛已通过，三组按150轮/14400秒预算运行；禁止第4候选/重试/提前test。
正式六组和旧conditional深度队列仍未获本入口准入；原126/130及所有旧run保持。
[具体执行与验收协议](CALTECH_CAPACITY_PROTOCOL_20261014.md)。
[真实门槛、启动和早期CPU证据](reports/caltech_repair234_20261014/capacity_start/README.md)。
[原始图片数量核验与保持完整网络的大数据候选](reports/caltech_repair234_20261014/DATA_SIZE_AND_NEXT_DATASET_20261014.md)。

## 四层复合修正／正则验证已结束，未获正式准入（2026-10-14）

三候选各150轮，父及三GPU子实际退出，GPU2/5/6项目显存释放（包括A100）。完整CPU保存审核通过，test0。
按最低验证EMA balancedNLL选E137/E146/E149：MixUp.2/.4/.6 train92.78/90.45/91.65%，val63.98/62.07/64.37%，macro62.65/60.15/63.62%，gap28.79/28.38/27.29pp。
三组均未通过val65%及gap<=15pp，前两组还有逐类最低召回失败；无合格profile，未启动条件2/3或正式六组、未test。
本轮恢复训练拟合能力但没有解决过拟合／泛化问题，不用其他epoch精度替换NLL选模、不额外删类或自动追加候选。
[完整失败记录、CPU收据和已实际PDF渲染视检的图表](reports/caltech_repair234_20261014/L4_regularization_reviewed/README.md)。
保存相位CPU诊断已完成（无模型构造/前向/data/test/GPU）：908552参数，平滑项约.0026；容量和频谱是描述性证据，不代表唯一原因。
[诊断及待人工批准的三组相位容量对照方案](reports/caltech_repair234_20261014/saved_phase_diagnosis.md)。新方案尚未实现或启动，不追加当前已失败预算。
保存预测额外检查：三组train混淆矩阵的20类对角线均为行内唯一最大计数，验证总体偏软；普通温度锐化不改变argmax，不能补硬分类精度。
[分类窗口／置信度证据及口径](reports/caltech_repair234_20261014/saved_window_confidence.md)。新预算审批问题仍待人工回复，GPU保持释放。
此前启动/准备说明均为历史，运行126源仍46011b823保持不变。

## 复合修正与独立四层正则验证（2026-10-14，已启动）

全九消融已完成并审核；输入RGB、连续四槽路由、保留478全局场进入独立复合图，组合收益尚待验证。
只新20类train1246/val261，四层rho.3最多MixUp.2/.4/.6三候选，150轮/14400秒上限，三卡仅2/5/6。
每类召回、验证NLL、路由变化、过拟合差距和实际8bitBMP共同设门槛；没有合格候选不强行正式训练。
实际04:05 CPU父3210879启动，GPU6/2/5三子3211094/3211169/3211235，三生产门槛passed；首轮约55-58秒，E2全部CPU保存审核通过。126运行源冻结46011b823，旧121源未改。正式2/3/4六组需修正/正则和2/3层独立验证通过后才启动。
[实际启动、数值/BMP及CPU证明](reports/caltech_repair234_20261014/README.md)。
[预先固定的图、数据、预算及验收条件](CALTECH_REPAIR234_PROTOCOL_20261014.md)。
匹配478全局输入/连续四槽的独立出口已完成，六种静态包及18项动态CPU原生BMP回放概率误差0，旧126运行源未改；选中训练权重及实拍校准仍需验收。
[出口合同和真实检查](reports/caltech_repair234_20261014/matched_export.md)。
2/3层独立验证入口已准备，尚未启动：须四层合格候选完整CPU审核、选中训练权重原生BMP通过且父子实际退出；每深度仅一个rho.3配置，不新增正则搜索。
[条件、预算和预先逐类门槛](CALTECH_REPAIR23_PROTOCOL_20261014.md)。
Linux真实CPU检查提前queue/fit/native均在创建输出前被门槛拒绝，旧GPU进程未变、forward/test0；这不等于已完成深度训练。
[入口准备与实际拒绝收据](reports/caltech_repair234_20261014/depth_validation_preparation.md)。
最终once协议原语已准备，Linux12项合成检查通过，真实锁/test0；完整正式训练器和科学CPUfinalizer尚待条件满足后冻结，不能把该工具检查当正式结果。
[协议边界和CPU收据](reports/caltech_repair234_20261014/once_protocol_preparation.md)。
2/3层获选权重原生BMP检查入口已独立准备；Linux真实CPU提前执行拒绝、六项fresh原生回放概率误差0，运行GPU身份未变。须两深度完整CPU审核和逐类门槛通过后才验实际选中EMA；当前未启动深度训练。
[检查合同与执行边界](reports/caltech_repair234_20261014/depth_native_preparation.md)。
三候选保存结果绘图已准备，Linux入口缺完整CPU审核时拒绝创建图/PDF；完整审核后才生成并实际PDF渲染视检。
[口径与实际入口检查](reports/caltech_repair234_20261014/plot_preparation.md)。

## 全九组消融已完成并通过CPU审核（2026-10-14）

九组同旧20类/同40轮，全部父子实际退出，2/5/6项目GPU释放，CPU独立保存审核通过，test0。
RGB/dense/global信息保留改善明显；单独更强MixUp未恢复，所有组仍有验证零召回，未达正式门槛。
旧9expert参考train65.82%/val50.88%、gap14.93pp，不能直接代替新物理合同。
[全九真实曲线、逐类指标、CPU审核和原因解释](reports/caltech_diagnose_20261014/all9/README.md)。
下一步有界复合物理修正与最多三项4层正则验证，2/3层也须验证；通过后才正式六组。
下方在训快照是历史，旧图、数据、best/last和test全部保留。

## 消融部分阶段：输入/路由/全局图三组40轮已审核（2026-10-14）

与基线val28.98%比较，RGB输入44.17%、四专家连续路由47.00%、保留478全局场45.94%；
宏召回42.76%/48.17%/45.11%，三项bNLL均改善。全部真实8bit，不新增学习前端。
仍有2/2/4个零召回类，40轮未充分拟合；不将小gap当过拟合已解决或保证75%。
[共同预算曲线、逐类指标和CPU审核](reports/caltech_diagnose_20261014/second3_stage/README.md)。
六个诊断已完成，最后MixUp.2/.6/legacy自动在物理2/5/6接续，test0，源121冻结不变。
所有九完成后全审核，再验证修正/正则/逐类/BMP，通过后才正式六组。

三个真实已训练E40候选的随机/低亮度/黑输入九项设备BMP回放概率误差均0，黑输入五层振幅均0；
源和PT不变、CPU检查无GPU/设备/test，不冒称实拍验收，preserve_global仍需对应478输入独立出口。
[训练后BMP合同和逐层亮度](reports/caltech_diagnose_20261014/trained_candidate_bmp_contract.md)。

同一每类前5 TRAIN组件抽查，Base96/100固定(2,3)，RGB和preserve分别有slot1/2未启用；
Dense四槽均非零且权重随图变化，但不均衡，不把小样本当完整训练统计或因果证明。
[已训练4层路由槽位证据](reports/caltech_diagnose_20261014/trained_L4_router_component.md)。

## 消融部分阶段：前三组40轮CPU已审核（2026-10-14）

基线/关闭量化/整面LN同40轮、同init/order增强，均valNLL选E40。
train/val33.75/28.98%、34.69/30.39%、33.82/32.16%；整面LN零召回减少但macro/NLL变差，
关闭量化仅小幅改善，尚未解决拟合/泛化问题。完整保存CPU审查、PNG/SVG/CSV实际视检通过。
[共同预算曲线和各类support/recall](reports/caltech_diagnose_20261014/first3_stage/README.md)。
三子已退出，RGB/dense/global保留由GPU2/6/5接续，仍未全九完成，不提前正式/test，运行源e4cb冻结。

## 最新GPU指定与持续推进（2026-10-14）

人工要求只用物理GPU2/5/6，最多三子，A100允许；每天北京时间09:00报告实际结果。
旧6/2/1诊断已完整状态迁移，新run `caltech_selected20_s17_L4rho03_regression_diagnose40_gpu256_20261014`
CPU父3077108：GPU6基线、GPU2不量化、GPU5整面LN，三组原E17/E17/E12恢复raw/EMA/Adam完全一致，
已过实际GPU恢复门槛，完成轮次不重播；其余六个40轮消融自动接续，test0，正式训练仍需诊断/逐类/BMP门槛。
121运行来源e4cb1400f冻结，旧prefix582保留；其他项目和旧全部结果保留。
[迁移、实际三GPU状态及独立验证](reports/caltech_diagnose_20261014/README.md)。
下方初始GPU1选择是已被最新指定取代的历史。

## 当前要求：先消融、逐类检查，再正式训练（2026-10-14）

人工授权最多三张GPU，本轮允许A100；旧GPU6禁止只适用于已交付旧run，不适用于本新方案。
先固定旧20类做九个4层残差共同40轮train/val消融，定位退步；旧布局整套参考不是单因素对照。
四层残差后续单独验证正则，逐类零召回/低召回必须报告。新正式类别排除Faces/Motorbikes/airplanes，保留17类、train/val选3替补。
九项真实数值门槛已过，CPU父3054100已启动三卡GPU6/2/1消融（含A100）；旧所有数据/best/last/test/来源冻结保留。正式训练仍未启动，先完成诊断。新类car_side/joshua_tree/stop_sign替补，train1246/val261/test430已冻结。
[消融预算和类别门槛](CALTECH_DIAGNOSE_PROTOCOL_20261014.md)、[实际诊断证据](reports/caltech_diagnose_20261014/README.md)。
下方75%失败和旧A100排除均为已完成旧实验事实。

## 最终状态：六组已交付，75%目标未达到（2026-10-13）

主干2/3/4层各rho0/.3共六组完整150轮，全六CPU审核和排他锁后，各获选EMA首次单次test；验证NLL选模，test未调参。
测试总体准确率无残差/残差依次为2层27.86/53.68%、3层26.31/7.41%、4层50.19/46.41%。
**4层残差46.41%，未达到75%验收目标；不据此换权重、追加或自动重训。**
4层残差宏召回42.81%，train/val57.22/46.64%、差距10.57pp；过拟合/泛化差距仍在。
测试前三大类占81.06%，3层残差在三大类召回为0，须结合宏召回、support和完整CM看结果。
全六CPU保存审计通过，五份PDF已实际渲染视检。所有训练/测试父子实际退出，项目GPU释放，全程无A100。
六组标准/设备联合振幅相位部署资产及SHA已核验，离线合同通过；未连接设备，实拍LUT/ROI/曝光/相干rho仍需新校准。
[最终曲线、测试/过拟合解释、全CM/support、部署与退出证据](reports/caltech_deploy234_20261013/final/README.md)。
下方“在训/test0/尚未验收”均为受保护的历史阶段记录。

## 当前人工要求：可部署主干2／3／4层，最多三卡（2026-10-13）

本节覆盖下方旧八臂200轮计划。人工明确主干计数、router另算，2／3／4层实际
3／4／5次捕获；4层rho.3最终test总体准确率≥75%是验收目标，仍只按验证NLL选权重。
旧队列后续派发已暂停，旧权重／历史／数据／test全部保留，不继续旧pending或旧test。
新增T18独立入口采用封存ABO的2×2专家、478场、532nm／10cm、保零tanh振幅、
CCD log／LayerNorm／ReLU；残差由振幅＋相位联合BMP编码，8bit量化进入训练。
更强MixUp.4、验证NLL降学习率／早停、保留真实trainvalgap，最多150轮／臂；
不会把75%目标或离线BMP一致性冒称已达到精度或已通过实拍验收。
[新布局、非线性、归一化与训练协议](CALTECH_DEPLOY234_PROTOCOL_20261013.md)。

三卡mem2版本已实际启动，4层从E7完整接续，GPU1/2/4、A100排除；
六模型BMP/实际micro1门槛及Linux4项协议检查通过，真实部分状态CPU核对通过。
[新实际启动、布局图、数值与状态证明](reports/caltech_deploy234_20261013/README.md)。

2/4层rho.3已完成150轮，独立CPU保存审核、全部150轮增强/MixUp/梯度/优化器核验通过，test0且无部分选模锁。
均选E150：2层train36.22%/val36.04%，4层train57.22%/val46.64%、macro48.69%、差距10.57pp；
4层验证NLL末段仍改善约1%，泛化差距扩大，不能称充分收敛。75%最终test目标尚未验收。
[已完成两组残差曲线、PDF/CSV及CPU审核](reports/caltech_deploy234_20261013/residual_L2L4_stage/README.md)。

3层rho.3也已完成150轮并通过CPU全部保存核验，选E150：train47.81%/val35.69%、
macro41.26%、gap12.12pp，末十验证NLL仍改善1.14%，泛化差距扩大。三个残差组均完成，test0；
独立导出工具标准/设备BMP命名及尺寸已区分，十项Linux CPU检查与已训练4层静态导出核验通过，
不改运行7源，不连接设备。[3层曲线、审核与部署证明](reports/caltech_deploy234_20261013/residual_L3_stage/README.md)。

4层两rho共同150轮已完成，配对CPU保存审核通过，test0/无部分锁；rho0选E147：
train52.84%/val48.76%、macro49.33%、bNLL2.01008、gap4.08pp；rho.3选E150：
train57.22%/val46.64%、macro48.69%、bNLL1.86911、gap10.57pp。
残差验证准确率低2.12pp但NLL更低、差距更大，不宣称精度收益。
[4层完整配对曲线/PDF/CSV和CPU证据](reports/caltech_deploy234_20261013/L4_pair_stage/README.md)。

## 当前人工要求：全部八组延长至200轮（2026-10-13）

用户明确要求2/4/12/16层两rho全部延至200轮、最多四卡，并选择保留现有正则，
按验证NLL选最佳权重。独立续训入口恢复E100原始模型、EMA、AdamW与学习率调度
全部状态，全球epoch101-200、不重播增强；原100轮最佳仍参与1-200选模。
原100轮已完整结束、各首次test和最终CPU审计已交付，原数据/PT/来源/封存不改。
新结果因旧test已见仍属开发指标；选中旧相同EMA则复用保存test，不重复推理。
只有新选中参数全8审核锁后才首次单test，失败claim不重试。
实测四卡追加100轮总预计6-7小时，最终审核测试另留20-30分钟；预计不保证。
[续训协议及过拟合控制](CALTECH_SELECTED_EXTEND_PROTOCOL_20261013.md)。

09:42已实际启动CPU父2043961：16层GPU1/2、12层GPU0/4四子恢复E100全部状态，
生产数值门槛通过；2/4层随后自动接续。Linux8项CPU检查通过，真实E102/E103
部分记录的恢复/全部相位梯度/独立MixUp/数据/76源码及原Gitblob检查通过，test0。
训练预计今天15:45-16:45结束，完整审核图表交付16:15-17:15（北京时间，估计）。
[真实启动、恢复证明与完成入口](reports/caltech_selected_extend200_20261013/README.md)。

12层两组共同200轮已完成，GPU0/4原子2045853/2045994均退出；完整CPU保存审核通过，
test0，没有部分选模锁。首次最低验证NLL选rho0E182/rho.3E150：train92.49/95.92%、
val61.84/65.72%、macro58.79/63.12%，gap30.66/30.19pp；比原100轮验证有收益，但差距扩大。
末十验证NLL比前十改善仅0.489/0.212%，后期趋平台，不称充分收敛或70成功。
[12层200轮曲线、PDF、CSV和CPU审核](reports/caltech_selected_extend200_20261013/L12/README.md)。

2026-10-13共享GPU0/1/4新增其他占用，原预计窗口已不适用；队列只在允许卡真正空闲后
接续2/4层，若占用持续完整交付可能18点以后。当前及后续训练/测试只允许GPU0/1/2/4，
用户明确要求A100（物理GPU6）不得使用，保持排除；不抢占其他项目、不放宽来源或实验协议。

## 历史人工要求：筛选20类、缓解过拟合、以12层替换8层

新的开发实验使用2/4/12/16层、两rho共同100轮，最多物理GPU0/1/2/4。
固定非学习缩略图特征只用train/val进行一次类别筛选，不按残差收益挑类别；
全部20类和数据身份冻结后从头训练。共用更强仿射增强、MixUp、label smoothing
与相位平滑；旧Caltech20最终结果/权重/数据保持原身份。新任务因验证筛选与
旧test已观测，报告为探索性开发，不能保证精度或残差优势，也不能与旧20类
直接作受控精度比较。详见[新实验协议](CALTECH_SELECTED_PROTOCOL_20261012.md)。

新20类已固定：3719图，train1372/val283/test2064；一次固定特征筛选宏召回
29.79→59.59%仅为筛选参考、不是光学模型成绩。新八臂队列已经实际启动，
12层两rho在GPU1/2、16层在GPU0/4，2/4层自动接续；全4fresh生产门槛通过，
70训练源码原Gitblob/当前文件匹配，test0。数值模式2/4micro4、12/16micro1，
详情和真实启动收据见[新报告](reports/caltech_selected20_20261012/README.md)。

12层两rho已完成共同100轮，CPU保存结果审核与独立100轮MixUp hash核验通过，test0。
验证NLL选rho0E100/rho.3E91：train69.02/80.90%、val54.77/60.78%，gap14.25/20.13pp；
残差验证高6.01pp，但训练提升更大、末段验证趋平台，过拟合迹象仍在，不能称充分收敛。
原12层GPU子退出，队列在GPU1/2自动接4层；16层仍在GPU0/4训练。
[12层曲线、macro/NLL和审核证据](reports/caltech_selected20_20261012/L12/README.md)。

4层两rho共同100轮已完成并通过CPU保存结果及100轮MixUp核验：rho0E99/rho.3E100，
train61.81/76.90%、val49.82/61.84%、gap11.98/15.06pp，残差val高12.01pp。
末十验证NLL仍改善3.377/5.450%，未充分收敛，不能仅凭差距归因为严重过拟合；test0。
原4层子已退出，GPU1/2自动接续2层；16层继续原预算。
[4层完整曲线及审核证据](reports/caltech_selected20_20261012/L4/README.md)。

2层两rho共同100轮已完成并通过CPU保存结果及100轮MixUp核验，均选E100：
train53.57/57.80%、val45.58/51.59%、gap7.99/6.21pp，残差val高6.01pp。
末十验证NLL仍改善2.005/3.040%，train也低，拟合/优化尚不足，不判为严重过拟合或容量上限。
GPU1/2项目进程实际退出释放，只有16层继续；test0。
[2层完整曲线及审核证据](reports/caltech_selected20_20261012/L2/README.md)。

## Caltech20最终结果：八组100轮已交付

2/4/8/16层两rho各100轮，全部验证EMA NLL选模锁定后各首次单test；CPU审核通过。
最高总体test accuracy69.77%（4层rho=.3），未达70%；该组宏平均召回50.06%，
测试类别数量不均，不能混淆两个指标。16层残差总体/宏召回下降，不支持越深越有利。
模型/数据/配置不追加，GPU0/1/2/4全部释放；旧实验保留。
[真实train/val/test、macro、NLL、预算及全部曲线](reports/caltech20_20261012/final/README.md)。
以下启动、在训、未test及预算等待内容均为历史记录。

## 最新人工要求：Caltech20，每类增加训练图像（2026-10-12）

2026-10-12人工指出两层完成后等待预算回复十余小时未推进。现继续原先全深度100轮
比较，不再把重复预算选择当启动条件。直接复用已完成L2两臂，启动fresh L4/L8/L16
各两rho100轮；最多GPU0/1/2/4四卡，初始16pair在1/2、8pair在0/4，释放后接4pair。
两层val41.5/43.5%负结果保留，尚未test。来源/数据/rho/配置不改，执行微批4层4、
8/16层1，原数值及fresh生产门槛不放宽。计划见
[有界八臂计划](caltech20_full_suite_plan.json)，队列仅训练/验证，全8审核锁定后首次test。
新队列已实际启动，父1055706；16层两组在GPU1/2、8层两组在GPU0/4，四新生产门槛通过，
4层两组排队。test0。实际启动收据和后续进展见[Caltech20报告](reports/caltech20_20261012/README.md)。

20类覆盖先前30/50类提议。按原图数量至少80的25类池和固定seed20261012选20类，
每类60个train组、10个val组、其余test；旧50类train/val图片不进入新test。
各深度2/4/8/16两rho共同100轮，按验证EMA balanced NLL最低首次选模。
先两层完整拟合与16层计时，数值/实际速度通过后继续深层；不保证70%或充分收敛。
旧Caltech50两层100轮已完成且test0：rho0 train24.5%/val14.8%，rho0.3 train32.2%/val18.8%，
均选E99；末段NLL仍改善，全部原进程退出。旧数据/结果保留，不扩展50类深层。
完整新合同见[Caltech20协议](CALTECH20_PROTOCOL_20261012.md)，实际数据与运行见
[Caltech20报告](reports/caltech20_20261012/README.md)。

# T18 自然物体2／4／8／16层MoE：相干未调制分量消融

## 新人工要求：换自然多实例任务、共同100轮，最多四GPU（准备阶段）

用户指出COIL受控物体视角任务过于简单，允许四卡、提出100轮；旧COIL结果和一次test身份保留。
下一候选是预固定Caltech101的50个自然物体类别，类别按种子选、不按精度或残差收益挑。
每类20组训练／5组验证／其余测试，像素重复同组，全50类同rawinit.1，四深度两rho共同100轮。
这是明确的小样本50类子任务，不假称完整Caltech101，不保证70%。
先准备及数据核验、fresh train-only数值／两层可学习性／16层计时预检，尚未启动正式八臂。
旧20类fresh小初始化probe表明8／16层micro4未过门槛，不能直接假称可加速。
详见[新协议](CALTECH50_PROTOCOL_20261011.md)；新数据必须重新测试实际执行模式，test保持不读。
下文旧COIL“全部结束／不再追加”只指已经交付的旧run，不限制最新人工批准的新任务。

## 最终交付：COIL20验证选模，八组各首次单次test，GPU已释放

全部训练及唯一一次16层配对10轮初始化调整已结束，不再追加；全8权重／方案锁定后才读取test。
两层test79.44%／81.94%，达到用户70%目标；平均约11秒一轮，30轮约5分半。

|层数|无残差train／val／test|rho=.3 train／val／test|获选epoch|
|---|---|---|---|
|2|94.79／84.17／79.44%|94.27／85.00／81.94%|30／30|
|4|98.44／100.00／91.11%|98.13／94.17／88.89%|30／30|
|8|99.06／97.50／90.28%|97.71／88.33／78.61%|30／30|
|16|85.94／70.83／60.83%|91.15／82.50／71.11%|10／10（重启方案）|

16层按两rho最佳val NLL均值.895909<原base1.809468选整个rawinit.1方案，
没有为无残差反换旧base权重。原base30低精度残差、两方案best／last与预测全部保留，未选方案不test。
2／4／8预算30轮；16总预算每臂base30+一次从头10，获选模型只训练10轮，非累计40轮权重。
全部获选方案预算末端验证NLL仍改善，不能声称已充分收敛或把16较差归因为容量／纯深度效应。
两层与16层部分trainval差较大，需要保留泛化差距；不是只看高train判严重过拟合。
数据是20个已知实体的留出视角识别，不能冒称未见物体个体泛化或普遍自然图像70%。

[最终报告／NLL／macro／支持](reports/coil20_20261011/final/README.md)、
[完整曲线](reports/coil20_20261011/final/learning_curves_all.png)、
[获选汇总](reports/coil20_20261011/final/selected_summary.png)、
[CSV](reports/coil20_20261011/final/final_summary.csv)。CPU独立核验八次test和三划分预测全部通过。
全8global lock SHA4b0a2c00278673052f18ab988b29b97272a4e389b7d86a62fe138067cc7edc3c，
每臂test1／test_used_for_selection=false，CPU审计不做模型复推；GPU1／2／4项目进程实际退出，释放收据已保存。
以下各时刻是历史阶段快照，其“test0／正在训练”不代表最终状态。

## 历史：完整COIL20物体留出视角，两层速度／70%验证门槛通过

按用户偏好小规模自然叶片／物体且类别>8：30类叶片30轮预检速度通过但70%未过，
负结果保留；只再试一次完整作者20物体／1440灰度照片的COIL20留出视角任务。
每相邻6视角成组及精确重复合并，240组，train960／val120／test360，全部图片保留。
不是20个多实例语义类／未见实体泛化。作者公开研究下载，未附明确CC声明，不再分发原图。
两层两rho各30轮已结束并释放GPU，来源commit2b4c28f93／67文件；验证NLL均选E30。
rho0 train94.79%／val84.17%，rho.3 train94.27%／val85.00%，平均10.91／10.61秒一轮，
每组约5分半。两层验证accuracy／macro70%与速度门槛均过，test仍0；全部CPU审计、图CSV已保存。
trainval差约10.6／9.3pp需关注，但valNLL仍下降、best在末轮，没有严重过拟合恶化证据。
4／8层新数据micro4数值过，16层micro4未过、不放宽，正式使用micro1。
两轮计时已经结束，普通137秒、周期均摊投影144秒；不是正式精度臂。
当前协议见[物体有界预检](COIL20_PILOT_PROTOCOL_20261011.md)，
实际数据身份、真实曲线和各来源见[COIL20报告](reports/coil20_20261011/README.md)。
16层两轮计时完成：含全train172秒、普通137秒，周期均摊投影144秒，原计时GPU已释放。
20:46:11新CPU队列`coil20_s17_L2L4L8L16_val_fast_20261011`父654702接续六个深层臂，
GPU1／2为16层rho0／.3，GPU4为8层rho0，之后以current_processes.json核对子PID。
4／8层micro4、16层micro1；每深度两rho同模式、有效16，完整30轮只val选EMA。
源码父队列b8bbe7ccf／泛化控制器29a；复用旧2b来源两层30轮，实际版本分别保留审计。
初三子各67源／发布blob匹配、GPU smoke过。实际普通轮8层约20秒、16层140–144秒，
首轮30轮训练暂估22:00前后，拟合审查／锁定／首次test另算，当前test0。
队列只fit，完成释放GPU后停在待验证审查，不自动提前测test或无限追加。

21:44更新：4／8层两rho各完整30轮，全部保存结果／PT／来源／数据／预测通过独立CPU核验。
4层train98.44／98.13%、val100.00／94.17%；8层train99.06／97.71%、val97.50／88.33%。
全部验证最低NLL首次为E30，test0；真实图CSV见COIL报告L4／L8子目录。GPU4四个fit子已退出。
16层尚在训练，rho.3 E23 val11.67%、E20 train8.96%，明显学习停滞，不是高train过拟合。
独立新鲜train4诊断完成并释放GPU4：分类梯度非零且远大于平滑项，epsilon占比小；
raw初始化范围1→.1时两rho的样本预测差异都增大，但初始NLL更高，不能证明训练会改善。
预登记仅16层一次成对新profile、10轮重启，保留base30；等全部base完成才增加10轮入口并发布。
按两rho各自最低valNLL的均值选择整个pair，精确同分base，不根据test调整，不无限追加。
此变化是初始化优化，模型／rho／数据／类别不改；不把不同预算的结果冒称纯深度因果或已收敛。

22:04阶段更新：全部base八臂完成，父654702及GPU子654708／654709实际退出，GPU全释放。
16层rho0选E30 train95.9375%／val86.6667%／bNLL.63674522；rho.3选E26
train8.6458%／val9.1667%／bNLL2.98219091，负结果全部通过独立CPU审计并保留。
rawinit .1一次配对10轮在`coil20_L16_s17_init01_paired10_20261011`启动，
CPU父695521、GPU1／2子695523／695524，随后以current_processes.json核对，勿重复启动。
来源b7ccea8d6，67实际／发布blob通过，新初始化fresh smoke概率／梯度差0，双方initialSHA一致。
实际配置maximum/requested_epochs均10；原base身份按原commit保护，不修改旧结果。
Linux6项CPU检查通过；全8锁定及独占首次test标记的finalizer两项CPU检查通过，尚未读真实test。
入口`coil20_finalize.py`独立发布；只有候选完成、保存结果审核、成对方案与全8global lock之后才执行test。

## 历史：停止慢速Butterfly，完整30类叶片有界两层预检（已完成）

用户19:29要求更快、两层精度目标≥70%，选择叶片／物体自然图像且类别>8。
Butterfly父560528及三子569626／569692／569761已于19:31:51停止，项目GPU实际释放；
L16两组各3轮、L8rho0共7轮，best／last CPU可读并保留SHA，test0。
停机收据在[原报告](reports/butterfly60_20261011/speed_change_stop_20261011.json)。
下文Butterfly继续训练／100–200轮队列计划已取消，不能按旧PID重启。
新候选是作者完整Medicinal Leaf v1的30类／1835张原生背景处理叶片图，CC BY4；
类别／数量由作者API全文件清单核对，不使用页面矛盾的1500／40类描述。
先核验保留相干残差的批处理数值和速度，再完整两层train／val短程预检，
以验证accuracy及macro≥70%和可接受耗时作为扩展门槛，不能保证成功。
新协议见[快速预检](MEDICINAL_PILOT_PROTOCOL_20261011.md)，未过门槛不直接启动深层长套件。
完整新数据已于19:54前准备核验：train1273／val263／test299，30类无筛选。
两层两rho于19:54:11在GPU1／2启动，独立新profile，训练来源67个逐一匹配Git。
显式保留相干残差的micro4专家batch新数据smoke通过，首轮约18秒；
最多30轮／约30分钟上限，train每5轮和端点，选模只val最低NLL、test完全不评估。
数据身份、完整吞吐数值失败候选、短试验状态和完成后CPU审计见
[叶片快速预检报告](reports/medicinal30_pilot_20261011/README.md)。70%仍是筛查目标，不是保证。
两层30轮已完成、两GPU已退出释放；两组验证NLL均选第30轮。
无残差train45.33%／val42.59%，残差train47.84%／val42.21%，
实际均14.32／14.72秒每轮，总434／446秒。速度通过，70%门槛未过，不展开四深度。
train也低且验证NLL仍下降，是未充分拟合／未收敛迹象，不能归因为严重过拟合。
67来源、PT／config／数据、预测／指标／顺序／增强CPU审计全部通过，真实图和CSV已保存。

## 历史：完整60类Butterfly-10K、2／4／8／16层（已于19:31停止）

用户最新要求换更多类别、按验证集挑选并检查过拟合／欠拟合／未收敛，取消12层。
采用作者完整发布Butterfly-10K，固定revision实际60物种／10,013图，全部保留，
CC BY4；按标本及精确重复连通分组划分。最多GPU1／2／4三张，用完释放。
每深度rho0／rho0.3配对，统一100轮起、未收敛有界延至200，明显过拟合允许
一次配对正则重训；全部方案和PT只看train／val，锁定后获选EMA单次test。
详细数据身份、长尾／许可映射限制、诊断阈值和公平预算见
[新协议](BUTTERFLY_PROTOCOL_20261011.md)。新60输出／四深度入口已发布，
服务器14项CPU检查通过，63个实际／发布依赖SHA一致。
当时队列`butterfly60_s17_L2L4L8L16_val_3gpu_ready_20261011`父PID560528，
截至18:06完整数据准备成功，10013图／6780分组，train6980／val1441／test1592；
标本号及原始／缓存像素精确重复均无跨划分。四深度GPU smoke全部通过，
18:01后三卡开始八臂base100训练，当前GPU1／2为16层rho0／0.3，GPU4为8层rho0，
子PID569626／569692／569761；后续以current_processes.json为准，不重复启动。
18:06启动快照尚无完成epoch，test评估0。实际数据身份及数值检查见新报告。
19:11首个周期train评估完成：8层rho0第5轮train9.04%／val7.49%、
val宏召回20.28%，验证balanced NLL从3.715降至2.901；16层两组各第2轮。
新模式8层普通轮约682秒、周期train轮881秒，均摊约12分钟／轮，
base100暂估约20小时；16层普通轮初测21分钟，周期评估仍待实测，
全suite及适应追加结束时间未定，预计跨天。早期低精度不直接证明欠拟合。
状态与后续真实曲线见[新报告](reports/butterfly60_20261011/README.md)。

FishNet验证队列已于北京时间17:12确认停止，父305617／四层子518458退出，
实际GPU释放，六层完整两组与四层部分best／last／history保留，不能重启旧队列。
12层run于17:05只做smoke即失败，batch2/single概率差2.265e-6、最大梯度相对L2
0.03671，未通过既定门槛，未进入训练；原失败日志和代码保留，现在取消。
六层train／val差约6.7pp／6.1pp，晚期验证NLL略回升，有过拟合迹象，
不能直接称严重过拟合或数据太简单；残差train也低，需同时考虑拟合／优化不足。

## FishNet新数据集配对实验（2026-10-11，现已停止并保留）

用户下午曾授权最多三张GPU并试12层，同时查看train精度／过拟合，随后取消。
当时新增`fishnet_depth12.py`复用原训练函数，
12层六周期、九专家、61张可训练相位、2,649,088参数，GPU2／GPU4分别rho0／rho0.3。
同100轮、seed17、验证NLL选EMA、锁后一次test，结束释放，不预设加深或残差一定改善。
新图配置`fishnet_depth12_config.json`，详细授权和身份见[协议](FISHNET_PROTOCOL_20261011.md)。
新增run `fishnet_val_L12_s17_e100_3gpu_20261011`须先通过12层GPU smoke；
最新启动／检查状态写入[验证选模报告](reports/fishnet_validation_20261011/README.md)。

**当前规则已改为验证集选模。**用户11:00要求暂时不用test选PT；旧test-selected
队列在六／四层两组100轮完成后停止，两层尚未训练，旧结果与best／last完整保留。
四个已完成组验证最优轮72／97／97／97的权重未保存，不能从旧test-best／last冒称补选。
新run `fishnet_val_s17_e100_gpu1_20261011` 六组从头seed17、同数据／划分／光路／优化预算
完整100轮；每轮验证EMA balanced NLL最低选best、精确同分保留首次。
test只在验证PT锁定后评估一次，不逐轮测、不按test挑raw／EMA／last。
默认profile为`fishnet_validation_profile.json`，严格关联原准备profile与缓存SHA。
仍只物理GPU1、microbatch2累积有效batch16；已于北京时间11:07:35启动，父PID305617，
后续子PID以新run的current_process.json为准。训练源码main `ce7edf074`已同步，
Linux11项检查通过；Windows9过／2项原生Torch检查跳过，不改其它环境。
新[验证选模记录](reports/fishnet_validation_20261011/README.md)与旧开发报告分开。
六层rho0已于北京时间14:05:48完成100轮，验证balanced NLL最低选第72轮EMA：
train98.91%／val92.19%／test89.98%（413/459），test宏平均召回86.05%、
balanced NLL0.35783。100轮history无逐轮test，锁定PT后仅测试一次；保存预测、
best／last、缓存及60个发布源码身份CPU核验通过，没有重复test推理。
六层rho0.3已于北京时间16:28:43完成100轮，验证NLL最低选第97轮EMA：
train96.06%／val89.96%／test85.40%（392/459），test宏平均召回81.30%、balanced NLL0.47787。
六层配对初始化、100轮顺序／增强、数据、参数和60个源码身份一致，配置唯一区别rho；
各PT锁定后仅测一次test，CPU核验保存预测与pair_lock通过。本次残差低4.58个百分点，
保留负结果，不追加训练或反换权重。真实PNG／SVG／PDF／CSV见上述验证报告。
四层rho0接续PID518458后按用户换数据要求停止；四／两层未完成，
不提前作整体深度结论。
因旧test已多次观测，新结果仍是探索性评估，不能重命名为独立泛化。
以下test开发选模段落仅属于保留的旧run。

用户要求使用空闲物理GPU1、另试符合许可/类别要求的数据集，保留当前最终结果。
选作者FishNet v1，原生3,013张／八个鱼虾类别、CC BY4.0、DOI10.17632/p3xh4fs7cp.1。
原文件SHA与八类数量已核对；671张HEIC也完整解码，不漏类别、不挑子集。
同T18光路与固定电子输入，2/4/6层各从头配对rho0/rho0.3、100轮seed17，
共同base优化设置；按此前授权逐轮raw/EMA test开发accuracy最高选PT，双方同规则。
这不是独立测试泛化，不能保证残差为正。只GPU1 UUIDe8837b85...，六/四/二层串行。
数据已准备：640×640、train2106/val448/test459、1组精确重复不跨划分。
首run `fishnet_s17_e100_gpu1_20261011`因GPU1临时被其它项目占用而预检失败，未进入训练。
沿用此前剩余显存授权；`fishnet_s17_e100_gpu1_spare_20261011`附加逐像素梯度检查未通过，
只读train诊断确认前向差1.49e-7、梯度方向余弦≥0.999958，但不逐像素精确相同。
保留两次失败启动及诊断记录，采用整体梯度/前向稳定性检查，新run `fishnet_s17_e100_gpu1_checked_20261011`，
`fishnet_ablation.py --use-spare-memory`限制Torch3GiB，双方microbatch2累积有效batch16，
每子进程前核对剩余显存/UUID，不终止其它项目。详见[数据与实验协议](FISHNET_PROTOCOL_20261011.md)。
新run已于北京时间2026-10-11 01:24通过六层GPU smoke，启动父PID11618、
首个六层rho0训练子PID11951（后续PID以current_process.json为准），实际GPU占用约1748MiB。
训练源码main e57e3b518，Linux4项CPU检查通过；Windows3过、Torch原生检查因本地DLL问题跳过，
不改共享环境。数据[manifest](reports/fishnet_data_manifest_20261011.json)与
[数值诊断](reports/fishnet_batchshape_diagnostic_20261011.json)已保存。
六层rho0首轮已完成：val39.06%、test开发EMA39.87%，训练记录179秒（含初始化评估、
首轮train/val/test评估，非纯训练时间）。六组单卡串行暂估15–20小时，受其它GPU进程
竞争和周期评估影响，须按后续实测修正；先产生六层配对结果，不能把首轮当最终表现。
当前100轮六组仍在进行，性能与耗时只按实际history汇报；旧芒果88.04%开发结果、
best/last、失败候选和历史同预算表全部保留。本轮只修改T18，main不新增分支/工作树。

六层rho0已完成100轮（北京时间04:15:53），test开发按既定规则选第65轮EMA：
train98.62%／val91.96%／test91.72%（421/459），test宏平均召回87.86%、
balanced NLL0.36398。PT/58个源码/保存逐样本预测CPU核验通过，没有重复测试。
第100轮raw同为91.72%，但NLL更高；保持既定同分规则。
六层rho0.3也已完成100轮（北京时间07:04:24），选第99轮EMA：
train96.01%／val89.73%／test开发86.27%（396/459），test宏平均召回82.54%、
balanced NLL0.48026。双方仅residual_rho不同，初始化、100轮顺序与增强哈希、
数据、源码与1,329,544参数均一致；保存预测CPU核验，无重复测试。
本次六层30%相干振幅混合比无残差低5.45个百分点，保留负结果，不能以此宣称普遍结论。
两条六层进程均退出。四层rho0已于北京时间09:04:32完成100轮，选第99轮raw：
train93.16%／val89.06%／test开发88.67%（407/459），macro recall84.75%、
balanced NLL0.47853；21张相位、889,696参数与保存预测／58个源码身份核验通过。
四层rho0.3已自动接续PID242090（之后以current_process.json为准），仍仅物理GPU1；
无残差组实测约71.4秒／轮、全组约2小时。四层完整配对预计11:05–11:20，
两层耗时待实测。详见[阶段报告与曲线](reports/fishnet_20261011/README.md)。

## 不再等待空卡：受限显存执行（2026-10-10用户最新指令）

用户明确要求不要等待、立即训练。新`--use-spare-memory`在指定GPU1/4剩余显存
启动，PyTorch分配上限3GiB；启动时至少额外留800MiB余量，不修改或终止其它项目。
这覆盖此前等待独占空卡的安排，仍最多两张卡、只使用用户指定GPU1/4。
`gpu_budget.py`记录UUID、空闲显存、其它进程占用、限制及源码SHA；不足时报错保留日志。
教师有效batch64不变；光学学生microbatch2、累计到有效batch16才裁剪梯度/AdamW更新/EMA，
各小批量损失按样本数加权（包括尾batch、收光/平滑/蒸馏项）。教师训练时正常更新BN，
仅在学生蒸馏时使用eval固定BN统计（本次教师未达门槛，未进入蒸馏）；
光学OEO为逐样本归一化，因此不引入跨样本统计变化；浮点求和顺序仍可能造成细微差异。
原数据、光路、rho、有效优化步数、每轮test开发选模都不变，记录实际执行配置。
实际并行启动：GPU1 `mango_rho03_L6_focal_testdev_spare_20261010`，PID3988180，
focal光学续训、microbatch2／有效batch16、最多30轮；GPU4
`mango_rho03_L6_teacher_testdev_spare_20261010`，教师合格才接学生。
光学focal已完成19轮并早停，test开发选第9轮EMA：train97.50%、val88.48%、
test88.04%（368/418），仅比父多1条正确；末轮raw87.80%、EMA87.56%。
原PT、best/last、19轮history均保留，GPU1本项目进程已退出；
[最终报告与曲线](reports/focal_spare_testdev_20261010.md)核对保存预测、PT SHA和训练源码身份。
教师已完成35轮，test开发选第27轮：train92.54%、val88.73%、test88.28%（369/418）。
未达预设90%门槛，未进行蒸馏，GPU4本项目进程已经退出。
教师分数不能当作光模型分数；[完整教师证据](reports/teacher_bn32_testdev_20261010.json)
核对了history选模、418条唯一预测、混淆矩阵及PT SHA。

## 每轮test开发选模（用户2026-10-10最新覆盖）

用户明确要求训练过程中每轮测test，不再按验证准确率选PT。新run均启用
`--test-development`：教师每轮test，学生每轮分别测raw和EMA，按test accuracy
最高、同分test balanced NLL最低保存best；验证只诊断，早停也由test开发分数决定。
训练梯度和蒸馏输入/标签仍只来自train；此集合现在是开发集，不声称独立泛化。
每轮指标保存history，逐样本预测只保存更新best，best/last保留，不另存逐轮PT。
父87.80%仍参与选择，同权重历史预测直接复用；完成后不重测已选权重。

首个31,752参数教师train/val79.30/78.92%，未满足90%验证门槛，未启动学生，
`mango_rho03_L6_teacher_distill_detpool_20261010`完整保留，teacher未读test。
这支持该小教师欠拟合（训练也低），不能把其成绩当光模型或架构上限。
新训练期教师`bn32`扩大至110,024参数：三Conv通道32/64/128，每层BN/ReLU，
同样固定4×4均值池化，head Dropout0.1+Linear2048→8；仍接收同一固定振幅输入，
仍不进入任何光学推理。最多80轮、其余训练设置相同，不加载外部预训练。
教师test开发accuracy≥90%才接续学生，未满足则停止；学生最多30轮，光路及rho不改。
新run `mango_rho03_L6_teacher_testdev_20261010`，GPU4独占，上限始终两卡（GPU1被其它项目使用）。
该testdev run在教师GPU占用预检失败，未进入训练、未产生有效新权重，日志保留。
此前两张指定卡均有其它项目占用，独占预检失败的run没有开始训练。
用户随后要求不要等待，已改为上文两卡受限剩余显存执行；旧gpuqueue安排取消，
不再重复启动。此有界focal已结束，交付后暂停T18自动跟进，本项目GPU均已释放。
目前光模型最高88.04%（368/418），89.56%以上目标未达到，尚差7个正确样本。

## 训练期电子教师蒸馏（2026-10-10继续优化授权）

MixUp/focal继续训练未突破开发87.80%，[负结果保留](reports/residual_L6_mixup_focal_20261010.md)。
GPU1被其它项目占用，两项实际顺序使用GPU4，均已结束。
接续`distill_suite.py`在GPU4依次训练教师、光学学生和比较保存状态，始终只占一张卡。
教师接收**与光网络相同的1×100×100固定振幅编码**，不是高分辨率原图或额外特征。
电子CNN为Conv(1,16,5,s2)→Conv(16,32,3,s2)→Conv(32,64,3,s2)，每层ReLU，
AdaptiveAvgPool4×4→Linear(1024,8)；只用于训练，不加入光学推理，不加载外部预训练。
教师AdamW lr0.001、weight_decay0.01、batch64，最多40轮/至少10/耐心8，
验证balanced NLL选best。只读train/val，验证宏平均召回≥90%才继续蒸馏，否则停止。
学生从原开发87.80%的同一光学父权重出发；保留完整六层九专家/global/OEO及CCD，
lr0.00015、EMA0.99、最多30轮/至少10/耐心10，原分类/收光/相位平滑项不变，
增加0.3×T² KL(teacher_T || student_T)、T=2，教师固定，输入/软目标都只来自train。
学生T软概率由log(CCD归一化能量)/T的softmax获得；教师只在训练loss内运行。
完成后比较学生best/last raw/last EMA和旧父，原父仍参与用户授权的test开发选模，
同权重预测复用。不改rho、数据、标签、光路、无残差及2/4层，不虚称公平同预算。
run `mango_rho03_L6_teacher_distill_20261010`，教师/学生各存best和last，结束释放GPU。
首次教师在首个batch因CUDA adaptive-average-pool反向不支持严格确定性而失败，
未产生有效训练权重，日志/metadata保留，不算完成结果。改用相同分区的显式均值，
数学上保持4×4 adaptive平均池化，继续保留严格确定性；前向等价与GPU反向须通过。
修复run为`mango_rho03_L6_teacher_distill_detpool_20261010`，旧失败run不覆盖。

## 六层训练期正则继续优化（2026-10-10用户批准）

从测试开发扫描选定`smooth`第8轮best EMA（原PT SHA058b758c...、开发87.80%）
分别训练`mixup`／`focal`。`regularized_continuation.py`仅在本任务定义训练损失，
复用同一光学模型、编码、评估，不修改历史训练器或共享模块。
MixUp在训练仿射后的振幅输入上混合两个训练样本，Beta(0.1,0.1)，
取较大系数保留主样本；相同系数混合平滑soft label，按目标类别加权。
Focal只把训练分类项加上(1-p)^gamma、gamma=1，关注低置信训练类别。
二者都从同一父权重、同seed17出发，lr0.00015余弦至0.000015、EMA0.99，
保留相位平滑0.05和收光loss0.2；最多30轮、至少10、patience10、
验证balanced NLL min_delta0.0002选best EMA，父epoch0仍参与best选择。
优化器/EMA重新初始化，增强epoch偏移200；只存best/last与完整曲线。
训练梯度仅来自train，验证用于早停；完成后按用户已授权test开发口径比较
两个候选的best／last raw／last EMA，加原父状态，共7个候选，重复权重复用。
这不是独立泛化；不改光路、rho0.3、数据划分、类别、2/4层或无残差基线。
仅物理GPU1/4、上限两卡，run `mango_rho03_L6_mixup_focal_20261010`，结束释放。

## 测试集选模开发扫描（2026-10-10用户明确批准）

用户要求直接在test上选PT、查看当前最高值。`test_sweep.py`只比较保留的
Mango六层rho0.3、seed17各训练臂best EMA及last原始/EMA状态，不训练、不集成，
不改变数据/光路，不重选无残差基线。扫描前锁定36个状态的文件/张量SHA、配置、
已发布训练源码身份与数据SHA；同状态已有逐样本测试记录直接复用。
按test accuracy最高、同分test balanced NLL最低选择，导出的权重保留训练来源
并明确记录原文件和state key。结果只能称**测试集选模的开发成绩**，不是独立泛化，
也不是与固定无残差组同预算的公平消融。原验证选模结果与失败记录全部保留。
扫描已完成：12个训练臂、36状态／33组不同权重，27组新增推理，其余复用；
最高87.80%（367/418），选中`smooth`第8轮best EMA（test同分按NLL选择）。
宏平均召回88.06%，仍未达到89.56%目标；比固定无残差仅多一个正确样本。
[完整比较与权重SHA](reports/test_selected_L6_20261010/README.md)及PNG/SVG/PDF/CSV
已保存。开发选定PT独立导出到扫描run，不覆盖旧best/last；GPU4已释放。

## 当前追加：只优化30%残差（2026-10-10用户批准）

用户要求抓紧冲刺六层89.56%以上：从同一EMA best epoch14启动仅验证的
`lowlr`和`lowlr_ls`候选，两者lr0.0001余弦至0.00001、EMA0.99；后者仅额外
把label smoothing从0.02增至0.05以缓解过度自信，其它数据/模型/光路/损失不变。
run `mango_rho03_L6_lowlr_ls_20261010`，GPU1/4最多两卡，最多50轮、至少15、
patience12，仍先按验证选模。目标非结果保证：测试418条，达到89.56%以上
至少须375条正确（89.71%），较当前364条需净增加11条。
采用条件仍为验证NLL改善且宏平均召回>=父88.8571%，合格者按NLL选胜者，
先锁SHA再仅测试胜者一次；不以测试调参，无残差及2/4层不动。
本轮已完成：lowlr epoch9/21验证NLL0.320529、召回88.6257%，lowlr_ls
epoch1/15为0.322030、88.3669%，均未达到采用条件，均未测试。
六层保留父EMA原测试87.08%，89.56%目标未达到；项目GPU已释放。
[失败记录与曲线](reports/residual_L6_lowlr_ls_20261010.md)已保存。

用户再次要求六层优化：从EMA候选best epoch14同一权重接续两个仅验证候选，
`capture`将收光损失0.2降到0.05，`smooth`将相位平滑0.02升到0.05，
其它均继承lr0.0003/EMA0.99/原增强，最多50轮/minimum15/patience12。
run `mango_rho03_L6_capture_smooth_20261010`，GPU1/4，上限两卡。
动机是训练−验证差约8.51pp及辅助收光目标的潜在竞争，尚非已证实原因。
只按验证选择：NLL改善且宏平均召回不低于父88.8571%才合格，父NLL0.320628；
合格者按NLL选胜者，锁定后只测胜者一次，失败者不测试。无残差和2/4层不动。
本轮已结束：capture验证NLL0.321226变差；smooth NLL0.320538略降但召回
88.8193%低于父88.8571%，均未达到采用条件，均不测试。继续保留父EMA六层
测试87.08%，项目GPU已释放。[本轮完整报告](reports/residual_L6_capture_smooth_20261010.md)
保留失败证据与学习曲线，不自动无限追加。

公式改以alpha及模块算子描述，router无残差；当前非线性是OEO而非Kerr，
[完整表述](OPTICAL_RESIDUAL_FORMULATION.md)注明传播与残差位置。
用户要求进一步优化六层后，另比较aug/ema两个验证候选：从同一round2最佳
epoch11出发，lr0.0003余弦、最多50轮/minimum15/patience12；aug只增强到
旋转15度/平移4/尺度10%，ema只把EMA改0.99，模型与光路不变。
分别使用GPU1/4，run `mango_rho03_L6_aug_ema_20261010`，不自动测试。
候选仅在验证NLL优于父且验证宏平均召回不低于父时考虑采用，二者按验证NLL
选胜者，只对胜者做一次探索性测试；失败者不测试，不按测试挑方案。
该轮已完成：aug验证下降被拒绝不测试；EMA0.99获选epoch14/26轮，
验证NLL0.32063、宏平均召回88.86%，一次测试87.08%。最新六层仍比固定
无残差87.56%低0.48pp；历史残差87.56%保留，不能按已见测试反换权重。
[本轮报告](reports/residual_L6_aug_ema_20261010.md)及图表已保存，GPU已释放。

用户随后要求继续六层：从GPU4第二段续训最佳epoch25（测试87.56%）
再warm-start最多50轮，lr0.0006、minimum15、patience12，其余不变；
仅验证，旧父仍参与选择。run `mango_rho03_L6_continue50_round2_20261010`，
仅GPU4(UUID 1b963983...)，新一轮只因用户明确继续而启动，结束释放。
该轮已完成：23轮早停，按NLL改进阈值选epoch11，验证NLL0.32443；一次测试
86.84%（父87.56%），测试NLL0.38681（父0.39334）。最新验证选定六层结果
应记录86.84%，不能按已见测试反换父权重；历史87.56%保留原身份。
[本轮报告](reports/residual_L6_round2_20261010.md)记录阈值、早停与指标分歧。
GPU已释放，不自动无限续训。

本轮已完成并按验证NLL锁定三个获选权重，各仅测试一次：2/4/6层残差
75.36/84.21/87.56%，对固定无残差基线差+0.96/+1.91/0.00pp。
最终使用用户指定GPU1和GPU4，全部项目GPU已释放。
[追加报告与迁移身份](reports/residual_continuation_20261010.md)明确预算不对等、
旧run保留、六层换卡重启优化器及单种子限制；此轮并未确定性能理论上限。

无残差展示基线固定为原100轮74.40/82.30/87.56%，不再训练。
从lr0.003候选各深度rho0.3最佳EMA权重续训，当前起点测试
75.36/83.25/86.60%。`continue_residual.py`复用历史训练器且不改原source身份，
重启AdamW和EMA，lr0.0006余弦至0.00006，最多50轮、至少15轮、patience12；
其余损失、增强、光路、数据不变，增强/顺序epoch偏移100避免重复前期序列。
原最佳checkpoint仍参与验证NLL选择，续训变差不采用；训练不读test。
`continue_suite.py`三个深度各一卡，仅验证，不自动测试；run
`runs/simulation/mango_rho03_continue50_20261010_3gpu`。
这是仅残差额外预算的性能探索，不能冒称与固定基线同预算公平消融。
用户随后将上限减为两张GPU：两层进程已停止，已有best/last和日志保留，
四/六层继续。父调度先停止以防一个子任务退出触发整组清理，后续分别监控与汇总；
不恢复三卡调度。以UUID核对，结束释放，完成/失败/需处理时自动通知。

2026-10-10用户要求在未使用过的CC BY4.0数据集重做：另设
[MangoLeafVarietyBD v2原始八品种协议](MANGO_VARIETY_PROTOCOL.md)，新run及图表
与下方Kather历史分开。光路/残差及训练预算不变；输入类别含义变更。
新数据六组已完成：2/4层残差测试提升1.67/1.20pp，6层下降7.18pp；
不能声称残差跨数据集始终有益。精确结果、划分及证据见上述协议。
同预算100轮也已完成：2/4/6层无残差测试74.40/82.30/87.56%，残差
73.92/82.54/83.97%，不支持随深度扩大优势。当前训练/验证诊断及获选epoch
见协议；[30/100轮对照图](reports/mango_figures_20261010/depth_accuracy.png)、
[完整学习曲线](reports/mango_figures_20261010/learning_curves.png)、
[可编辑SVG](reports/mango_figures_20261010/depth_accuracy.svg)、
[数值CSV](reports/mango_figures_20261010/source_data.csv)。公共lr0.003候选已完成，
六组平均验证NLL由0.524229降至0.477688，整套配置锁定后各测试一次。
最终2/4/6层无残差74.64/82.78/89.23%，rho0.3为75.36/83.25/86.60%；
仍不支持深度越大残差优势越大。原结果保留，未开启第三个方案，项目GPU已释放。
详细训练/验证/测试、最佳/末轮、泛化差、梯度及功率见
[候选最终报告](reports/mango_lr3_final_20261010.md)。新图并列保留三套配置，
学习曲线展示最终公共lr0.003候选；单种子探索不加误差条。

新增两层主干消融：router→九专家并行相位→OEO→global相位→OEO→CCD
（相位之后均有原5cm传播，末端另有5cm传播）。专家各1张相位，共9专家＋1global，
router额外1张，总参数449,848；2次主干OEO。`--depth 2`，rho0／0.3配对，
30轮seed17，其余设置不变。run：`runs/simulation/paired_L2_s17_20261009`。
两层需独立smoke；与4／6层结果一同报告，保持每个深度内部配对同预算。

2026-10-09新增用户批准的四层主干配对：router→专家层1→global1→专家层2→global2。
仍是九专家，专家各2张相位，共18专家＋2global相位，router额外1张；
总参数889,696，4次OEO。rho0/0.3、30轮、seed17，其余数据和训练配置不变。
入口新增`--depth 4`（默认6）；四层run为`runs/simulation/paired_L4_s17_20261009`。
旧六层成绩保留，四层结构需独立smoke，不能把九个并行专家误称九层。

用户2026-10-09批准振幅混合：`U'=[(1-rho) exp(i phi)+rho] U`，固定rho=0或0.3。
不是30%功率；相干干涉使透射功率依赖相位。理想透射强度范围为
`(1-2rho)^2`至1，rho0.3时为0.16至1。不额外做逐层能量恢复，OEO沿用原协议。
未调制支路绕过相位调制，但仍经相同传播与OEO，不是绕过整个块的恒等捷径。

## 完整模型合同

沿用 `demo_check/reproduction/kather2016_experiment.py` 的历史六层MoE＋逐层OEO。
RGB150×150 → 固定抗混叠缩放/训练仿射 → 四块50×50 `[R,G;B,meanRGB]`
→ 振幅100×100。没有CNN、Qwen或可训练电子读出头。
路由相位100×100、5cm传播、九个32×32探测框，温度100 softmax得p，入口振幅
`p/sqrt(sum(p^2))`；3×3九专家均激活，理想relay复制同一图像，孔径146×146、
间距176。router不加残差，保持输入条件路由不变。

三个周期，每周期九张独立专家相位（局部padding30、206×206角谱传播裁回146）
→ 整面专家OEO → 全局498×498相位 → 全场传播 → 全局OEO。
九并行相位面算一层，每周期两层，合计六层＋一个额外router。
总参数1,329,544，仅相位参数；27专家相位＋3全局相位采用固定rho，router原样。
六次OEO：有效498×498整面强度LayerNorm（无仿射，eps1e-6）、ReLU、Softsign、
零相位振幅重编码。波长532nm、像素8um、画布500×500、主干及末端距离5cm。
最后八个互不重叠32×32 CCD框（两行四列、起点126/198、步距40），能量归一化
后argmax分类，无电子Linear。初相位沿用此历史协议raw uniform[-1,1]、2pi sigmoid。

任务模型仅替换30张主干PhaseLayer的前向；保留同一Parameter与state_dict键。
为确保专家批量FFT捷径不会绕过残差，**两组都关闭专家vectorize**，使用同一串行
专家实现。smoke须核验rho0与原计算一致、初始参数逐元素相同、全部主干都有梯度。

## 数据、配对与选模

Kather2016原5000张八分类，固定3496/752/752 train/val/test、每类437/94/94。
原数据CC BY4.0；固定缓存及manifest哈希核验，图像级划分，非患者独立验证。
服务器缓存 `/DATA/DATA1/guest3/demo_reproduction_data/kather2016/kather2016_fixed_split.npz`。
两组从头训练：seed17、batch16、AdamW lr0.002余弦至0.0002、30轮上限、至少15轮、
patience8、EMA0.95、梯度裁剪1、label smoothing0.02；分类NLL＋0.2捕获损失＋
0.02圆周相位平滑；训练旋转±10°、平移±3、尺度±5%。所有设置共用，验证NLL
选EMA checkpoint，不以测试选比例/超参。相同seed保证相同初始化、样本顺序和增强。
历史Kather测试已见过，本实验属探索性单种子，不给标准差或预设残差一定更好。
rho0重新训练作配对对照，不直接拿旧MoE表混作同次实验。

每组run写入本任务 `runs/simulation/`，只存best/last，保存配置、命令、Git、
源SHA、数据SHA、环境、功率诊断、完整曲线、逐样本结果。训练不读test；两组完成
后核验配对并锁定，再分别测试一次。GPU上限两张，按UUID选择空闲卡，进程退出释放。
不修改历史源码或旧run，不新建分支/工作树，源码按main发布后服务器Git同步。

入口：`run.py --phase smoke|train|evaluate --rho 0|0.3 --data ... --out ...`。
配对执行入口：`campaign.py --data ... --out runs/simulation/paired_s17_20261009
--gpus GPU-<uuid1> GPU-<uuid2>`，预检两张物理卡无占用，再并行训练。
自动核验样本顺序/增强哈希相同并锁定结果，之后串行各测试一次。
smoke `runs/smoke/contract_s17_20261009` 已通过rho0精确前向、初始化不变、
30主干相位及router分类梯度与干涉功率范围检查。

## seed17配对结果（已完成）

| 固定未调制振幅系数 | 验证选中轮次 | 验证准确率 | 测试准确率 | 验证NLL | 测试NLL |
|---|---:|---:|---:|---:|---:|
| 0（无残差） | 26 | 71.94% | 69.28% | 0.68653 | 0.69506 |
| 0.3（相干残差） | 29 | 72.47% | 72.07% | 0.73062 | 0.71742 |

测试准确率提升2.79个百分点，验证提升0.53个百分点；但NLL反而较高，不能声称
所有指标改善，也不能以单种子证明稳定优势。八类均衡，因此准确率等于宏平均召回。
测试窗口捕获率46.60%→49.51%，是窗口／整场能量比例，不能称总光效率提高。
两组均30轮，同初始化规则、1,329,544参数，样本顺序及增强哈希完全一致。
获选权重各只测试一次；run状态complete，训练和评估进程均退出，两张项目GPU已释放。
实际训练commit及完整指标、权重SHA、逐层功率见
[结果证据](reports/paired_s17_20261009.json)。原曲线、best/last和逐样本结果在
服务器本任务`runs/simulation/paired_s17_20261009`保留，未改历史消融表。

## 四层主干seed17配对结果（已完成）

| 未调制振幅系数 | 验证选中轮次 | 验证准确率 | 测试准确率 | 验证NLL | 测试NLL |
|---|---:|---:|---:|---:|---:|
| 0（无残差） | 29 | 69.41% | 67.02% | 0.75982 | 0.76020 |
| 0.3（相干残差） | 29 | 71.68% | 69.68% | 0.74891 | 0.73682 |

四层残差组测试提升2.66个百分点，验证提升2.26个百分点，NLL也改善。
两组同30轮、seed17、889,696参数，样本顺序及增强哈希通过配对核验，
每组获选checkpoint只测试一次。完整证据见
[四层配对记录](reports/paired_L4_s17_20261009.json)。两张项目GPU已释放。

| 主干深度 | 无残差测试 | rho0.3测试 | 残差准确率增益 |
|---|---:|---:|---:|
| 2（1专家层＋1global） | 62.63% | 65.29% | +2.66pp |
| 4（2专家层＋2global） | 67.02% | 69.68% | +2.66pp |
| 6（3专家层＋3global） | 69.28% | 72.07% | +2.79pp |

三个深度单种子探索均观察到准确率改善；尚不能据此断言跨种子稳定增益。

## 两层主干seed17配对结果（已完成）

| 未调制振幅系数 | 验证选中轮次 | 验证准确率 | 测试准确率 | 验证NLL | 测试NLL |
|---|---:|---:|---:|---:|---:|
| 0（无残差） | 29 | 64.10% | 62.63% | 1.01213 | 1.00278 |
| 0.3（相干残差） | 29 | 65.69% | 65.29% | 0.97179 | 0.97647 |

两层测试提升2.66个百分点（20／752个样本的净准确数差，不代表每个样本都改善），
验证提升1.60个百分点。两组同30轮、seed17、449,848参数，样本顺序／增强
配对哈希一致，获选权重各只测试一次。完整证据见
[两层配对记录](reports/paired_L2_s17_20261009.json)。本项目两张GPU已释放。
2／4／6层三组配对均完成；旧三种子历史表保持其原含义，不混用单种子结果。
