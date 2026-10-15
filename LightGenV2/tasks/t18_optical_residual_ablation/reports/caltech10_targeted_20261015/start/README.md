# 基于诊断主动换策略：二层蒸馏与四层量化SAM实际启动

最新人工要求专家针对性分析与换策略，不接受只重复学习率/长时间同配置。新203run `caltech10_L2_distill_L4_quantSAM_s17_20261015`，训练来源 `1c76b60c5c67b38864eb6f2aae84698700b80e9e`=current/original，CPU3057111实际运行，GPU2二层蒸馏/GPU4四层量化尺度SAM各一子，避免繁忙GPU0。自己的新恢复态门槛、首个完整10轮段独占科学及选中测试均已真实passed，无failure/A1000，不是已达到82/10pp。旧197四条SAM/dropout已全部真实退出且各科学/测试SHA独立核一致，完整负结果见prior_methods_final.csv，不重训或重测。

诊断是新199source、两固定TRAIN macro16每深度，optimizer0/test0；诊断SHA `d14eea9a7bc58e15556bc56689161007dfac6a81bf4cf76390b7e649cf221e09`。辅助捕获项梯度为分类项约5-12%，不能认定主导训练。旧SAM半径.05实际仅改变.17-.32%8bit相位码，.5改变1.6-2.9%（四层2.24/2.42%），所以依据训练物理量化尺度选.5；这仅是策略依据，不证明精度必升。全部精确扰动还原，来源与结果保留，聚合见diagnosis_summary.csv。

二层train78.06而bonsai训练召回35.96%，主要有拟合困难，旧dropout继续降低拟合。新L2_distill从fullLAST950 rawEMA/Adam45600/scheduler950恢复；教师为冻结已接受四层E488，在相同训练增强/MixUp输入上提供soft targets，alpha.5/T2，仅教师同意dominant训练标签且maxprob>=.5时参与。教师SHA `60d21da0fe1536651bf19a08ddf307bf54fec1f39239d06fe91cab9e193f9495`，参数digest `dba67338fa0b3018e29df03b4731e2e2a1c51aa7230537af38b7905fa8c6ee4a`；教师eval/no-grad、不被优化，不参与二层验证/测试/部署，二层仍原6张量479364相位/原纯光推理。保留原MixUp.1/LS.02/TV.01/capture.2/EMA.95，不用SAM/dropout；一次951LR.0003/statistics重启，Adam矩/EMA和globalepoch保留，不重播。[蒸馏原论文](https://arxiv.org/abs/1503.02531)。

四层train99.61/val87.26，泛化差仍在。新L4_qsam从fullLAST725 rawEMA/Adam34800/scheduler725恢复，SAM半径.05->.5，其余MixUp.4/LS.08/TV.05/capture.2/EMA.98不变、dropout0；一次726LR.0001/statistics重启，保留globalepoch/Adam矩/EMA。完整macro16/tail9两次反传、clone精确还原后只有一次Adam/EMA；908552相位11张量、8bit推理保持。旧E488候选保留，不因GPU空闲盲目重复试验。[SAM原论文](https://arxiv.org/abs/2010.01412)。

最新已完整核验的启动后结果（中途快照，不当最终；所有百分比同一选中模型）：

| 路线 | 实际总轮数/本次新轮数/选中E | 训练 | 验证 | 总体测试 | 测试macro | 对冻结100总体差 |
|---|---:|---:|---:|---:|---:|---:|
| L2_distill | 970/20/E970 | 78.5808% | 73.8854% | 80.4692% | 75.4979% | +2.7823pp |
| L4_qsam | 735/10/E488 | 99.6058% | 87.2611% | 93.3988% | 90.1104% | +9.4926pp |

二层新增20轮test80.4692%比前选79.7054%提高.7638pp，但valaccuracy并未相应提高、macro75.4979%，未82；不能只挑test上涨宣称泛化解决。四层本快照新10轮未改善选模，旧权重真实复用test，不第二次forward，差仍9.492635未10。对照全部实际100冻结：二层E98 train67.9369/val64.9682/test77.6869/macro71.8949%；四层E100 train84.7569/val78.3439/test83.9062/macro80.6888%。三层已接受125/E123/test87.4523 vs80.2510差7.2013pp不动。

CPUproof `efbc1d010747506d16e68ed187d6e41720409dca736c20d4a295aa28d70b1d2f`真实一次通过：父950/725、诊断身份/教师PT、SAM解析恢复/单toyAdam和KD解析梯度，真实模型forward/test0。两新numeric和actualproducer各自检查ordinary+新方法all10/macro16/tail9 micro1vs8、教师冻结/真实方法梯度，最大rel1.755e-7/6.437e-7、optimizer0/test0。首个完整段身份：

- L2_distill: numeric `5eb2978819f91130e6c586a2cab60cbc9fb907a721156571aebe886a6e9e02e5`；真实生产/科学rc0退出，science `8ef4eb5b07a27825ffc0d8561e003fa92b41d2190877038b7bfd7c14df7c9151`；selected PT `ef846f66187adc4495e852ce01a3fd7053cbc272db39b0ade785b81e5dea9e46`；test result `a1c26b9820183c686bed679542e70773db9a0c3bbffa6b8efe671c549b0a5cce`/saved audit `679d61454c62d7c5e7e5f57966dd357c48cf9a7fd78b5a51e100a537377ab245`，completedSHA真实匹配。
- L4_qsam: numeric `25fd157b469f6dd61ac5051157dc7ee0bdef109e13d3b36e924582215a9618e2`；真实生产/科学rc0退出，science `cd96a838677e812c65e4d7bf56d52839a03bad09c6ee20fca023ff600bcacfd1`；selected PT `a3828f7709661b5dc39bedfd63668aa8ab91dadeea2e62c4bd86ac485cc97848`；test result `4d20da3245da1072b4c88b409d9db0a689cb5453cfa9a4e73e8e0c2d9c0f0367`/saved audit `0f79d6375caa733049ee4a363164229af50f1ee1df95313a71303def6577e9b7`，completedSHA真实匹配。

每10新轮完整保存+自己的独占科学：来源/config/PT/完整prefix/rawEMA/6or11grad/Adam48E/orderaffineMix/三次历史LRrestart replay/逐macro教师计数KL或SAM精确恢复/保存ID-label-argmax-CM-macroNLL/firstminimum。实际producer rc0/退出及自己的audit/completedSHA真匹配才测新selected；新权重一次、旧参数复用，继承所有旧路线已知digest。全prefix首次最低验证EMA balancedNLL选模，不按最高test选轮。额外残差预算+蒸馏+测试知情开发，不能当独立泛化或同预算纯残差因果优势；旧测试已观察与近重复未排除、test支持极不均等局限保持。固定十类/数据/角色ID/光学模型/8bit原推理不改，无电子分类头。

实时复盘已落实：每10轮记录当前endpoint train/val，和selected模型指标分开；20新轮NLL无.001改善且valaccuracy无新高则四层进入策略复盘，二层还须10轮无.5pp训练进步才暂停。不是只看LR/NLL，也不重复同配置死等75轮；无固定总epoch cap，1800fit秒每段故障保护。t18改为5分钟静默复盘、09保持ACTIVE。新人工明确允许在这些固定约束内依据新诊断继续换训练策略，先冻结新config/source/gates并保留旧run，不盲扫/改类/削弱对照/挑test最高轮。

源码仅Git、rawJSON/逐样本ID/PT/日志私有，旧所有sources/claims/数据/失败/图保留。Git精确T18paths串行，foreignT06保持，servertrackedclean且incoming203来源SHA不变才ff。目标实际二层test>=82/四层总体差>=10，不四舍五入或承诺实现。普通进展静默，真实达标或需处理一次通知。
