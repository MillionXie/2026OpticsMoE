# 根据梯度与颜色敏感性实测启动两条针对性训练

新213run `caltech10_L2_focal_L4_color_consistency_s17_20261015`，真实训练来源 `3d7abe1edc4945076d76ec9b9f32c3876c1f98fc`=current/original，CPU3257149、GPU2=L2_focal/GPU4=L4_color各一子已实际训练，root focal_color_refinements_running，globalT18GPU2/A1000/无failure。新CPUproof/两numeric及两actualproducer自己的恢复态门槛均真实passed；这是新模型训练，不是重复旧203/197worker。无残差100与已接受三层保持冻结。

新209TRAIN-only诊断来源 `dcee960dfe1f7d1dab49dd0b7ee54bad6b9b0a80`，SHA `2170e946282cd54e373faaa72d9ff64fad25d1bf5b161898ac785c2472f1c2aa`，两个固定macro20梯度组和TRAIN761固定光照敏感性，0Adam/0验证图/0测试图，模型digest保持。承接207已证实原教师门控89.93%和几何增强TRAIN98.4%以上，不按错误的“教师不给信号”假设盲调。梯度/逐类光照实测聚合见train_diagnosis.csv。

二层KD梯度与分类方向cos为均衡组.6602、bonsai+airplanes难组.9143，强度仅CE17-23%，不支持“KD主导且冲突”说法。预先固定的focalgamma1.5保持CE方向cos.9801/.9997，难组梯度增14.4%、均衡组增9.4%；这是两个小组的训练依据，不是精度收益证明。新L2将监督weightedCE改为 `(1-sum(softtarget*p))^1.5 * weightedCE`，保留已帮助测试提升的KDalpha.5/T2正确conf.5门控、MixUp.1/LS.02/TV.01/capture.2/EMA.95；不套dropout/SAM去压低本已不足的拟合。fullLAST1040 raw/EMA/Adam49920/scheduler1040完整恢复，1041起一次LR.0003/statistics重启、Adam矩与EMA/全局epoch保留，无重播。

四层在现有几何变换下TRAIN98%以上，但冷色+低对比TRAIN降到88.3049%（Faces78/Motorbikes84），暖色+高对比96.7148%，原99.6058%。故本次针对原训练分布缺少颜色/对比度变化：每TRAIN样本CPU确定性RGBgain和contrast各uniform[.8,1.2]，clamp/roundbyte后走原几何增强/编码/MixUp；原图增强版本的冻结已接受E488教师为有颜色变化学生提供alpha.25/T2一致性，仍要求教师correct/conf>=.5。FullLAST745 raw/EMA/Adam35760/scheduler745完整恢复，746起一次LR.0001/statistics重启、Adam矩/EMA/epoch保持；MixUp.4/LS.08/TV.05/capture.2/EMA.98原样、不用SAM/dropout。教师仅训练，不进入验证/测试/部署，原四层纯光推理结构/8bit不改。颜色敏感不等于证明heldout误差全由颜色造成，也不保证差距到10。

教师PT SHA `60d21da0fe1536651bf19a08ddf307bf54fec1f39239d06fe91cab9e193f9495`，frozen/no-grad、从不优化。两条新策略只用现固定10类与原角色/cache，原L2 selected1018的80.7420%、L4 selected488的93.3988%与冻结对照77.6869/83.9062、额外预算与测试知情开发局限均保留，[前阶段完整实际轮数/选中E/train/val/test/macro/逐类诊断](../../caltech10_targeted_20261015/review_1040_745/README.md)。本启动快照仅早期训练，不冒称新完整段科学或新测试已完成。

CPUproof `a89b4b72cd82e8687622caed17adefc90669d18134c8972de4bde6a95fadf9e6`真实一次passed：父1040/745/209证据/教师PT身份、独立审核装载、focal有限差分/解析KD及CPU颜色确定性/零强度恒等，0realmodel/forward/test。

- L2_focal: numeric `a91258b11daaad80362032d7b21ca0007cda0c18f13ca308e7ed076882a89710`，新方法最大梯度relativeL2 `1.801379881e-07`，all10/macro16/tail9 micro1vs8、教师冻结/参数保持，optimizer0/test0；实际producer自己的ordinary+新方法恢复门槛另真实passed。
- L4_color: numeric `32cba36d048c2c48c57a28ee0afd75aff472a3875edfb374e77e8d019889fe90`，新方法最大梯度relativeL2 `6.46155172e-07`，all10/macro16/tail9 micro1vs8、教师冻结/参数保持，optimizer0/test0；实际producer自己的ordinary+新方法恢复门槛另真实passed。

每10新轮保存完整段，1800fit秒仅故障保护、无预设总epochcap。自己的独占science核source/config/prefix/rawEMA/shapeinit/6或11grad/Adam48E/ordersaffineMix/全部历史LRrestart replay/方法证据，四层逐batch颜色byte SHA由独立CPU重建，保存IDlabelargmaxCMsupportmacroNLL/firstminimum；生产原始rc0且实际退出、ownauditSHA/completed/auditrc0真匹配后，selected新digest最多测试一次。继承所有旧digests至少L2=33/L4=14、相同参数真实reuse，共享原子claims，冻结100不重测。全prefix首次最小valEMA balancedNLL选，不按test最高epoch。

二层目标actualtest>=82，四层actual总体test差>=10pp，不四舍五入；旧9.4926pp可接受候选保留。20新轮无.001显著NLL改善且valaccuracy无新高则复盘，二层另要求训练无.5pp进步10轮；selected与当前endpoint分开。只按真实结果报告，不用新训练准入passed代替科学/精度达标。未来改策略先新来源/准入，不篡改本固定213。

固定十类761/157/1833、test大类支持不均须macro/全逐类；验证知情筛类和先前多次test反馈开发，额外残差预算+教师/focal/颜色策略，非独立泛化或同预算纯残差因果优势，近重复未独立排除。旧205失败和207修复真实证据保留，不再旧helper。源码仅Git，rawJSON/ID/PT/log私有，仅此新目录两聚合路径写入，shared README/T06/L3bench并行工作不修改。所有旧源/数据/权重/history/claims/失败/图保持。两monitorACTIVE，普通进度/内部启动静默，真实新失败或目标达成一次通知。
