# 新SAM与训练输入dropout比较：四卡实际启动

最新人工明确允许用dropout/SAM等方法继续二/四层优化。新run `caltech10_L2L4_sam_vs_dropout_s17_20261015`，197训练来源 `925a8a7a4278e60c1da93dd0771fa9ad6b3eb66d`=current/original，CPU campaign2889673已实际启动，四个worker/四张非A100一子一卡，自己的新方法恢复态gate全部passed、无failure，A1000。不是仍在只调学习率，也不是已达到82/10pp。

| 路线 | 物理GPU | 实际早期轮次 | 暂时验证 | 暂时bNLL | 本次numeric proof SHA |
|---|---:|---:|---:|---:|---|
| L2_sam | 0 | E728 | 74.5223% | 1.1769637319 | `333799e6348a61faedf1d34dc93f6e8e60d1b21eefef2ceaaa4ac12c2d38a220` |
| L2_dropout | 2 | E730 | 73.2484% | 1.1783064023 | `c0240aefb9c50c1559d8adf7724821e7f18278540b856c9a380a0321547751ce` |
| L4_sam | 4 | E652 | 87.2611% | 0.5756823338 | `e0f32adf1306dd723dcf161273b85484a9e1422174d31726983505e506de81a7` |
| L4_dropout | 5 | E654 | 85.9873% | 0.5657665193 | `a757d33b5261068d5d4452205e2c75417ee2246918fe0478dadc98ec19b4350e` |

上述仅真实启动快照，不跨轮拼差距或把早期验证当最终测试。四层dropout早期NLL改善仍须完整段、独占科学及选中模型实测后确认。CPUproof `5b4c25e7fe345a3348ab50893ab87b827df870b8f2c8cccb4ae04d70da6563e8`：真实父恢复/独立审核编译、toySAM解析二次梯度+精确参数恢复+一次toyAdam、确定性mask，真实模型/真实forward/test0。四numeric的ordinary+新方法all10/macro16/tail9 micro1vs8均真实通过，最大新方法梯度相对误差6.11e-7、optimizer0/test0；四个actualproducer也各独立通过自己的gate，不套numeric收据充数。

同层两方法各从同一LAST完整状态出发：二层E725 raw/EMA/Adam34800/scheduler725，四层E650 raw/EMA/Adam31200/scheduler650；历史不重播，Adam矩与EMA不重置，旧best保留。一次LR/statistics重启二层E726 .0003、四层E651 .0001，全局epoch保留，后续段不再重启。原父MixUp/LS/TV/EMA/capture配置保留；方法分开不叠加。

- SAM半径.05是全相位vector L2扰动，与光残差rho.3不同；完整macro16/tail9 firstgrad扰动→同输入target重算secondgrad→从clone精确还原每参数→一次clip/Adam/EMA。记录两次反传、扰动范数、精确恢复、每轮48Adam，不按micro单独更新。参考[原论文](https://arxiv.org/abs/2010.01412)与[作者代码](https://github.com/google-research/sam)，此处不保证获益。
- Dropout明确是训练输入光振幅16x16空间块mask，二层p.02、四层p.05，确定性seed，不放大剩余振幅；验证/测试关闭mask，纯光推理图保持，不引入电子分类支路。并非电子分类器隐层dropout。

原十类/cache/split/train761/val157/test1833/角色ID/窗口/4专家/RGB100/8bit/光router不变。二层479364相位6张量、四层908552/11，三层已接受125选123差7.2013pp与全部无残差100冻结，不新训或再测。

| 本轮开始前选中模型 | 实际轮数/选中E | 训练/验证/总体测试 | test macro | 相对冻结100总体差 |
|---|---:|---:|---:|---:|
| 二层残差 | 725/E725 | 77.9238/75.1592/79.7054% | 75.0454% | +2.0185pp |
| 二层无残差 | 100/E98 | 67.9369/64.9682/77.6869% | 71.8949% | — |
| 四层残差 | 650/E488 | 99.6058/87.2611/93.3988% | 90.1104% | +9.4926pp |
| 四层无残差 | 100/E100 | 84.7569/78.3439/83.9062% | 80.6888% | — |

每25新轮只是保存/独占科学核验段，每段1800fit秒故障保护，无总epoch cap。生产原始rc0/actualexit，自己的source/config/全prefix/rawEMA/PT/shapeinit/6或11grad/Adam48E/orderaffineMix/所有LR重启replay/每macro SAM2passes或独立重建每maskSHA/savedID-label-CM-macroNLL/firstminimum science审计SHA与completed真匹配，才测试新selected。CPU科学0forward/test。各模型全prefix首次最低验证EMA balancedNLL选，不按最高test epoch取分。

继承二层19/四层12个旧参数digest，未变复用；新digest跨候选原子claim共享去重，每新参数最多一次test，冻结对照不重测。实际test二层>=82或四层总体gap>=10pp才达标，不四舍五入。否则最低LR2e-5且无.0002验证NLL显著改善75新轮平台暂停，失败保留停下游不retry/改阈值；只有这四候选，不自动扩组合/其他类或结构。

测试反馈用于停止，故明确额外残差预算、单seed17、验证/测试知情开发，不是独立泛化或同预算因果优势。test极不均与近重复未排除的局限保留，最终并列train/val/test/macro/实际及选中轮次/差距、全部逐类与测试次数，负结果不隐藏。原sources/PT/data/history/claims/失败/图保护，rawJSON/ID/PT/log私有，仅源码与聚合Git。保留t18和09 ACTIVE，普通进度静默，真正失败/最终状态通知一次。
