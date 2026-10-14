# 一次MixUp.2纯光优化完成：科学和四图QA通过，残差78.14%未达80

run resisc10_s17_mainL4_input100_rho_pair120_mixup02_micro8_gpu25_20261015，165冻结运行来源6c2309ba048f2fa12f13390ca6fbc62954dc2c74。父767046及两子768304/768385实际退出rc0，fail0/test0；GPU2/5项目释放，A100禁止。固定十类按此前20验证表现选，仍为validation-informed开发任务，不声称原20验收或独立泛化。

| 组 | 实际轮 | firstminNLL选轮 | train | val/macro | balancedNLL | gap(pp) | stop_reason |
|---|---:|---:|---:|---:|---:|---:|---|
| rho.3 | 92 | 92 | 86.8036% | 78.1429% | .789308841 | 8.6607 | validation_plateau |
| rho0 | 120 | 113 | 84.8571% | 77.1429% | .820685700 | 7.7143 | maximum_epochs |

firstminimum验证EMA balancedNLL选，不能按最高accuracy替换。两组严格val65/macro60/每类20/无zero/末20zero/gap15/all4/inputvariation准入均passed，但残差未达80。比原MixUp.6残差所选75.8571%提升2.2857pp，仅单seed和不同停止/选模轮次的开发观察，不宣布唯一原因或保证跨种子。共同E92 val78.1429/75.5714%，也仅本单seed配对描述。未test、未启动正式6/2或3层或现场校准。

仅MixUp.6->.2改变，原908552phase/10类/cache/split/input100/光图/8bit/loss/LS/LR/EMA/init/更新设置保持。真实独占保存科学SHA `ddf43abb366dc0f26a2dec9b0102c48dbc2cb81ced3df0d1a5002e098ab9d346`，2CPU模型核shape/seed、0forward/test，所有165currentoriginal/data/config/PT/ID-label-argmax/CM/support/macroNLL/firstmin以及每轮order/affineMix/tail16/LR/early/time独立核。350Adam/E，实际32200/42000，sameinit与共同92orderANDaug通过。

rho.3最后history拟合2884.72850/含端点总2894.97761秒，按既有显著NLL early规则停止；rho0分别7273.67953/7297.92918秒，实际stop_reason maximum_epochs。每epoch开始检查7200秒，最后已开始epoch和必要端点评估完成，不称硬秒杀，也不因未80扩预算或恢复已退出组。

残差弱类bridge train77.50/val62.86、wetland68.39/62.86、tennis81.25/65.71、thermal80.18/68.57；intersection从原68.57提高到82.86，bridge和wetland仍偏弱，不重新换类或提出唯一因果。按最新人工条件许可，科学/四页实际QA完成后立即一次490参数CPU小电后备，冻结本rho.3所选E92光EMA/checkpoint SHA97a2d851cfde3f1d166961224109d6ec6b34af8852570f587b088c5f69aec53b。固定光alpha.8、电.2概率融合，与rho.3相干场70/30不同；明确hybrid非纯光，原train/val保存光概率与对应RGB，test0，不新增GPU/A100或额外候选。

四PDF/PNG/SVG/聚合CSV已经一次真实作者、成功count4 marker、manifestSHA传输、Poppler120dpi和全部四页actualview_image QApassed。[实际视检](visual_review.md)。机器审核/manifest/transfer/visual JSON仅本地服务器；所有旧源码/PT/run/history/失败/claims/marker保护，不重审/重图。08:50统一报告各阶段实际成绩或未完成状态，03:30取消，09晨报保留。
