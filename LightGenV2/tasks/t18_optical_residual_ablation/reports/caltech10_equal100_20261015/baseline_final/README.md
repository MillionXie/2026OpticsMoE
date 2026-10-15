# 新双方100轮基线已实际完成并独占科学/冻结测试通过

这次六臂均真正100轮、maximum_epochs，无early，各4800Adam，CPU campaign2419295和所有GPU实际退出rc0。每深度自己的首次独占保存科学及每模型一次冻结测试的独立保存预测审核均通过。不是旧150改名。二层选E98不代表98停止；100已跑完，按100内首次最低验证EMA balancedNLL选E98。

| 深度 | 有/无实际轮数 | 有/无选中E | 有残差训练/验证/测试 | 无残差训练/验证/测试 | 总体test差 | macro差 | 目标 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 100/100 | 100/100 | 96.5834/85.9873/92.1440% | 84.7569/78.3439/83.9062% | +8.2379pp | +5.7975pp | 10pp |
| 3 | 100/100 | 100/100 | 89.4875/80.8917/87.1249% | 79.8949/75.1592/80.2510% | +6.8740pp | +6.5242pp | 6pp |
| 2 | 100/100 | 98/98 | 73.1932/71.3376/78.7234% | 67.9369/64.9682/77.6869% | +1.0366pp | +0.1850pp | 4pp |

所有百分比都对应选中同一模型，训练不是MixUp训练batch准确率，而是选中EMA对原训练集的评估。测试1833 IDs相同、无电子分类器。固定十类验证知情开发、旧test已观察；这不是未见独立泛化。每模型test1，选定后才各测一次，原始预测/ID/PT/完整日志只私有。

## 来源与独占收据

181源码首次发布e87caff8b529ef0639fc6b557739247b23270e01；本run metadata实际Git d61142f91605bd33fbbd5180541c4d488aeeb5ac，逐项源SHA与该发布闭包相同。共享main可能随其他任务推进，不能把首次发布SHA或报告HEAD硬当实际训练HEAD。代码、config、数据角色、源SHA在独立科学中逐项核验。

- L4 audit SHA 37c21452e98698abcbda8fd0367147cc838b2b03e022aa3101e27b8a42b89a9c，completed匹配、原始auditrc0。
- L3 audit SHA a2d6a7f0bd651e1e9bd4e2c1490c61937c089d41bf05bfb4035d423146d4f799，completed匹配、原始auditrc0。
- L2 audit SHA cd12dcc270b463b100062db644229bf2a2a9559403e4edddc9b3d833289d22ff，completed匹配、原始auditrc0。

| 模型 | selected PT SHA | test result SHA | saved test audit SHA |
|---|---|---|---|
| L4_rho03 | 53a6ef9ee2b589525292dcd3817870d3124b6e24d0499d15f43c07f718df6b68 | 26d8d630fb3d049b277c606e4d110bee221cd9286935d19dfdc148142d4a0df7 | 1fe4448bd6ffebe7a6d2ffaea7e968c7e724adba857b4ca5088d40466938eba7 |
| L4_rho0 | 693226bbac3fcdc166741a8df18f1005408e08314e384f6bdf7db8ad01ca6f36 | 646837daaa51bab3c2868cae5b59e0cfe603c753bb14ad46b6b907648c58cc56 | e58f8e16a3617d4cf50a9d884ffe060eb9b706e659d861c57410902aa930e92c |
| L3_rho03 | db2562e4ee1319921a19be993e3528a0f9739688639993036517aa21bc8eaeb6 | 776c6b4a055d1f99be9ebdf043610ff495a2780a2327d1b4b31582b6f2f57fa0 | d4412ad96c8441688c832432eda369e52c1ef8eff462c9f0ddec8bea1e6e2f18 |
| L3_rho0 | 62154e9067aa75494189955421a0d32e74cbe8ff14b4f3caf929262ed11445cc | 47d0402855eccb78aae54410016edfeb3f06f5e89dba4ee9149c4449db2d64af | f0ff9813d12abec229656fb8a11672dec1f6991ccee187922302d0816f3e1956 |
| L2_rho03 | 066b26ccba15a4c5f0fe6ab4a2559b7be4001a730cd57a979c921112d2ded0ab | ecfe6a1547c49a0353fd644c989026e870e64d5f28c606c9fa22fd6cc6437c75 | 3081d45ad0d6b2e7548672d903edc094243d06a3968fca082c9401c2754a9a58 |
| L2_rho0 | 5d0c8c2bee0e48d9d9297d31c68ce2163347e031af764e2c7c5e6d6afcded2d3 | 107432114bcee0dc6d9ebf475f1151b8ba173222a32916cc507aa9beac24cd49 | d2ecc7aca06420315408b3eddcb0ae0f07e4c213f84eafef7b9843e94a50dab5 |

## 接下来只训练残差

CPU控制器2446199已自动放行新caltech10_equal100_residual_open_20261015，185源b03fcdfa941669e53b6f67b4db8193f00972db85。从各残差LAST E100完整raw/EMA/Adam4800/scheduler恢复。无残差控制从现在冻结，不再训练或重复测试。残差MixUp.3中度正则假设，每25轮保存与自己的独立科学后继续，不设总epoch上限；目标二/三/四层总体测试差4/6/10pp。达到该深度目标则停止；验证停滞75新轮且LR最低时暂停并如实报告，生产/科学/测试失败停下游不重试。详情见新协议CALTECH10_RESIDUAL_OPEN_PROTOCOL_20261015.md。

每新selected参数digest最多测试一次，相同参数复用原结果；继续用验证firstminimum选模型，但测试反馈影响停止，因此接续整体属于test-informed development，不冒称独立泛化。额外残差训练与冻结100对照不是同预算因果消融。三层在100基线已超过6pp目标；二/四层未达到4/10pp，不能把目标说成已成功。

关于旧test显著高val：类别权重的量化解释、精确重复核查和未解决局限见[纠正与数据诊断](../correction/README.md)。macro与全逐类同时保留，不隐瞒低召回或负差类别。固定十类不换，不削弱对照、不挑test最大轮次扩大差距。旧150/所有数据/PT/来源/失败/claims/图保留，不重入旧阶段。
