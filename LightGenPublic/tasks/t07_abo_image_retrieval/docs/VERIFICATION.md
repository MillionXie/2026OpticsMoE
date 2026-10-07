# 独立性与固定权重复评

整理版保持前端、光传播、残差和读出数值核心及权重键名；去掉对原任务/历史实验目录的运行时依赖。训练重新组织为一个 high-alpha retrieval 续训流程，不包含原大规模预训练素材或其他试错profile。

本地和服务器各4项独立合同测试全部通过；覆盖无跨工程导入、同尺度融合、训练图库排除自身商品、实际相位反向梯度与模块审计。服务器隔离目录复评完成，禁用PYTHONPATH回退，代码从独立包目录导入，Hugging Face离线模式，没有加载完整大模型。

核验目标：同一个 `high_alpha_retrieval_20260910/artifacts/best.pt`，相同processor、2400图数据manifest、评估批量4；比较1440条train和480条test的64维特征逐项，以及正常/同权重去光 Hit@1。测试通过不等于从头训练轨迹相同。

| 核验 | 实测 |
|---|---|
| train 1440×64特征 | 逐位相同，最大绝对差0 |
| test 480×64特征 | 逐位相同，最大绝对差0 |
| 正常Hit@1 | 0.6895833333333333 |
| 同权重去光Hit@1 | 0.6458333333333334 |
| TF / attention模块 | 0 / 0 |
| 推理码来源 | 29e16bed0392dd87cd2aa9bec3ec2bb4cd89edfb |
| 环境 | Python3.11.15、torch2.6.0+cu124、单张4090 |
| 权重SHA256 | 814893fb430a73ce529acd7a1a80e264d7ed0253a37658df936cc40e34d6265b |
| 数据SHA256 | 2949a4035150a9f8718f2a6cace164c17394613d24fb9d0234c553bee8d77c97 |

证据在同任务 `runs/fixed_equivalence_20260911/`，交付包的 `reference/fixed_equivalence/` 保留报告、执行记录、equivalence.json及CCD审计CSV。复评PID2401384已正常退出。后续整理只补说明/绘图/打包，不修改此次验收的推理数学。

独立包另完成 **1epoch×1step、训练batch40的真实联合续训冒烟**，包括全量初始/EMA/live评估、保存best/last、恢复best后正常与去光评估。12片相位（含两个router）均记录非零参数变化，损失有限；此测试只验证训练链路可用，不作为新的性能提升。证据 `runs/continuation_smoke_20260911/`。PID2409672已退出，GPU显存释放；最终交付仍是原30epoch选中的68.9583%权重，不用冒烟权重替换。
