# 10 epoch 微调

从原始 checkpoint 重新训练10 epoch，不是从20/50 epoch权重续训。固定600张实拍TRAIN适配、200张实拍TRAIN验证，按验证Hit@1（同分MRR/MAP）选择权重，选定后在2400张TEST图库/100条标题上测试一次。使用实测冻结特征缓存上的FP32电子读出头，无需重采光场。

仅训练共享的 retrieval_readout，图像与标题都经过该头：LayerNorm(384) → Linear(384,64) → L2归一化。参数共25,408：LayerNorm权重/偏置768，Linear权重24,576、偏置64。FP32参数约99.25 KiB，不含优化器。其余电子网络、相位、router和融合参数不变。

- 验证最佳epoch 9，Hit@1=0.87。
- TEST：Hit@1=0.85，Hit@5=0.97，Hit@10=0.97，MRR=0.8952448964118958，MAP=0.7978265285491943。
- best_readout_checkpoint.pt为部署候选，last_readout_checkpoint.pt为第10轮权重；没有另测最后权重。
- 最佳权重SHA256：89e25360c8f12c9d13dbceba49e8d81073d4dd75c6acf667ab12a38b8fd293c5，本地下载已核验。
- history.csv记录每轮训练/验证指标；finetune_report.json记录最终测试结果。

本次仅测试一次，但项目此前已在同一TEST比较其他epoch预算，不能描述为历史上完全未接触的独立测试。此前异常高提升的完整审计尚未完成。
