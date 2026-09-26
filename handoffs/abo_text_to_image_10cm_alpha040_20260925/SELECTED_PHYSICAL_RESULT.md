# 文搜图实测采用版本

2026-09-26用户确认：采用finetune10/best_readout_checkpoint.pt作为后续实测结果，不改用其他轮数的更高成绩。

- 原始光学/电子主体checkpoint SHA256：cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae。
- 最终电子读出头：10 epoch从原权重训练，验证集选第9轮；600 TRAIN适配/200 TRAIN验证。
- 最佳权重SHA256：89e25360c8f12c9d13dbceba49e8d81073d4dd75c6acf667ab12a38b8fd293c5。
- 实测TEST Hit@1=0.85，Hit@5=0.97，Hit@10=0.97，MRR=0.8952448964118958，MAP=0.7978265285491943。
- 原始浮点结果、训练记录、best和last均保存在finetune10目录；报告口径与限制见该目录README.md。

这是电子读出头适配后的实测结果，不是未微调光学模型的结果。部署需加载原始主体并替换上述读出头。
