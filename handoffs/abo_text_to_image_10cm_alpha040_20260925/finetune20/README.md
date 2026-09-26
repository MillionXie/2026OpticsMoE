# 20 epoch 微调

从原始 checkpoint 重新训练 20 epoch，非续训。沿用固定 seed=20260926、600 张 TRAIN 适配/200 张 TRAIN 验证，训练超参数和共享电子读出头结构与 50 epoch 运行一致。仅按验证 Hit@1（同分 MRR/MAP）选择权重；选定后加载原 2400 张 TEST/100 个标题的实测冻结特征缓存，FP32 读出头测试一次，未额外测试最后权重或 BF16 前向。

- 验证最佳：epoch 18，Hit@1=0.90。
- TEST：Hit@1=0.91、Hit@5=0.98、Hit@10=0.99、MRR=0.9430115222930908、MAP=0.8242304921150208。
- 最佳文件：`best_readout_checkpoint.pt`；SHA256 `7c973c1f92dcd99649c7badecdd10740484ea1579ef1fa498c8d74ab83266a0d`，本地下载已核验。
- `history.csv` 记录训练/验证每轮指标，`finetune_report.json` 记录选定后的单次 TEST。

本次 TEST 只测试一次，不代表该 TEST 在项目历史上只被使用过一次。不同 epoch 预算结果供比较；不能在反复查看 TEST 后选择训练预算，仍将它描述为完全未接触的独立测试。前次异常高提升的完整审计尚未完成。
