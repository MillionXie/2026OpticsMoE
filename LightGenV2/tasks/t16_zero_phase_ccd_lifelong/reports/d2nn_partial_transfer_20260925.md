# D2NN 冻结迁移矩阵：EuroSAT／Speech 已确定的两项

这是正式四任务 4×4 矩阵的**已确定协议子块**，不是完整矩阵，也不是顺序无回放矩阵。行模型各自单任务训练；跨列直接用该模型的原光学相位和原 `Linear(784,10)` 在目标测试集推理，完全不更新参数、不换任务头、不遮掉非目标输出。评估需要目标数据集的真值标签，但推理模型不使用任务 ID。数字为完整测试集的宏平均召回，括号内是样本量。

| 固定 D2NN 来源／目标 | EuroSAT RGB＋SAR | Speech 语音＋词 |
| --- | ---: | ---: |
| EuroSAT | **57.59%** (5,429) | **0.00%** (1,734) |
| Speech | **10.21%** (5,429) | **67.19%** (1,734) |

两条对角线与各自已记录的单任务测试结果逐值一致。Speech 是二分类，却与 EuroSAT 共用 10 输出位置；EuroSAT 头在 Speech 上的预测可以落在 2–9 类，因而出现 0% 宏平均召回。这个数只说明**该固定完整光电模型**不能直接迁移到该目标，不能单独归因于光学相位，也不能推断目标任务本身难度。后续若 CLEVR／Physical 定义或输入编码改变，必须先独立训练其 D2NN，再扩成同协议四任务矩阵；不得把旧版模型混入。顺序 D2NN 的 A→B→C→D 下三角仍须等 B 任务确定后按次序训练。

原始证据：服务器 `runs/simulation/d2nn_frozen_eurosat_speech_2x2_d432/` 的 `result.json`、`config.json`、`command.txt`、`status.json`。评估源码 commit `d432cce3cf7d290be82a152ce2dc6c531c451fc4`，Python 3.11.15、PyTorch 2.6.0+cu124、CUDA 12.4。协议 SHA256：EuroSAT `8ffff7cf42719bc454ccca3f568c18dba1f056eeb7ada7f51520e749e21ffb48`，Speech `f97cb20c365729e9424c65686e3db927d65246b645de8053557eac2a222f3489`。权重 SHA256：EuroSAT `3c335b47c060cef52a21a7544f6a8160df8d7426c8f82e47402e1dc05dd1006c`，Speech `3c8e40485ac14c806706c903c1eb5d083dcf9d709ffb7b1efd47d410da89baf2`。
