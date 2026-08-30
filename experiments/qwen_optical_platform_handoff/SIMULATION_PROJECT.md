# 纯仿真与训练工程

## 职责边界

本工程负责数据、Qwen 特征、电子/光学替代 block、任务头、loss、训练、development
选模、sealed test、鲁棒性仿真和硬件 payload 导出。它不加载真实 SLM，不打开 CCD，
也不持有某一实验台的四点坐标、曝光或 LUT。

## 新任务先选择正确参考

| 任务 | 首选参考 | 输出与主指标 |
|---|---|---|
| 商品/图文检索 | ABO、Grocery、Caltech 四层 | embedding；Top-1、Recall@K、mAP |
| 质量评价 | KADID10K MOS | 标量；SRCC、PLCC、RMSE、MAE |
| 显著性 | SALICON + `vision2_hybrid_dense` | 稠密图；CC、SIM、NSS、KLD |
| 分割 | ISIC2016 + `vision2_hybrid_dense` | mask；Dice、IoU |
| 姿态 | LSP + `vision2_hybrid_dense` | heatmap；PCK/PCKh |

不要从检索工程复制一个 64 维 readout 后硬改成质量回归；dataset、head、loss、metric
和 checkpoint selection 必须作为一组修改。

## 标准执行顺序

1. 复制最接近的参考实验为新目录，不直接修改参考工程。
2. 写 task contract，并先运行合同验证。
3. 只接电子头跑 smoke，确认数据、shape、loss 和 metric。
4. 加光学仿真，跑 1 个 batch 的 forward/backward 和 finite-gradient 检查。
5. 小规模 smoke；随后正式训练。
6. checkpoint 只能由 development 指标选择；test 最终一次。
7. 导出 mask、每阶段 compact amplitude、样本 manifest、checkpoint SHA 和几何合同。
8. 把硬件 payload 交给硬件工程；不要把服务器 cache 当作采集输入合同。

## 仿真必须记录

- Qwen 模型 revision、冻结范围和输入预处理。
- 数据 split 的样本 ID，而不只是数量。
- wavelength、distance、logical pitch、active size、专家布局与 phase 翻转。
- phase/electronic/router 学习率和实际参与训练的参数表。
- pixel shift、k-space、零级分量、CCD 噪声与强度归一化。
- 最佳 checkpoint 的 development 指标及 sealed-test 未参与选模的证据。
- 每层导出的 amplitude/phase manifest 与 SHA256。
