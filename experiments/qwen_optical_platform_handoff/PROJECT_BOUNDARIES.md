# 两个工程与运行资产的边界

## 1. 仿真训练工程

包含完整的可复用训练源码、不同任务的参考实现、光学传播、鲁棒性扰动、评估与硬件导出代码。
它应在 GPU 服务器上完成：

1. 冻结数据划分并建立 task contract；
2. 训练电子 baseline 和光电模型；
3. 只按 development 指标选择 checkpoint；
4. 在 sealed test 上最终评估一次；
5. 导出任务 checkpoint、四层 payload、phase mask、manifest、SHA 和仿真参考 CCD。

该工程不携带数据集、模型权重、训练结果和缓存。这些体积大且与任务有关，必须放在服务器
统一存储，并在实验记录中写出绝对路径、revision 和 SHA。

## 2. 硬件实验工程

包含当前 Meadowlark/TUCam 驱动、设备标定、SLM 重建、逐层采集、一致性评估、MNIST4
方向诊断和 measured-CCD 本地微调代码。它不负责从零训练一个新任务。

新任务要真正运行，必须从仿真工程接收一份 `task_payload`：

- 初始 checkpoint 及 SHA；
- Qwen 模型 revision 或完整离线 snapshot；
- 冻结的数据 split 和样本键；
- 每层 compact amplitude、phase BMP、manifest 和仿真 detector 输出；
- 任务 head、loss、metric、development 选模规则；
- 光学几何与 detector normalization 合同。

硬件工程只保存当前平台重新测得的 LUT、曝光、四点 homography、时序、CCD 和微调结果。
这些平台状态不能回传为新任务的默认值，也不能提交到通用模板。

## 3. 当前 Caltech101 实例

`qwen_mnist4_early_robust_full_data_lab` 同时携带了 Caltech101 的 task payload、离线模型和
本实验平台流程，因此它能独立完成当前实验。它适合作为“跑通范例”，不应直接改名后用于
商品检索或质量评价；这样会残留 Caltech 类别、210/全量 profile、retrieval loss 和旧阶段键。

## 4. 新任务最小交付物

只有同时具备以下三项，才可称为可复现实验：

1. 代码：两份工程对应的 Git commit；
2. 任务资产：数据 split、模型 revision、checkpoint、payload manifest 和全部 SHA；
3. 平台资产：本机 LUT/曝光/homography/时序及生成报告。

ZIP 中的 `PACKAGE_CONTENTS.json` 只证明代码包内容完整，不能替代任务资产和平台资产。
