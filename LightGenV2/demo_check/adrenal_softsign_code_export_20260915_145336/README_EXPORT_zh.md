# AdrenalMNIST3D 光路由 MoE / D2NN 代码导出

本包直接导出自 connect.westc.seetacloud.com:17953 的
/root/autodl-tmp/Figure2b_adrenal_activation_search。
code/ 中源代码及配置保持服务器原始字节，已与正式训练锁定的源码 SHA256 核对。

## 内容
- code/：9 专家全激活 MoE、D2NN、2/4/6 层、有/无 OEO、训练、验证筛选、测试和绘图程序。
- code/config.yaml 与 configs/：50 轮，5 个配对种子 17/27/37/47/57，MSE 损失。
- code/optical_reference/activations.py：正式结果使用正半轴 Softsign，f(z)=ReLU(z)/(1+ReLU(z))。
- 同时保留 Softplus 和正半轴 Tanh 候选筛选代码；默认流程会按原规则重新筛选，不是强制只训练 Softsign。
- data/adrenalmnist3d/：原始 28³ 数据、28×28 平均投影训练数据及预处理记录；尚未切换 64³。
- data_tools/：原数据处理工具。
- environment/：原服务器依赖清单和环境记录。
- results/reports/：已有性能报告、均值/标准差 CSV、逐种子表、三种阈值策略的性能图。
- results/protocol/：已有训练、激活筛选和测试锁定记录，供复核。
- MANIFEST.json：逐文件 SHA256、原始服务器路径和导出信息。

本包不含已训练的 .pt 权重、虚拟环境、逐轮运行目录和缓存。不能直接载入历史权重做推理或断点续训；这些文件仍保存在服务器原项目 runs/ 下。
历史结果记录保存在 results/，不会被作为新训练的完成状态使用。

## 在当前 17953 服务器上重新训练
解压到新的独立目录，在解压后的 code/ 内执行：

```bash
bash run.sh
```

这是手动启动完整原流程的指令；本次导出未启动训练。
原 run.sh 使用 /root/autodl-tmp/Figure2b_adrenal_L4_5090/.venv/bin/python；
原配置读取 /root/autodl-tmp/Figure2b_adrenal_L4_5090/data/adrenalmnist3d/projected2d/axis2_mean/data.npz。
同一服务器上这两个依赖仍应存在。完整流程包括 12 组候选筛选和 60 组正式实验，复用其中 4 组，共 68 次独立训练。

## 迁移到其他机器
1. 配置支持目标 GPU 的 Python / PyTorch CUDA 环境；environment/ 记录的是原环境，不保证任意平台原样安装可用。
2. 将 code/config.yaml 中 experiment.data_npz 以及 code/configs/ 下全部 YAML 的 experiment.data_npz，统一改为解压后 data/adrenalmnist3d/projected2d/axis2_mean/data.npz 的实际绝对路径。保持各配置相同；无需改变 data_sha256。
3. 将 code/run.sh 中 PY 改为新机器的 Python 解释器路径。
4. 在干净的 code/ 目录运行 bash run.sh；不要把 results/protocol/ 的历史锁定文件复制到 code/protocol/。
5. 新训练会写入 code/runs/、code/protocol/、code/reports/。

此导出包是源码、数据和已完成结果的归档。仅进行了文件完整性及 Python 语法检查；未在新机器上重新训练验证。
