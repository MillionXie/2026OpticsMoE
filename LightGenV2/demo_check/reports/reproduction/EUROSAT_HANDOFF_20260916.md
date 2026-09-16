# 共享冻结前端版本交付验证

按用户指定，交付原20轮实验 `eurosat_shared_frontend_20260916`，训练源码 `8cb125cfaa4829e7714b5b23389f0c96e4844a72`；独立打包源码 `640355f3`。不替换为后续优化权重。

| 模型 | RGB验证准确率 | SAR验证准确率 | 综合验证准确率 | 最佳轮次 |
|---|---:|---:|---:|---:|
| 共享冻结前端＋动态四支路MoE | 79.50% | 74.90% | 77.20% | 20 |
| 共享冻结前端＋D2NN | 72.60% | 74.70% | 73.65% | 15 |

交付文件：`releases/eurosat_shared_frontend_moe_d2nn_7720_7365_20260916.zip`，72,925,740字节。
SHA256：`e60d09f36e787079a2013b82f70d60ff1c3be2dfb89fc2a045ccd36a58e3cbe9`。
由任务 `build_lab_package.py --variant shared_frontend` 生成；包内MANIFEST记录每个文件的SHA256，PROVENANCE记录数据、源码及权重身份。

包含可独立运行的模型与训练/复评入口、配置、同一冻结CNN、两组最佳权重及第20轮last权重和Adam状态、6,000张训练与2,000张验证图像、划分清单、原训练记录和中文README。只训练光学相位，电子前端与BatchNorm统计固定，无输出电子残差。数据处理、标签、选模及环境详见包内README。

在服务器独立目录解压，使用Python 3.11.15、PyTorch 2.6.0+cu124、NumPy 1.26.4、RTX 4090单卡执行：

```bash
python verify_package.py
CUDA_VISIBLE_DEVICES=0 python run.py --mode smoke --out runs/smoke
CUDA_VISIBLE_DEVICES=0 python run.py --mode evaluate --out runs/evaluate
CUDA_VISIBLE_DEVICES=0 python run.py --mode train --resume --epochs 21 --learning-rate 0.001 --out runs/resume_check
```

42个文件校验通过；两组光学相位梯度有效，电子前端冻结；Adam成功恢复第3760步。完整验证集指标及逐样本概率与原记录逐位一致。各续训一轮成功，此检查输出未纳入交付权重。本次属于固定权重复评及续训检查，没有重新训练完整20轮。

验证记录归档至 `runs/smoke/eurosat_shared_frontend_handoff_20260916/`。已知原权重空间测试结果（MoE 75.65%、D2NN 72.85%）另存包内reference，不与上表验证指标混用；该测试集已查看，不作为今后调参的未触碰测试集。
