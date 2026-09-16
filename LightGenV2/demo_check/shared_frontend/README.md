# RGB/SAR：同一冻结电子前端，光学完成分类

低学习率续训及独立测试已完成：[结果与过拟合判断](../reports/reproduction/EUROSAT_CONTINUATION_HOLDOUT_20260916.md)。
MoE验证77.55%，但独立测试仅75.65%→75.70%；D2NN测试72.85%→73.30%。不将验证小涨解释为稳定泛化提升。

本轮已完成：[结果及复现证据](../reports/reproduction/EUROSAT_SHARED_FRONTEND_20260916.md)。
同一2000张验证图像：MoE77.20%、D2NN73.65%；两组20轮实际输入特征哈希全部相同。

结构：`56×56×3图像 → 同一冻结CNN → 128维特征 → 固定光场编码 → MoE或D2NN → 十类探测能量`。
电子仅训练一次；本次直接复用 `eurosat_frozen_cnn_20260916` 中预训练20轮、验证选中的第19轮权重。
两种光学模型使用同一检查点，去掉电子dropout和128→10分类层，不分别训练或微调电子。
所有前端参数、训练集通道均值/标准差及BatchNorm统计量冻结，输出端无电子残差、无alpha。

前端卷积通道32/64/128，均为3×3卷积、BatchNorm、ReLU、2×2最大池化；
空间56→28→14→7，全局平均池化得到非负128维特征。保留的冻结前端为93,472参数。
特征按通道原顺序排列成16行8列，每个数值复制为14×28像素单元，得到224×224实振幅。
逐样本归一化振幅平方和为1；此步骤会去除特征整体幅度尺度。没有可训练的投影、通道重排或分类头。
两组使用同一编码；每轮实际输入特征字节哈希、数据顺序和翻转种子都须一致。

MoE和D2NN共用 `pure_optical/models.py` 的完整画布传播实现及相位初始化。
为输入特征增加 `forward_amplitude` 接口；smoke对原图输入路径与历史代码逐位核验前向输出及相位梯度，
避免这一接口重构改变既有纯光学/输出融合试验。
MoE动态四路全部参与相干传播，路由探测能量份额转换为入口功率比例；
D2NN为无路由整孔径结构。主路功率都为1，MoE额外路由测量不代表全系统功耗与D2NN相等。
有效孔径478×478、传播画布518×518；波长532nm、像素17μm、两段传播各0.1m。
最终十个32×32探测窗口能量归一化为类别概率，直接最小化真实类别NLL。

沿用同一RGB/SAR空间配对子集：10类，每类每域训练300张、验证100张，共6000/2000张。
只在训练时做图像水平翻转，然后经过共享CNN，不对特征图翻转，无ColorJitter。
seed42，每组20轮，batch32，Adam lr0.01余弦降至0.001、weight decay0、梯度裁剪1。
前端不进入优化器，只有光学相位更新。按验证准确率最高、其次NLL最低、再次较早轮次选模。
不访问测试集；本轮结果属于单种子、验证集上的配对试验。

服务器从仓库根运行，沿用xml环境；顺序为smoke、train、evaluate，各指定独立输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/shared_frontend/run.py --phase train --data /DATA/DATA1/guest3/demo_reproduction_data/eurosat/phase_only_v1/data.npz --electronic-run LightGenV2/demo_check/runs/simulation/eurosat_frozen_cnn_20260916 --out LightGenV2/demo_check/runs/simulation/eurosat_shared_frontend_20260916
```

smoke将 `--phase` 改为 `smoke`，输出到 `runs/smoke/`。固定权重复评使用 `--phase evaluate`，
再加 `--run LightGenV2/demo_check/runs/simulation/eurosat_shared_frontend_20260916`，输出到新的smoke目录。
独立前端检查点保存于本次run的 `frontend/best_checkpoint.pt`；各光学模型只保存best/last。
直接推理时可用 `SharedFrontend(checkpoint['model'])` 加载这份去掉分类头后的前端；
复评命令还核对其与来源CNN的特征权重一致，再使用本次保存的前端进行预测。
前端来源、数据/源码/权重哈希、配置、命令、环境、逐轮记录和逐样本预测均保存在run中。
前端并未用光学验证性能重新筛选，两组也没有分别调整其特征编码。

## 低学习率续训与过拟合监测

`continue_training.py` + `continuation.json` 从上述run的第20轮last权重和完整Adam状态恢复，
电子仍使用同一冻结权重。两组最多续训至60轮，学习率在新增40轮从0.001余弦降至0.0001。
每轮对完整6000张训练集及2000张验证集进行无增强、同权重评估，记录准确率/NLL差距。
验证NLL改善不足0.001连续12轮时早停；两组使用相同规则，但实际完成轮数可能不同，须如实报告。
保留原最佳权重作为候选，新增候选NLL须不高于原最佳准确率权重的NLL，再按准确率、NLL选择。
本轮不添加额外正则项，先检验低学习率续训是否还有收益；不据验证选模宣称完全消除过拟合。
输出必须是独立run，不覆盖原20轮结果。

```bash
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/shared_frontend/continue_training.py --phase train --source LightGenV2/demo_check/runs/simulation/eurosat_shared_frontend_20260916 --data /DATA/DATA1/guest3/demo_reproduction_data/eurosat/phase_only_v1/data.npz --out LightGenV2/demo_check/runs/simulation/eurosat_shared_frontend_continuation_20260916
```

运行前用 `--phase smoke` 检查恢复的预测、优化器矩和step一致，并验证一步更新；输出到独立smoke目录。
结束后用 `--phase evaluate --run <续训run>` 重新加载最佳光学权重，核验完整训练/验证指标及逐样本验证概率。

续训结束后的一次性泛化检查：`prepare_holdout.py` 从原始空间划分的test部分，每类固定哈希排序取100个RGB/SAR配对，
得到2000张测试图像；与全部原始train/validation空间组均不重叠。预处理复用 `pure_optical.prepare.decode_pair`，
并先重算40张既有训练图像验证像素逐位一致。`evaluate_holdout.py` 先封存原始/续训两架构的验证选定权重哈希，
再读取测试数组进行一次性评估，测试结果不参与本轮参数、早停或模型选择。
