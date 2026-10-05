# RGB/SAR：共享冻结电子前端＋MoE / D2NN

本包固定为原20轮光学训练版本，验证结果如下；后续优化请新建输出目录。

| 模型 | RGB准确率 | SAR准确率 | 综合准确率 | 最佳光学epoch |
|---|---:|---:|---:|---:|
| 共享前端＋动态四支路MoE | 79.50% | 74.90% | 77.20% | 20 |
| 共享前端＋整孔径D2NN | 72.60% | 74.70% | 73.65% | 15 |

## 包含内容

- `models.py`、`cnn.py`、`optical_model.py`、`optics.py`：模型和传播实现。
- `run.py`、`training.py`、`utils.py`：复评、梯度检查、重新训练光学、恢复Adam续训。
- `config.json`、`optical_config.json`：原训练和光学几何配置。
- `checkpoints/frontend/best_checkpoint.pt`：两模型共用的冻结前端。
- `checkpoints/{dynamic_four,full_d2nn}/best_checkpoint.pt`：上表对应的最佳权重。
- 两架构的 `last_checkpoint.pt`：均为第20轮，含Adam动量和step。D2NN的best和last不同。
- `data/data.npz`：实际使用的6000张训练、2000张验证图像，已完成预处理。
- `data/manifest.json`、`data/classes.json`：样本身份、空间组、标签映射与像素哈希。
- `reference/`：逐轮指标、逐样本验证预测、原运行元数据、已有核验记录和曲线。
- `PROVENANCE.json`、`MANIFEST.json`：训练/导出commit、文件及权重SHA256。

无需原仓库、Git、Qwen、原始遥感ZIP或其他工程目录即可运行。使用CUDA GPU。
原环境：Python3.11、PyTorch2.6.0+cu124、NumPy1.26.4、RTX4090。

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python verify_package.py
python run.py --mode smoke --out runs/smoke_001
python run.py --mode evaluate --out runs/evaluation_001
```

复评会检查全部2000张验证图像的概率是否与原导出逐位一致。更换硬件/依赖后若出现数值差异，
应先记录环境和差异，不将程序的严格一致性断言当成模型无法推理。

## 结构与公平性

图像56×56×3 → 同一冻结CNN → 非负128维特征 → 固定光场编码 → MoE或D2NN → 十类探测能量。
前端：三个Conv3×3-BatchNorm-ReLU-MaxPool模块，通道32/64/128，空间28/14/7，之后全局平均池化。
已去掉电子dropout和128→10分类头；93,472个前端参数、BN统计量和训练集通道归一化常数均冻结。
前端曾以电子分类任务训练20轮、选中第19轮：AdamW lr0.001余弦降至0.0001，weight decay0.0001，batch32，dropout0.2。
本包直接加载该已训练前端；比较两种光学结构时不要分别训练前端或放开其BN更新。

特征通道按原顺序排成16×8，各元素复制为14×28单元，形成224×224实振幅，逐样本总功率归一化为1。
无额外可训练投影、电子输出旁路或alpha。全局池化丢弃空间布局，功率归一化去掉整体幅度尺度。

MoE四支路全部参与完整光场的相干传播。路由探测能量归一化为入口功率比例，振幅按比例平方根加载。
D2NN为无路由整孔径版本。两者有效孔径478×478、传播画布518×518，波长532nm，像素17μm，两段距离各0.1m。
最终十个32×32窗口能量归一化为类别概率。MoE/D2NN可训练相位参数分别479,364/456,968。
MoE额外具有路由测量开销；共享电子相同不代表两架构参数量和全系统功耗完全相同。

## 数据与原训练

10类×两域，每类每域训练300张、验证100张。域0为RGB，域1为SAR。
RGB为64×64图像中心裁剪56×56；SAR VV/VH重投影至同一裁剪区域，逐图1%/99%分位截断，
使用固定均值[-12.59,-20.26]、标准差[5.26,5.91]映射到[0,1]，第三通道为前两通道均值，量化为uint8。
训练/验证来自原空间分组划分，同位置RGB/SAR保持配对，空间组无交集。
仅训练时图像水平翻转，然后输入CNN；没有ColorJitter，也不对特征网格再翻转。

原光学训练：seed42，20轮，batch32，Adam lr0.01余弦降至0.001，weight decay0，梯度裁剪1。
只优化光学相位，损失为真实类别探测概率的负对数。
按验证准确率最高、其次NLL最低、再次较早epoch选模。每轮记录电子冻结校验和实际特征哈希。

重新初始化光学、保持前端不变，重跑原20轮：

```bash
python run.py --mode train --out runs/retrain_001
```

从两模型各自第20轮last恢复，包括Adam动量，示例续训至总30轮：

```bash
python run.py --mode train --resume --epochs 30 --learning-rate 0.001 --out runs/experiment_001
```

该命令是供修改的训练入口，不是上表20轮结果本身，也不自动提供早停。
恢复时重新建立新增轮数的余弦学习率计划；每组只保留best/last，新输出不覆盖参考权重。
`config.json`可调整训练参数；改变模型/特征编码时，保证两组共用同一电子权重、数据划分和处理。

## 结果口径

上表为2000张验证图像的选模结果。附带 `reference/known_test_results.json` 记录同一原始权重后来做过的
独立空间测试子集结果：MoE75.65%、D2NN72.85%，各2000张；测试数据不随本包提供。
这些历史验证/测试信息已被查看，不应把它们重新包装成未来优化方案从未接触的最终测试证据。
这是供继续优化的单种子基线包，不包含后续续训模型。数据来源与许可见 `licenses/` 和 `data/DATA_LICENSES.json`。
