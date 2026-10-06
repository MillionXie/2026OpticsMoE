# RGB/SAR：共享冻结电子分支，等权概率融合

实测完成：[训练、结果及证据](../reports/reproduction/EUROSAT_FROZEN_ELECTRONIC_20260916.md)。
同一2000张验证图像：电子76.05%，MoE融合75.85%，D2NN融合75.90%；本次0.5融合未带来净收益。

本协议在 `pure_optical` 的动态四支路 MoE、整孔径 D2NN 上增加同一电子分类支路。
不使用旧版 Qwen 特征适配器；本次电子模型是从头训练的小型 CNN。

输入沿用原空间划分的配对子集：10 类、每类每域训练300张和验证100张，合计6000/2000张。
两域分别是 RGB 和 VV/VH 固定编码的 SAR 三通道图像，输入均为56×56×3。
只在训练时水平翻转，两条分支读取同一翻转结果；不使用色相增强，不读取测试集。
数据准备仍由 `../pure_optical/prepare.py` 完成。

电子：训练集两域合并计算逐通道均值和标准差；三个3×3卷积模块分别输出32/64/128通道，
各接 BatchNorm、ReLU、2×2最大池化，尺寸依次28/14/7；全局平均池化后得到128维向量，
接 dropout=0.2 和128→10线性层。AdamW，初始学习率0.001、weight decay=0.0001，
余弦降至0.0001，batch32，梯度裁剪1，训练20轮。仅按验证准确率最高、其次NLL最低选模。

选定一次电子权重后完整冻结，包括所有参数、输入归一化常数和BatchNorm统计量；
关闭Dropout，两种光学架构共用这一份模型。电子参数不进入光学优化器。
光学路径、相位初始化、几何、固定三通道拼接编码均直接调用 `../pure_optical/models.py`。
两种架构各训练20轮，Adam初始学习率0.01，余弦降至0.001，weight decay=0，
batch32，梯度裁剪1；同一seed42、数据顺序、逐轮水平翻转。MoE四路均参与相干传播，
光学路由的能量份额作为入口功率比例；D2NN是无路由整孔径版本。

电子概率 `p_e=softmax(logits)`；光学概率 `p_o=E_k/sum(E)`（含1e-12数值稳定项）。
融合概率 `p=0.5*p_e+0.5*p_o`，系数固定、不可学习。光学训练最小化真实类别的
`-log(p_y)`，按融合验证准确率最高、其次融合NLL最低选光学权重。
同时保存电子单独、光学单独及融合的逐样本概率与整体/RGB/SAR指标。
光学单独指标对应融合损失训练得到的光学权重，不等同于重新训练的纯光学基线。

运行（仓库根目录，服务器Python环境沿用xml；先smoke后train，输出目录不可复用）：

```bash
CUDA_VISIBLE_DEVICES=0 python LightGenV2/demo_check/frozen_electronic/run.py --phase smoke --out LightGenV2/demo_check/runs/smoke/eurosat_frozen_cnn_20260916
CUDA_VISIBLE_DEVICES=0 python LightGenV2/demo_check/frozen_electronic/run.py --phase train --data /DATA/DATA1/guest3/demo_reproduction_data/eurosat/phase_only_v1/data.npz --out LightGenV2/demo_check/runs/simulation/eurosat_frozen_cnn_20260916
```

每阶段只保存best/last检查点。光学检查点仅含光学状态，复评时还须加载同一run的
`electronic/best_checkpoint.pt`。配置、源代码commit/哈希、环境、数据哈希、电子冻结哈希、
逐轮记录和完整命令保存在run中。这里的0.5指归一化输出概率的权重，不代表系统物理光电功耗各半。
