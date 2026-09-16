# Kather2016：逐层OEO与MoE的配对验证

## 数据与任务

选择原Kather2016组织纹理八分类数据，150×150 RGB、5000张、每类625张。类别顺序固定为肿瘤上皮、简单间质、复杂间质、免疫细胞、碎屑与黏液、黏膜腺体、脂肪组织、背景。任务是组织图块分类，不是患者癌症诊断。

原作者在[论文Data usage statement](https://www.nature.com/articles/srep27988)明确以CC BY 4.0发布全部原始数据；原始记录为[Zenodo 53169](https://zenodo.org/records/53169)。这不是MNIST系列数据，也不是原PathMNIST的100K数据改名。

因Zenodo文件入口在本地和服务器均出现HTTP504，本次采用[1aurent/Kather-texture-2016镜像](https://huggingface.co/datasets/1aurent/Kather-texture-2016)，固定commit `4fd718bff27676d01e291057824449d6d1f569d0`，Parquet SHA256 `47756bfdb12c48870e8738dd601c7dfe8b116050399e69f4fe8b7bf082387f44`。镜像重新封装了图像，未完成与原ZIP逐像素等价核验；这一限制和镜像行ID保留在数据manifest，不能声称已核验原ZIP。镜像标签按其schema显式映射至上述顺序，绝不直接套用数字标签。

解码检查全部RGB尺寸、类别计数和精确重复。实测5000张均唯一。用固定RNG20260917在每类内置乱并分配437/94/94张到训练/验证/测试，合计3496/752/752，所有类别均保留，划分不随模型种子变化。每张图以镜像原行ID追溯，manifest保存三个完整清单及缓存SHA256。

公开数据来自少量组织样本，镜像未提供可靠患者/切片映射；不从随机文件名臆造患者ID。这里明确是图像级划分，不能据此声称患者独立泛化或临床有效性。精确图像不重复也不能排除同一切片的相关性。

## 模型和输入

直接从150×150原图进入RGB缩放流程，不先降成28×28。共同流程：除255→双三次抗混叠100×100并截断[0,1]→训练时共享仿射增强→双线性50×50→`[R,G;B,mean_RGB]`拼为100×100振幅。

四组为MoE、MoE+OEO、上采样D2NN、上采样D2NN+OEO；2/4/6层，seed17/27/37。MoE保留九专家动态路由和理想无损九端口，D2NN采用覆盖相位面的474/472/470输入，分块插值并恢复逐样本原总光功率；完整公平性及照明核查见[BLOODMNIST_MULTISEED_PROTOCOL.md](BLOODMNIST_MULTISEED_PROTOCOL.md)。参数量、探测窗口与传播几何沿用Blood实验。不增加电子特征提取器或分类残差。

有OEO组在每个主干相位层传播后读取光强，在498×498有效面做无可训练仿射项的归一化，施加ReLU后Softsign，作为零相位振幅继续传播；两架构使用相同函数。关闭组同时关闭专家和全局OEO，保留复光场传播、最终光强探测和MoE路由。主干深度不包含MoE额外的路由相位，但总参数计入路由。

## 公平优化和训练

复用已核查的八分类训练器，所有相位同时联合训练。类别平衡NLL＋标签平滑0.02、捕获率负对数权重0.2、圆周相位平滑；AdamW、batch16、EMA0.95、梯度裁剪1。每个候选均30轮上限、至少15轮、patience8、min_delta0.0005，依据验证平衡NLL选EMA权重。相同seed下所有方法每轮看到相同图像顺序和逐样本仿射参数。

先在四层、seed17上对四组给予完全相同的三个候选预算，共12次验证实验：

| 候选 | 初始学习率 | 相位平滑权重 | 旋转范围 |
|---|---:|---:|---:|
| base | 0.002 | 0.02 | ±10° |
| lower_lr | 0.001 | 0.02 | ±10° |
| regularized | 0.002 | 0.05 | ±20° |

学习率均余弦降至各自初值的10%，平移±3像素、缩放±5%，不做色相扰动。根据四组各自选中权重的验证平衡NLL取平均，选平均最小的一个共同候选，用于全部三深度、三种子；不以MoE−D2NN差值选配置。保留所有候选的训练和验证成绩。正式36组中复用所选候选的四组四层seed17，另外32组重训。

训练期间不读取测试用于度量或选模；最终锁定36份权重，重放全部验证预测，然后统一测试。考虑3090与4090间浮点舍入，重放要求每张图argmax完全一致、每个分数最大绝对误差≤2×10⁻⁶、各指标差≤2×10⁻⁶，并保存实际误差；超限即失败，不能静默放宽。

## 资源、结果与图

总计最多五张已授权GPU。Blood实验占用0/1/3/4时，本任务只使用2；待这些任务实际退出并确认显存释放后，才允许向本任务的`gpu_schedule.json`加入空闲设备。协调器记录自身与worker命令/PID；异常终止仅处理自己的进程组。

测试报告准确率、宏平均召回率、宏F1、宏平均OvR AUROC及混淆矩阵；三种子逐点、均值、样本标准差、中位数一并呈现。每张真实图片示例保留样本ID与选择规则，成功和失败预测均展示。所有指标、模型排序和OEO增益依据实际结果，不预设MoE+OEO必须最高；不通过弱化D2NN、测试调参或挑种子制造差距。

```bash
python LightGenV2/demo_check/reproduction/prepare_kather2016.py --mirror --data-root /path/to/kather2016 --out LightGenV2/demo_check/runs/smoke/<prepare_id>
CUDA_VISIBLE_DEVICES=2 python LightGenV2/demo_check/reproduction/kather2016_experiment.py --phase smoke --data /path/to/kather2016_fixed_split.npz --out LightGenV2/demo_check/runs/smoke/<smoke_id>
python -u LightGenV2/demo_check/reproduction/kather2016_experiment.py --phase pilot --data /path/to/kather2016_fixed_split.npz --out LightGenV2/demo_check/runs/simulation/<pilot_id> --gpus 2
python -u LightGenV2/demo_check/reproduction/kather2016_experiment.py --phase suite --data /path/to/kather2016_fixed_split.npz --pilot LightGenV2/demo_check/runs/simulation/<pilot_id> --out LightGenV2/demo_check/runs/simulation/<suite_id> --gpus 2
```
