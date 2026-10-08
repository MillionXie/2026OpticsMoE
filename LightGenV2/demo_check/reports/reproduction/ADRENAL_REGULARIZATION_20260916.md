# Adrenal 配对正则化实验

## 目的与范围

检查数据增强与相位平滑约束是否缓解深层模型的泛化差距。原12组重训已确认各层相位更新和梯度有效；
增加深度提高了末轮训练拟合能力，但未稳定提高测试AUROC。本实验不将测试指标单调性作为选模条件。
原结果与权重保留，不覆盖归档代码。原测试集此前已查看，本轮属于回顾性开发实验，不称为新的盲测。

增加深度同时增加固定距离的传播段和OEO次数；相位板设为常数不能一般性地消除新增传播及非线性。
当前深层模型没有通过关闭新增层严格还原浅层模型的显式路径，不能由层数增加推出测试性能必然单调。
正则化是否有效，需联合检查同一所选权重的训练/验证差距、验证性能及测试表现；只降低训练分数不算改善泛化。

## 实测结果：训练改进与选模问题同时存在

12组均完成50轮，每组7450次更新；只跑seed17，没有把本轮说成多种子结论。
增强和平滑确实缓解了六层MoE＋Softsign的泛化差距，但没有让每一组模型都提高。

同一验证所选checkpoint上，六层MoE的训练/验证AUROC从0.8859/0.7817变为0.8378/0.7853，
差距从0.1042缩小至0.0525；测试从0.7155升至0.7413。
六层D2NN＋Softsign的训练/验证从0.8488/0.7315变为0.8028/0.7865，
测试从0.6326升至0.6810。两种架构均获益，不能只改进MoE而保留旧D2NN作比较。

沿用原验证选模规则的完整测试AUROC如下，数字不是分类准确率：

| 模型 | 2层：原→新 | 4层：原→新 | 6层：原→新 |
|---|---:|---:|---:|
| MoE＋Softsign OEO | 0.7216 → 0.7391 | 0.7013 → 0.6911 | 0.7155 → 0.7413 |
| D2NN＋Softsign OEO | 0.6564 → 0.6395 | 0.6386 → 0.6455 | 0.6326 → 0.6810 |
| MoE，无中间OEO | 0.6720 → 0.6758 | 0.6845 → 0.6882 | 0.6812 → 0.6872 |
| D2NN，无中间OEO | 0.5569 → 0.5384 | 0.6196 → 0.5634 | 0.6007 → 0.6086 |

四层MoE＋Softsign仍出现低谷，且两/四层无OEO D2NN退步，故不将这个正则化配置直接替换所有原baseline。
对固定测试样本作2000次类别分层、样本配对bootstrap，六层MoE＋Softsign新减旧AUROC的95%分位区间为
[0.0016,0.0485]，六层D2NN＋Softsign为[0.0148,0.0840]。
这是固定权重在本测试集上的样本抽样区间，未按患者分组，也不包含训练种子波动，也没有作多重比较校正；不据此宣称普遍显著优势。
逐项区间、召回率、阈值和混淆矩阵保存在run，而不是只保留提高的组。

## 选模诊断：统一第50轮再作配对比较

新版四层MoE＋Softsign由验证集选中了第8轮：
验证AUROC为0.757177，测试为0.691096。
第50轮的验证AUROC为0.755981，仅低0.001196（对应76×22个正负配对中约2对的净排序优势），
测试却为0.737295。记录说明：报告中的明显低谷与选中早期权重有关，不能全部归因于深层过拟合。
验证集只有76例正常、22例增生，这种细小验证优势不足以证明一个checkpoint更能泛化。

为分离训练变化与选模变化，原版和新版的12组模型都统一复评各自第50轮，不只替换四层MoE。
以下两侧都使用相同轮次，不能将“原版best→新版last”的差异归因于正则化：

| 模型 | 2层：原→新 | 4层：原→新 | 6层：原→新 |
|---|---:|---:|---:|
| MoE＋Softsign OEO | 0.7219 → 0.7388 | 0.7324 → 0.7373 | 0.7150 → 0.7414 |
| D2NN＋Softsign OEO | 0.6564 → 0.6544 | 0.6386 → 0.6455 | 0.6341 → 0.6810 |
| MoE，无中间OEO | 0.6715 → 0.6763 | 0.6845 → 0.6882 | 0.6812 → 0.6883 |
| D2NN，无中间OEO | 0.5569 → 0.5384 | 0.6205 → 0.6098 | 0.6326 → 0.6312 |

新版MoE＋Softsign为0.7388/0.7373/0.7414，四层的大幅低谷消失，
但2→4层仍有约0.0015的下降，不能写成严格单调提升；更符合本轮观察的是性能接近平台。
无OEO D2NN在统一50轮下仍有浅层退步，这也说明“曲线单调”本身不是性能改进证据。

**第50轮复评是在查看本轮best测试结果后提出的事后诊断，不替换原来按验证集选模的正式记录。**
两套策略全部保留，不逐模型混选更高的测试分数，也不把该诊断冒充预先确定的最终论文协议。
下一轮确认实验应在训练前固定选模规则，并用多种子检查稳定性；验证规则不应以测试曲线形状调整。

固定0.5判决下，六层MoE＋Softsign仍全部预测正常；正则化提高AUROC没有自动解决判决阈值。
验证集选出的阈值约0.3358，在测试上检出23/69增生、误报24/229正常，
平衡准确率0.6143、总体准确率0.7651。因此不能把AUROC提升写成分类准确率全面提升。

## 固定协议

数据沿用同一NPZ和官方1188/98/298划分，正常/增生分别为929/259、76/22、229/69。
三维形状的轴向平均投影与28→100双三次插值不变，输入为单通道振幅 `B×1×100×100`。
只在训练时进行二维仿射增强：角度均匀采样于±8°，缩放0.97–1.03，归一化采样坐标的平移范围
相当于输入100像素尺度上的±3像素。双线性重采样、零填充、`align_corners=False`，不翻转、不做颜色变换。
仿射参数由种子、轮次及样本位置确定，独立于模型随机数；同一种子下12组模型使用完全相同的逐样本变换和顺序。
标签不变，验证及测试输入不增强。该扰动假定小幅姿态/位置差异不改变形状类别，不声称它模拟了全部临床变化。

模型沿用原2/4/6层MoE、D2NN以及Softsign OEO开/关，共12组。参数量、传播距离、探测窗口、
理想九端口分光及局部传播实现均未改变；无Qwen、电子残差、α或可训练电子分类头。只训练相位参数。

总损失为原归一化探测面MSE加0.05倍圆周相位平滑项：

`L = L_original_MSE + 0.05 × [Σ_(所有相位板相邻像素对) (1−cos(φ_i−φ_j)) / 相邻像素对总数]`

相邻对只取每块相位板内部横向和纵向的相邻像素，不跨板连接，也不跨边界周期连接。
实际相位为 `φ=2πsigmoid(raw_phase)`。使用圆周差异避免将0与2π错误视作相差很远；
按总相邻对数平均，避免层数或专家张量数增加直接放大惩罚系数。路由相位也包含在约束中。
本实验检验增强与正则化的组合效果，不能单独归因于其中一项。

训练仍为seed17、50轮、batch8、Adam lr0.001、weight_decay0、StepLR每10轮乘0.7，
同一次反传联合更新所有层，每模型7450次更新。没有通过减少深层训练预算追求较小差距。
按验证AUROC最高选checkpoint，1e-6容差内按验证探测面MSE较低者选取，规则与原版相同。
旧代码已有最佳验证权重选择，因此不把“增加早停”当作原代码缺失的功能。

同时保留固定0.5判决，并在验证集选择平衡准确率最大的阈值（并列依次取准确率高、距0.5近、阈值小者）。
所有模型的checkpoint、配置及阈值SHA256封存后，独立测试命令才读取测试数组。
阈值调整不改变AUROC，也不是缓解过拟合的措施。

实现采用[PyTorch仿射网格及采样定义](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.affine_grid.html)，
两处统一使用`align_corners=False`。增强在无梯度输入上执行，不引入可训练电子模块。

## 运行与核验

源码入口：`../../reproduction/train_adrenal_regularized.py`；配置：`../../reproduction/adrenal_regularization.json`。
独立CPU核验：`../../reproduction/verify_adrenal_regularized.py`。

训练源码commit：`956eb9c15b03a0ae7e2a63c31889bdafcad874b9`。
数据SHA256：`22bc193a85c43be85f093bf12f3ffdc5ed79b5efda63925bd9e063a70594bf93`。
环境：GPU0 RTX4090，Python3.11.15，PyTorch2.6.0+cu124；完整依赖清单保存在metadata。

run：`runs/smoke/adrenal_regularized_20260916`、`runs/simulation/adrenal_regularized_s17_20260916`。
全部使用已推送GitHub的代码；输入数据及归档源文件均记录SHA256。
每模型config.json保存归档结构配置，其中旧experiment路径字段未用于加载本轮数据；
实际数据路径以metadata中的命令为准，实际训练参数以同文件中的protocol为准。

```bash
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/reproduction/train_adrenal_regularized.py --phase smoke --data /DATA/DATA1/guest3/demo_reproduction_data/adrenal/data.npz --out LightGenV2/demo_check/runs/smoke/adrenal_regularized_20260916
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/reproduction/train_adrenal_regularized.py --phase train --data /DATA/DATA1/guest3/demo_reproduction_data/adrenal/data.npz --out LightGenV2/demo_check/runs/simulation/adrenal_regularized_s17_20260916
```

独立核验已通过12组的训练/验证/测试AUROC、混淆矩阵、验证选模轮次、阈值，
原/新参数量、初始化哈希及50轮训练顺序一致性，新版12组增强参数哈希一致。
所有相位参数张量更新，检查点SHA256与下载清单一致。

权重在服务器保存best及last；本机保存best、逐样本预测和下载manifest。
新旧固定第50轮的24份checkpoint身份已在服务器校验，逐样本指标另在本机独立重算；
本机不保存这些last二进制，身份核对使用先前审计与服务器记录。

附加run：
- `runs/simulation/adrenal_regularized_fixed50_audit_20260916`：新版统一50轮诊断，源码commit `8f6a28ad`。
- `runs/simulation/adrenal_original_fixed50_audit_20260916`：原版统一50轮诊断，源码commit `dcab887e`。
- `runs/smoke/adrenal_fixed50_verification_20260916`：24份固定轮次预测的独立核验及策略比较PNG/PDF。

[原规则的训练/验证/测试对比图](../../runs/simulation/adrenal_regularized_s17_20260916/regularization_comparison.png)；
[选模策略对比图](../../runs/smoke/adrenal_fixed50_verification_20260916/selection_policy_comparison.png)。
对应PDF在同目录，可独立导出。图中保留所有原始数据点，没有平滑或修改指标。

重放第一版固定权重测试需使用训练commit `956eb9c15b03a0ae7e2a63c31889bdafcad874b9` 的干净检出；
入口会拒绝源码哈希与test lock不符的运行。后续NLL选项来自另一commit，不绕过此检查。

```bash
# 第一版测试，使用上述956eb9c1源码。
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/reproduction/train_adrenal_regularized.py --phase test --data /DATA/DATA1/guest3/demo_reproduction_data/adrenal/data.npz --out LightGenV2/demo_check/runs/simulation/adrenal_regularized_s17_20260916
# 固定轮次诊断，参数接口在dcab887e中同时支持新旧两版。
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/reproduction/audit_adrenal_fixed_epoch.py --run LightGenV2/demo_check/runs/simulation/adrenal_regularized_s17_20260916 --data /DATA/DATA1/guest3/demo_reproduction_data/adrenal/data.npz --out LightGenV2/demo_check/runs/simulation/adrenal_regularized_fixed50_audit_20260916
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/reproduction/audit_adrenal_fixed_epoch.py --original-depth-audit LightGenV2/demo_check/runs/simulation/adrenal_depth_audit_20260916 --data /DATA/DATA1/guest3/demo_reproduction_data/adrenal/data.npz --out LightGenV2/demo_check/runs/simulation/adrenal_original_fixed50_audit_20260916
python LightGenV2/demo_check/reproduction/verify_adrenal_regularized.py --run LightGenV2/demo_check/runs/simulation/adrenal_regularized_s17_20260916 --task LightGenV2/demo_check
python LightGenV2/demo_check/reproduction/verify_adrenal_fixed_epoch.py --task LightGenV2/demo_check --out LightGenV2/demo_check/runs/smoke/adrenal_fixed50_verification_20260916
```

再次运行训练或诊断须使用新输出目录，入口拒绝覆盖已有run。

## 类别平衡分类损失的小对照（仅训练/验证）

为检查光斑MSE与分类目标的关系，另对六层MoE＋Softsign和六层D2NN＋Softsign各训练50轮；
除损失外均沿用上面的增强、平滑及优化设置：

`L = L_MSE + 0.05R + 0.1 × mean_b[w_(y_b) × (−log p_(b,y_b))]`

其中`p`仍为两个探测窗口能量的归一化比例，`w_c=N_train/(2N_train,c)`，
正常/增生权重约为0.639397/2.293436。按样本数求平均，不按每个batch的权重和重新归一化；不重复采样少数类。
0.1是训练损失系数，不是电子残差混合系数；没有增加可训练电子分类头，推理结构不变。
保留MSE项以维持探测面能量聚焦目标；没有只优化两个窗口的能量比而忽略整场。

| 六层模型 | 增强＋平滑：验证AUROC | 再加平衡NLL：验证AUROC |
|---|---:|---:|
| MoE＋Softsign | 0.7853 | 0.8020 |
| D2NN＋Softsign | 0.7865 | 0.7596 |

两组都选第49轮，所选权重训练AUROC分别为0.8377、0.8018。
默认0.5阈值下，验证集MoE仅检出1/22增生，D2NN为0/22；
这一个0.1系数的小对照没有普遍解决判决退化，也没有对两种架构一致提高验证AUROC。
因此仅保留为可选实验配置，未升为统一默认，未进一步以测试成绩筛选系数。
**本对照不计算测试指标，0.8020是验证AUROC，不能混入上面的测试表。**
只检查了六层Softsign，不据此作完整深度或OEO因果比较，也不泛化为“分类损失无效”。

源码commit：`f32f228fe8c00eebdcd685a12fff4c182dcde423`。
配置：`../../reproduction/adrenal_classification_regularization.json`。
run：`runs/simulation/adrenal_classification_probe_s17_20260916`；
smoke：`runs/smoke/adrenal_classification_regularized_20260916`。
两组各7450次更新，训练/验证指标、初始化、顺序、增强一致性和best权重SHA256已独立核验。
`test_lock.json`仅封存候选权重，并未执行测试入口。

```bash
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/reproduction/train_adrenal_regularized.py --phase smoke --profile LightGenV2/demo_check/reproduction/adrenal_classification_regularization.json --variants moe_L6_oeo_relu_softsign d2nn_L6_oeo_relu_softsign --data /DATA/DATA1/guest3/demo_reproduction_data/adrenal/data.npz --out LightGenV2/demo_check/runs/smoke/adrenal_classification_regularized_20260916
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/reproduction/train_adrenal_regularized.py --phase train --profile LightGenV2/demo_check/reproduction/adrenal_classification_regularization.json --variants moe_L6_oeo_relu_softsign d2nn_L6_oeo_relu_softsign --data /DATA/DATA1/guest3/demo_reproduction_data/adrenal/data.npz --out LightGenV2/demo_check/runs/simulation/adrenal_classification_probe_s17_20260916
python LightGenV2/demo_check/reproduction/verify_adrenal_regularized.py --run LightGenV2/demo_check/runs/simulation/adrenal_classification_probe_s17_20260916 --task LightGenV2/demo_check
```

本轮共完成14组50轮训练，训练进程均已退出；GPU0无计算进程、占用回落至12MiB。
诊断入口兼容性另在`runs/smoke/adrenal_fixed50_refactor_parity_20260916`检查：
重构前后24份验证/测试预测CSV逐字节一致。没有改写历史指标或权重。
