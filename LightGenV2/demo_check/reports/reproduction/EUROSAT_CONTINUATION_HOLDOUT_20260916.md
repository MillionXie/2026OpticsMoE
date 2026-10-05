# 共享电子前端：低学习率续训、早停及独立空间测试

完成MoE/D2NN的配对续训与一次性独立测试。MoE验证准确率从77.20%提高到77.55%，
但独立测试仅从75.65%到75.70%；这轮没有证据证明MoE获得稳定、显著的泛化提升。
没有观察到续训导致整体测试准确率或NLL恶化，但仍有训练/测试差距，不能保证完全没有过拟合。

## 续训协议

源run：`runs/simulation/eurosat_shared_frontend_20260916`。
新run：`runs/simulation/eurosat_shared_frontend_continuation_20260916`。
原结构、6000/2000张训练/验证图像、标签、空间划分及预处理见 [共享前端报告](EUROSAT_SHARED_FRONTEND_20260916.md)。
两组继续共用同一冻结CNN，BN统计量、特征编码和输出探测方式不变，不增加电子旁路或正则项。

从第20轮last检查点恢复光学相位及完整Adam动量/step（每个参数3760次更新），
最多追加40轮，总轮次60。新增学习率从0.001余弦降至0.0001；batch32、梯度裁剪1、
weight decay0、seed42及图像水平翻转不变。每轮额外进行完整训练集和验证集的无增强、固定权重评估。
验证NLL改善不足0.001连续12轮早停；保留原best作为候选，新增候选NLL不得高于原best的NLL，
再按验证准确率最高、其次NLL最低选模。所有规则在训练前固定，测试不参与选择。

MoE完成到第60轮，选中第49轮；D2NN第53轮触发早停，选中第41轮。
两组具有相同最大预算与早停规则，实际新增轮数分别40/33，不能称实际训练步数完全相同。
共同执行的21–53轮，其采样顺序、翻转种子和实际前端特征哈希全部一致；冻结前端状态始终未变。

## 结果及过拟合判断

准确率均按每张图像计算。测试数据为后述独立2000张原始test子集。

| 模型 | 训练准确率（固定权重、无增强） | 验证准确率 | 测试准确率 | 验证NLL | 测试NLL |
|---|---:|---:|---:|---:|---:|
| 原MoE | 80.38% | 77.20% | 75.65% | 0.7097 | 0.7515 |
| 续训MoE | 80.72% | 77.55% | 75.70% | 0.7004 | 0.7457 |
| 原D2NN | 78.58% | 73.65% | 72.85% | 0.8877 | 0.9559 |
| 续训D2NN | 79.10% | 73.65% | 73.30% | 0.8728 | 0.9327 |

MoE训练−验证差距约3.18→3.17个百分点，训练−测试差距约4.73→5.02个百分点。
D2NN训练−验证差距约4.93→5.45个百分点，训练−测试差距约5.73→5.80个百分点。
因此不能因为验证损失下降就声称不存在过拟合；当前更准确的判断是“续训没有明显整体泛化恶化，但收益有限”。
MoE验证上涨7张，独立测试仅净增加1张；D2NN测试净增加9张。

| 模型 | 测试RGB | 测试SAR |
|---|---:|---:|
| 原MoE | 78.90% | 72.40% |
| 续训MoE | 79.20% | 72.20% |
| 原D2NN | 75.60% | 70.10% |
| 续训D2NN | 75.30% | 71.30% |

MoE测试中纠正13个、引入12个错误；D2NN纠正28个、引入19个错误。
按原始空间组整体重采样（82组、10000次、seed20260916），续训−原模型测试准确率差值的
95% percentile bootstrap区间：MoE +0.05个百分点，区间[-0.75,+0.78]；
D2NN +0.45个百分点，区间[-0.19,+1.37]。均包含0，不能宣称稳定显著提升；该区间也不替代多训练种子验证。
MoE−D2NN的综合测试差距从2.80降至2.40个百分点；不能只报告验证差距扩大而省略测试结果。

## 独立测试如何保证隔离

从原 `SPLIT.json` 的test部分，每类按 `SHA256(shared_frontend_holdout_v1|pair_id)` 排序取100个RGB/SAR配对，
10类×100配对×2域=2000张，涉及82个空间组。与全部原始train/validation空间组均无交集。
解码、RGB中心裁剪和SAR重投影/三通道编码复用原预处理函数，先重算40张已有训练图像验证逐像素一致。
没有根据模型预测筛测试样本；不读取测试结果来改学习率、早停或选模。

原始与续训的四份验证选定权重先写入 `selection_lock.json`，之后才读取测试数组和计算指标。
测试run：`runs/smoke/eurosat_shared_frontend_holdout_20260916`。这是原测试划分的固定均衡子集，不是完整原测试集。
本轮测试后未再调参或换选权重；后续若继续在该测试结果上指导设计，应承认这次测试信息已被使用。

## 核验、源码及文件身份

训练commit：`ef70e50f`；测试/独立核验代码commit：`54ca0718`。
GPU0 RTX4090、Python3.11.15、PyTorch2.6.0+cu124；测试准备使用同一xml环境衍生的demo_pureoptics虚拟环境。
完整环境、配置、原始命令及源码哈希在对应run metadata中。
训练入口 [continue_training.py](../../shared_frontend/continue_training.py)，协议 [continuation.json](../../shared_frontend/continuation.json)。

```bash
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python -u LightGenV2/demo_check/shared_frontend/continue_training.py --phase train --source LightGenV2/demo_check/runs/simulation/eurosat_shared_frontend_20260916 --data /DATA/DATA1/guest3/demo_reproduction_data/eurosat/phase_only_v1/data.npz --out LightGenV2/demo_check/runs/simulation/eurosat_shared_frontend_continuation_20260916
/home/guest3/.venvs/demo_pureoptics/bin/python LightGenV2/demo_check/shared_frontend/prepare_holdout.py --root /DATA/DATA1/guest3/demo_reproduction_data/eurosat --out /DATA/DATA1/guest3/demo_reproduction_data/eurosat/shared_frontend_holdout_v1
CUDA_VISIBLE_DEVICES=0 /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/demo_check/shared_frontend/evaluate_holdout.py --original LightGenV2/demo_check/runs/simulation/eurosat_shared_frontend_20260916 --continued LightGenV2/demo_check/runs/simulation/eurosat_shared_frontend_continuation_20260916 --data /DATA/DATA1/guest3/demo_reproduction_data/eurosat/shared_frontend_holdout_v1/data.npz --out LightGenV2/demo_check/runs/smoke/eurosat_shared_frontend_holdout_20260916
```

SHA256：

- 训练/验证数组：`f543d0c9ce330272063a1669ebacef082b26dc77994ec76b251f477634e350bd`
- 测试数组：`28724ecedaeb3b98cc4544c5dc73debc9a5a4ecb7ef4b81e1141749d66d8d61d`
- 测试manifest：`f8d5dbaaf7593f2ee380bd4b1eae06d7791d4c8fe4f61b7a7cc4401a4dc78b60`
- 同一冻结前端检查点：`61d3c189a18d814a22e5ffea2986cd0902c8e9b2cb4de6dbd90aae94e9ace15f`
- 续训MoE最佳权重：`c8ac9663722ab067515c62eaa6998dadb9ab8beb10040376b34309c3129ff317`
- 续训D2NN最佳权重：`3f12f173b30b8796bc8112ca44176a6c20528fde6c1cb33d1821744cd665d5ba`

恢复smoke：`runs/smoke/eurosat_shared_frontend_continuation_20260916`，检查原预测、Adam矩/step和有效更新。
新进程复评：`runs/smoke/eurosat_shared_frontend_continuation_reload_20260916`，
两组完整训练/验证指标及2000张验证概率逐位一致。本地 `verify_continuation.py` 独立重算选模、NLL门槛、早停计数、采样/特征一致性，全部通过。
`verify_holdout.py` 独立核验2000张测试像素哈希、标签、ID/空间隔离、锁定模型身份和全部CSV指标，全部通过。
权重、预测、测试数据子集、清单与续训曲线已保存本地；原始run未覆盖。服务器保留best/last，训练及评估进程已结束。

现阶段不建议仅靠增加轮数继续追逐MoE验证准确率。若做下一轮结构优化，应优先检验空间特征/编码，
保持共享前端与公平对照，并用新的、预先固定的泛化验证协议；本报告不把未执行的方向记为改进结果。
