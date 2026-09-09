# 两参数读出校准诊断

这是独立候选，不改原模型/原指标，不代表已达到CC=0.87。
在多轮续训未超过0.858120后，检查是否能通过极低容量的输出校准改善泛化。

## 结构边界

原光电网络完整冻结，包括光router Top2、两层光电融合、alpha≥0.4及原读出头。
最终logits增加`z' = s*z + b*r(x,y)`：

- 仅训练两个标量，`s=exp(log(2)*tanh(a))`在[0.5,2]内，`b=2*tanh(c)`在[-2,2]内。
- `r`是固定`-(x²+y²)`减空间均值，x/y为像素中心映射到[-1,1]的坐标。
- 初始化s=1,b=0，严格保留原输出；位置偏置允许为负，不强制中心更亮。
- 不读取图像内容来生成新分支，没有attention/Transformer/额外特征主干。
- 这是最终读出增加2个参数，不是新增光学步骤，也不是CCD图像预处理。
- 不能把此候选和原Qwen头称为参数严格相同：原光电解码头85412，此候选等效85414。
  若采用为正式模型，要披露此差异，并按需要为baseline提供同类校准对照。

参考[DeepGaze IIE (ICCV2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Linardos_DeepGaze_IIE_Calibrated_Prediction_in_and_Out-of-Domain_for_State-of-the-Art_Saliency_ICCV_2021_paper.pdf)
中读出校准和空间先验的动机。本实现使用两标量有界解析形式，不复现其主干、集成、模糊模块或整套训练。

## 训练、选择与评估

1. 从`moe_alpha40_hint_control.yaml`指定的不可变best加载完整原光电网络，SHA校验沿用原入口。
2. 对全部10000张train在原标准eval模式生成logits，只临时保存在CPU RAM的float32缓存中，
   和真值合计约4GB；不保存额外大型中间文件。不启用图像/光学随机增强，不更新原网络。
3. 只拟合a/c：Adam lr=.03，20轮，batch64；损失为`1-CC + .001*((log s)^2+b^2)`。
   按完整训练集CC保存校准best，不用test选校准epoch。所有拟合ID必须是唯一train ID。
4. 训练完成后在相同5000张public test，成对计算原模型/校准后结果；不改测试归一化、标签、样本或光学条件。
   原项目和源权重已使用public test选模，这依然不是从未接触测试集的独立盲测。

```bash
python -m pytest LightGenV2/tasks/t03_saliency/tests -q
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m LightGenV2.tasks.t03_saliency.calibration_probe --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_hint_control.yaml --output LightGenV2/tasks/t03_saliency/runs/simulation/moe_alpha40_readout_calibration_fp32_seed42 --epochs 20 --batch-size 16
```

GPU编号为示例，先确认余量；输出目录必须不存在。正式数据/模型依赖与复现入口相同。
`calibration_report.json`记录两套完整指标；`per_image_cc.csv`保存逐样本配对结果。
`protocol.json`记录源码commit、命令、源SHA、训练ID SHA和口径，`training_history.json`记录两个标量。
`best_checkpoint.pt`/`last_checkpoint.pt`仅包含`calibration`、`base`来源及格式标识；
**不是能直接传给旧run入口的core checkpoint**。复现需要源checkpoint，以及
`SpatialLogitCalibration`严格加载`payload['calibration']`后应用到原logits再softmax。
若性能确实改善，仍需完整集成、独立复评并审计光学参数/专家分布，才能作为正式部署候选。

## 推理精度审计

首个`moe_alpha40_readout_calibration_seed42`使用源码84d11b5f，误在缓存与测试的整个forward外开启AMP。
它的原模型CC=0.85823985、校准CC=0.85854394；前者与正式源值0.858120有偏移，
**该轮不能更新正式成绩**。产物保留为可追溯诊断，不静默覆盖。
修正后使用`standard_logits`显式禁止外层autocast（保留Qwen前端自身权重dtype），
与原`legacy.evaluate_model`对齐；新增精度回归测试与源CC漂移门限2e-5。
新目录带`fp32`指光电body/head标准计算精度，不表示把Qwen前端权重转换成FP32。

## 完成的标准精度诊断

`moe_alpha40_readout_calibration_fp32_seed42`已完成；源码`4723a0d519e23deb58402b7b57f94865be838301`，
全部10000张train拟合20轮，按train CC选第11轮，然后一次成对评估5000张public test。

| 指标 | 原光电 | 加2参数校准 |
|---|---:|---:|
| CC | 0.8581201385 | 0.8586080803 |
| KLD | 0.1153222898 | 0.1157713949 |
| SIM | 0.8220984152 | 0.8221044438 |
| NSS | 0.9655874199 | 0.9666025529 |

最终`s=1.006434083`、`b=0.040369064`。逐图CSV有5000个唯一ID，
配对平均CC增益0.0004879468，中位数0.0003958941，2999张改善；原参考值通过2e-5漂移检查。
这些是同批预测的描述统计，不是独立多seed显著性结论。
CC小幅改善、KLD略变差；该方案不足以达到0.87，因此保留为诊断，不替换正式可部署core checkpoint。

证据位于该run：

- `best_checkpoint.pt` SHA256：`68061c28d34e5609c4efac0cec6d0f7789b9f4d1056a4a794d7c9d7012463e87`
- `calibration_report.json` SHA256：`dda9015f63cbe5eafa3bee938fbc02f1e2657d907dad353a6b851b975c4688f9`
- `per_image_cc.csv` SHA256：`d7ec0e9d810ddd02935dc08d102123ef6b2a36440be3f2029838af992daa1dd5`
