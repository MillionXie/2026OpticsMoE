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
| KLD | 0.1153222898 | 0.1157139295 |
| SIM | 0.8220984152 | 0.8221044438 |
| NSS | 0.9655874199 | 0.9660255295 |

最终`s=1.006434083`、`b=0.040369064`。逐图CSV有5000个唯一ID，
配对平均CC增益0.0004879468，中位数0.0003958941，2999张改善；原参考值通过2e-5漂移检查。
这些是同批预测的描述统计，不是独立多seed显著性结论。
CC小幅改善、KLD略变差；该方案不足以达到0.87，因此保留为诊断，不替换正式可部署core checkpoint。

证据位于该run：

- `best_checkpoint.pt` SHA256：`68061c28d34e5609c4efac0cec6d0f7789b9f4d1056a4a794d7c9d7012463e87`
- `calibration_report.json` SHA256：`dda9015f63cbe5eafa3bee938fbc02f1e2657d907dad353a6b851b975c4688f9`
- `per_image_cc.csv` SHA256：`d7ec0e9d810ddd02935dc08d102123ef6b2a36440be3f2029838af992daa1dd5`

## 后续固定权重平滑诊断（2026-09-10，不采用）

在已核验87ad权重（完整SHA见SAM_TRAINING）的标准CPU推理上，检查输出是否仅因平滑不足而受限。
这是前64张有序train图的探索，不是完整测试、不修改权重、不保存新PT，也不占用第三张GPU。
源码718639bd，原`moe_alpha40_extra_control.yaml`、batch4、无增强、eval关闭随机扰动，
逐图用float64 Pearson再平均。原空间softmax密度图依次做sigma=0/1/2/4/8像素的
可分离高斯平滑（224输出像素单位、半径3sigma、reflect padding、核和为1）。
仅预测被处理；真值、坐标、CC定义不变。若采用，必须视为新输出处理而非原模型分数。

|sigma（像素）|前64张训练图CC|
|---|---:|
|0（原输出）|.8799952037|
|1|.8800250506|
|2|.8800390023|
|4|.8796210585|
|8|.8744298866|

最佳探索增益仅.00004380，不支持把额外平滑加入正式模型，也不足以解释.88目标差距。
与上面的两参数校准是不同来源/不同小样本协议，不混合比较或声称是独立泛化收益。

## 师生空间尺度差距诊断（2026-09-10，不改变评分）

目的：区分是否仅缺局部细节。没有拟合/更新任何参数，没有保存额外PT或缓存，不占GPU。
执行源码`aa0dc202`，CPU/xml、OMP/MKL各2线程，seed42。使用
`moe_alpha40_extra_control.yaml`初始化87ad的原core和头，标准FP32光电推理、无外层AMP，
冻结前端保留自身dtype；eval关闭随机光扰动和phase dropout，RGB224无增强，batch4。
取原训练manifest有序前128个ID（均曾用于训练，非验证集），ID逐行加末尾换行的SHA256为
`de031dc4f1e5b139e39db68e3c229fa68e1a521b4fadff342f5f982fb26c5ff0`。
教师使用原10k训练logit缓存（a45a90fe，完整SHA见SAM_TRAINING/相关蒸馏配置），
载入后核查全部ID顺序与教师531c SHA一致，不重新调用教师Transformer或访问测试图片。

复算：`prepare_salicon(settings,persist=False)`后，`legacy.build_loaders(...,training=False)`
的train dataset取`Subset(range(128))`，保留原collate，num_workers=0。
学生由`standard_logits(model,preprocess_vision(...))`获取logits；按同ID选缓存教师logits。
二者分别用原`density_from_logits`转概率密度，GT保持原dataset密度标签。
对S/T/GT三者都使用`adaptive_avg_pool2d(value,(z,z))`，z依次7/14/28/56/112/224；
每张转float64、展平减自身均值，再算Pearson并跨128图平均，无逐图配准或尺度拟合。
这会改变诊断网格，**不同z不是同一个正式指标，不能将粗网格高分填入论文或替换224测试**。

|诊断网格|学生–GT CC|教师–GT CC|学生–教师 CC|
|---|---:|---:|---:|
|7×7|.91100484|.93238391|.92012431|
|14×14|.89000258|.91213472|.90641895|
|28×28|.88053752|.90193719|.90010429|
|56×56|.87770477|.89861086|.89800084|
|112×112|.87695769|.89761204|.89732791|
|224×224|.87674842|.89733421|.89713450|

原网格师生GT差约.020586，7×7仍约.021379。在这128张中，误差并非仅由细节分辨率造成，
因此不据此增加锐化/平滑后处理；先验证已运行的空间关系蒸馏。此诊断不能证明关系蒸馏一定有效，
也不能把小样本训练差距外推为全部test或实测CCD结论。CPU进程1723514正常退出，未占第三张GPU。
