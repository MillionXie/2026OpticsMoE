# 电子SAM子空间的GSAM训练对照

## 方法与边界

根据[GSAM，ICLR 2022](https://research.google/pubs/surrogate-gap-minimization-improves-sharpness-aware-training/)
及[作者代码](https://github.com/juntang-zhuang/GSAM/blob/main/gsam/gsam.py)，
令g0为原权重梯度，g1为SAM扰动权重梯度；取
`v = g0 - dot(g0,g1)/dot(g1,g1) * g1`，再用`g1 - 0.1*v`更新。
v在欧氏坐标中与g1正交，用于约束surrogate gap，而不增加推理模块。

这里只修正既有SAM电子子空间；相位使用原SAM第二次反向的普通梯度，Qwen前端始终冻结。
采用AdamW，受其预条件影响，不能把欧氏正交等同于最终参数更新正交，也不声称完整复现论文。
0系数与旧SAM逐值相同；不支持ASAM或其他辅助训练组合。两次反向共用同一噪声实现，
权重与随机状态异常安全恢复；一次optimizer/EMA更新，原梯度裁剪保留。
无新增推理参数，未加入Transformer、attention、分支或测试后处理。

## 诊断与协议

训练集随机32图CPU无更新诊断（seed20260913，非测试成绩）：g0/g1余弦.89957，
正交分量占g0范数.43678，系数.1的修正占g1范数.02709。
795146个电子坐标全部恢复逐值相同，optimizer更新0次；
`runs/smoke/gsam_direction_20260913/report.json`保存样本ID、来源SHA和原始数值。
这仅表明修正有非零方向，不证明泛化收益。

继承`moe_alpha40_extra_control.yaml`，固定87ad best初始化；原batch32/SAM .05/EMA .995、
GT多指标与空间CC教师权重2不变。20轮预算，第16轮进入既有低LR精修；
Top2、alpha>=.4、同尺度融合、478ROI、224专家、17微米/10cm、训练DC20–30%均不变。
冻结Qwen、85412参数头和所有推理权重形状不变；SAM/GSAM系数是训练超参数，不是融合alpha。
完整10000 train/5000 public-test；第1/5/末轮测试选best，无独立validation。
持续下降时提前停止并记录实际last。保留best/last，不生成每5轮PT。
标准测试关闭随机光学噪声，报告不能冒称实际漏光性能。

```bash
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --phase all --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_gsam_20260913.yaml
```

正式运行前需要CPU回归、真实输入冻结/梯度检查和GitHub同步。默认只用一张空闲GPU，
不与其他用户作业争抢；测试通过与启动信息后补，当前文档不代表已得到新成绩。
