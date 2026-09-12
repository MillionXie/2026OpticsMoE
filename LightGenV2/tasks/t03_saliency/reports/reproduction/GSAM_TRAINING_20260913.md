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

## 测试、真实更新与启动

源码`c811fb0732692442721e414cc87f62957567fca6`通过261项CPU回归（43.28秒，13条既有警告），
已push GitHub分支`experiment/salicon-gsam-20260913`。
最初测试暴露浅层临时配置路径越界；仅为可选缓存回退添加深度保护，未改变正式路径或网络。
GSAM公式、0系数逐值等价、相位不受修正、单次更新、随机状态与异常恢复均有测试。

CPU真实8图单步`runs/smoke/gsam_20260913/report.json`：前端3933184参数冻结且逐值未变，
原生Transformer调用0，读出头85412，六张相位均有有限非零梯度与更新，optimizer更新1次。
实际GSAM相对梯度修正.027115；router原始参数RMS更新1.2684e-5，其余约.0001945–.0001981。
单批训练CC .89519不是测试成绩，不能填论文表格；正式训练仍为batch32。

UTC2026-09-12 20:42:35启动PID/PGID646567，GPU1 UUID
`GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d`，固定worktree `.worktrees/t03_balance`。
运行时禁止checkout；配置SHA `d99329b00fc2bcb708229f4006cfdc2d5a5f1b3faa5f5be112c97ab25b7dd9079`。
产物`runs/simulation/moe_alpha40_gsam_20260913_seed42`的`launch_record.json`记录完整命令/环境。
仅20轮预算，不等于已完成；第1/5/10轮检查完整public-test，持续回落时停止并保留best/last。
旧console前缀沿用`[student SAM]`；配置中的`gsam_coefficient=.1`与训练指标
`gsam_relative_correction`记录实际启用修正，不能仅凭console前缀判断是否使用GSAM。
