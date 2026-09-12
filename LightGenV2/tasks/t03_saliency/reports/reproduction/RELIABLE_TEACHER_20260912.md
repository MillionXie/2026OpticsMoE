# 训练时减弱不可靠教师监督

候选配置：`configs/moe_alpha40_reliable_teacher50.yaml`，不增加推理参数，不改结构、读出头、
光router Top2、alpha>=0.4、光路几何和20%–30%训练直流扰动。

依据：CA-MKD使用训练标签衡量教师可靠性，避免低质量教师监督。
原文：[Confidence-Aware Multi-Teacher Knowledge Distillation](https://arxiv.org/abs/2201.00007)。
这里只借鉴可靠性加权动机，使用单教师、相对空间CC；不是其多教师分类算法的复现。
并无论文证据保证这一改编在SALICON或当前光电模型上有效，需要实测。

每个训练样本计算学生对GT的CC `cs`，教师对GT的CC `ct`，随后停止权重梯度：

`w = max(0.25, exp(-max(0, cs-ct)/0.025))`

原来的教师CC损失改为 `mean(w * (1-CC(student,teacher)))`，蒸馏系数仍为2。
教师更好的样本权重保持1，教师差于学生的样本最低降至0.25。不删除任何训练样本，
原GT的KL/CC/SIM/NSS全部保留。恒定GT不能评价教师可靠性，保持普通KD；
恒定教师没有空间信息，继续按历史空间KD逻辑排除。权重不按batch重新归一化，
因此总蒸馏强度可能下降；不能声称它与固定KD2具有完全相同的损失总权重。

SAM两次forward沿用相同随机扰动；可靠性权重分别由各次预测重新计算并detach，
与GT和教师均不反传。不是额外的可训练gate，也不会在推理时用GT或教师。

本轮从更早已核验的 `c88e1a41...`、CC约0.85953的同结构权重开始，50 epoch。
与已完成的 `moe_alpha40_sam_spatialcc_kd2_seed42` 同源同日程，只有上述KD规则不同。
不从近期测试下降的hard-CC最后权重续训，保留0.86205正式候选不动。
教师缓存继续锁定原始531c4a33...同头baseline的train-only缓存；
新50 epoch baseline的结果不会在本轮中途自动替换教师，以免改变运行含义。

```bash
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --phase all \
  --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_reliable_teacher50.yaml
```

命令入口以 `run --help` 核验后执行；运行PID、完整source SHA和启动命令保存在run记录。
epoch1/每5轮/最终完整测试5000张，检查训练/测试分化、可靠性均值、教师较差样本比例、
alpha、相位更新、专家分布、非有限数及显存释放。只保存best/last，目标0.87不代表已经达到。
