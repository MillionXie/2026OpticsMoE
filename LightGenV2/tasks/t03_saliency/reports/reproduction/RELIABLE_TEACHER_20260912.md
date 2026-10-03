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

## 启动检查

代码 `d8653fa6057f96b52d53259999343ff2117c1526`，225项CPU测试通过（42.58秒，13条既有警告），
推送 `experiment/salicon-reliable-teacher-20260912` 后启动。
真实train前4图进行前后向检查：总任务损失.446589，可靠性权重均值.82234，
所有梯度有限，光router/四位专家/全局相位的梯度均非零；本次检查没有optimizer更新或权重写入。
85412参数头不变，初始alpha .431023/.441234。

正式PID/PGID414381，GPU1 `GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d`；
工作树 `.worktrees/t03_baseline50_20260912`，结果落到主工程T03的
`runs/simulation/moe_alpha40_reliable_teacher50_20260912_seed42`。
与GPU0的50轮Qwen同头重训同时运行，总共两卡，不触碰GPU4/6的其他用户任务。
启动不等于已经获得新的CC成绩。

## 已提前停止（2026-09-13）

第1/5/10/15轮完整公开测试CC分别为.86172698/.86194520/.86161460/.86138080，
训练CC继续升高而测试回落。未超过正式候选.86204960，因此停止，不把它包装成提升。
保留第5轮EMA best和第16轮last；不是完成50轮，不删除权重或数据。
停止前验证PID414381的UID、PGID、工作目录及命令，只终止该进程组；
停止后父进程、5个子进程及GPU1显存分配均确认消失。运行目录的`manual_stop_report.json`记录原因。

中途CPU核查`state_audit_midrun_20260913.json`确认：head85412参数、core/head键和形状不变、
六张相位合计479364参数且已发生更新、alpha .430724/.441071、权重有限。
这份状态审计不能单独证明运行图合规或泛化性能提高；后者以真实forward与完整评估为准。

## 后续梯度冲突诊断（不新增训练方法）

在正式87ad权重、训练模式下，抽取32张train图、四个batch8，分别对真实标签主损失
（含既有router正则）与固定train-only CC-KD2求完整可训练参数梯度，没有optimizer更新。
原始身份及结果：`runs/smoke/teacher_gradient_conflict_20260913/report.json`，seed20260913。
四批梯度余弦为+.56985/+.64170/+.27502/+.33016，没有负向冲突。
这不是全训练集统计，也不排除个别样本冲突；但不支持现在贸然增加PCGrad式投影训练。
因此仅作诊断，未加入投影代码、额外分支或启动相应试验。
参考机制：[Gradient Surgery for Multi-Task Learning, NeurIPS2020](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)。
其负梯度内积触发投影是多任务方法，不是已验证的SALICON光学优化结论。
