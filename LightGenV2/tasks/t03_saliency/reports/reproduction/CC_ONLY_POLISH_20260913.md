# 真实标签CC目标精修（训练方法对照）

## 为什么做

既有SAM+GT多指标+空间教师CC已到.86205；小batch首轮可至.862396，后续却下降。
跨样本均衡估计首轮.862493也在第5轮回落。因此不能只凭首轮微小提高就宣称持续有效。
之前把GT CC系数提高到3时仍保留KL/SIM/NSS/教师损失，没有实质突破。
本对照检验另一件事：短期将图像监督目标完全聚焦真实标签CC，是否有助于这一个主指标。
不宣称有论文证明它必然有效；也不修改测试指标/标签/数据划分。

## 只改变训练目标

继承`moe_alpha40_extra_control.yaml`，从固定87ad正式权重出发，不从未复评在跑权重出发。
图像监督设为1.5*(1−CC)，KL/SIM/NSS和教师权重均0；**路由均衡、phase DC及既有CCD工作点
正则仍保留原值**，所以总loss并非只含CC。不消除均衡，不解冻Qwen前端，不增加网络。
batch32、SAM .05、EMA .995及各组LR与原control一致；只安排20轮，16轮进入原低LR精修。
仍训练既有光相位/电子残差/相同85412参数读出头，Top2、alpha>=.4、478ROI、224专家、
17微米、10cm、训练DC20–30%与标准clean-eval全部不变。
不做推理后处理、TTA、多模型集成或新纯电子模型。

这不是降低教师权重的同义重复：教师完全撤掉的同时，GT监督也从联合指标改为仅CC。
因此不能把任何收益独立归因于撤教师，必须按“目标精修组合”报告；可能损害KLD/NSS/SIM，
全部指标仍在完整5000张上报告。它也不是证明CC-only从头训练最好。

```bash
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --phase all --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_cc_only_polish_20260913.yaml
```

10000官方train/5000官方val作为public-test，不另划val；第1/5/末轮评估、最高CC选best。
保留best/last，必须注明实际完成/提前停止轮数；保留公开测试反复选模偏差说明。
先通过CPU测试、真实图像更新检查并push源码，再启动；同一助手最多两张GPU。
当前是备选配置，不表示已有新成绩，也不改变冻结Qwen50轮baseline的.87483830记录。

## 实际验证与启动

最初配置仍继承spatial_cc蒸馏模式，被“必须有活跃教师”的配置检查拒绝，未启动正式训练。
修复只把无教师模式选回普通loss路径`distillation.loss: kl`、教师权重仍0；没有放宽保护条件。
最终源码`8dc9be8b44ac4fd6cbfafdd1a7cd85c5c493fd60`通过253项CPU测试（44.95秒，13条既有警告），
已推送GitHub `experiment/salicon-cc-only-20260913`。

CPU真实8图SAM单步通过；它不是正式batch32实验成绩。
冻结前端3933184参数逐值未变、无梯度，原生Transformer调用0；读出头85412参数。
六张相位有有限非零梯度及更新；alpha .43072152/.44106668；教师缓存未加载、KD项0。
保留的正则系数：soft balance .08、hard .1、importance .02、phase DC .005、CCD工作点 .02。
证据`runs/smoke/cc_only_20260913/report.json`含实际学习率、样本ID及各相位更新幅度。

于UTC2026-09-12 20:10:06启动PID/PGID622725，GPU1 UUID
`GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d`，固定worktree `.worktrees/t03_balance`，运行中不得checkout。
与GPU3/PID601787的cross-sample组并行，总计两张自有GPU。
产物`runs/simulation/moe_alpha40_cc_only_polish_20260913_seed42`，
`launch_record.json`记录完整命令、源码与配置SHA；当前是已启动状态，不是完成20轮或新最佳。

## 实际停止结果（2026-09-13，覆盖上文运行状态）

第1/5/10轮完整测试CC为.86177253/.86106870/.86048941，均低于初始化.86205；
训练CC从.87571升至.87809，存在训练提高而公开测试下降的迹象。
因此UTC2026-09-12 20:32:22停止PID622725及全部五个子进程，保留best/last、不删除run。
`manual_stop_report.json`确认全部退出；不是完成20轮，也没有新的性能提升。
本轮不采用CC-only目标，不因主指标目标改变而隐瞒失败。
