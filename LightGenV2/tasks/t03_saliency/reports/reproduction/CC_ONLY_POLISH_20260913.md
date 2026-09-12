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
