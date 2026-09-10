# 教师监督预热：训练方法对照（2026-09-10）

目标为完整5000张public-test CC≥0.88，尚未达到。当前已完成且独立核验best为
0.86204960；不得将新启动或训练集指标写成正式提升。

## 两组唯一差异

| 配置 / run（均加 `_seed42`） | 1–15 epoch | 16–60 epoch |
|---|---|---|
| `moe_alpha40_reheat_joint` | GT＋空间CC教师损失 | GT＋空间CC教师损失 |
| `moe_alpha40_teacher_pretrain15` | 仅空间CC教师任务监督 | 恢复GT＋空间CC教师损失 |

这里“仅教师”不移除router均衡、相位DC等物理正则。预热只把GT的KL/CC/SIM/NSS
四个系数临时置0，不修改GT本身；日志仍计算GT指标，但梯度不受GT标签影响。
第16轮恢复原权重1/1.5/.25/.1，优化器连续运行、不重置。两组相同60轮、SAM .05、
EMA、batch、学习率、随机种子、训练数据与初始化；每轮日志记录实际监督阶段和GT系数。

来源为完成best `moe_alpha40_sam_spatialcc_kd2_seed42/best_checkpoint.pt`，SHA256：
`87ad4db51e3f58f9a41d6df09092439e5a09e81e93f88a3bfe2d3d008fafb29a`。
新建优化器，是warmstart，不声称从旧optimizer断点精确恢复。
两组共同重新加热：E=3e-5，相位=5e-4，router=5e-5，CCD读出=3e-5，头=5e-5，
CFFN空间项=2e-4；1–45轮余弦下降，46–60轮小步精修。教师系数2恒定，关闭额外图像增强，
不启用早停，避免教师预热尚未切回GT即被停止。历史最佳在epoch0保留，失败不覆盖原run。

## 原理与边界

参考[分阶段知识蒸馏研究](https://arxiv.org/abs/1812.01819)将知识迁移与任务学习拆开的思路。
**本试验不是该论文SSKD复现**：该论文讨论骨干特征迁移后任务头学习；这里仅在同一SALICON
训练集上先拟合教师输出，再联合优化已有学生与任务头。是否有效需要配对结果，不承诺提升。
这也不是额外数据预训练；没有新增数据集、没有用5000张测试图产生训练梯度。

训练使用10000张train2014；val2014的5000张按既有协议作为public-test，每5轮、首轮、末轮
和初始化测试，按CC最高选best，平分保留较早权重。该选模有选择偏差，不宣称独立未见测试泛化。
教师缓存仅train，SHA `a45a90fe1dc029961464304373473d1271594128fc7b7f2ac80775f8638dd60e`，
来自同规格头Qwen教师；数据、sigma19标签和归一化规则均不变。

冻结Qwen patch/position前端不变，无新增Transformer/attention/VGG；现有两条光电支路、
光Router Top2、478总ROI、224专家、17μm采样、10cm传播、alpha≥0.4与同尺度凸融合均不变。
训练继续20%–30%随机相干零级；标准eval关闭随机光扰动，不能把仿真CC当作实测鲁棒性分数。
没有新增推理参数，光相位仍参与梯度更新。只保存best/last，不恢复每5轮PT。

## 启动与资源

先确认两张卡确实空闲；本轮预选0、1号4090，不碰其他用户进程。每卡一个任务，
同时最多两卡（含诊断），不自动补开第三卡。下列命令在两个终端分别执行；运行前checkout
已测试并推送GitHub的同一commit。run自动记录commit/config/environment与教师SHA。

```bash
# 终端1：仓库根目录，xml环境
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_reheat_joint.yaml --phase all

# 终端2：同一个源码版本
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_teacher_pretrain15.yaml --phase all
```

完成/失败后检查进程退出及对应显存释放；不kill未知PID、不执行GPU reset。
中断run不自动续写。最终结果以training_report及重载best完整复评为准，候选还需独立float64
复评、alpha/路由均衡/相位变化审计；保留原best及历史对照。
