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

## 已核验启动记录

源码 `a032aefc4035de159c8845a9170d4bc5bb0d708e`，服务器xml环境112项T03测试通过
（24.56秒，13项既有matplotlib弃用警告），测试与正式运行前已push GitHub。
0/1号4090各自完成CUDA矩阵运算与有限值检查，测试进程退出后显存回到12/25MiB。
正式两组均batch32、评估batch48、workers2，初始化SHA与上文一致，教师缓存存在。
服务器worktree为 `/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t03_sam`，
产物通过既有runs链接保存在仓库主目录的T03 `runs/simulation/`。
联合续训PID3559541（GPU0），教师预热PID3560835（GPU1）；PID仅为当次运行身份，
后续清理前必须同时核对命令、启动时间与路径，不能因PID复用误杀其他作业。
各run的 `console.log`、`run_manifest.json` 和 `metrics/training_history.csv` 为实时进展证据。
此记录仅确认启动，不代表完成60轮或取得新性能。

## 完成结果（2026-09-10 15:15 CST核查）

两组均完成60/60轮，`stop_reason=epoch_budget`，没有提前中断；重载best完整5000张结果：

| 方案 | best epoch | 重载best CC | 最后epoch CC | best alpha |
|---|---:|---:|---:|---|
| 联合续训 | 1 EMA | .8620965302 | .8591156996 | .43061742 / .44101283 |
| 教师预热15轮 | 0（保留来源） | .8620496944 | .8590113702 | .43072182 / .44106704 |

联合组仅比来源提高约.000047，尚无独立float64配对复评，**不据此替换已独立核验的87ad候选**。
预热组训练结束没有超过初始化；其best相位变化为0是因为选中epoch0，不是训练未更新相位。
联合组四专家选择2351/2622/2299/2728，有效专家3.9794，没有明显坍缩。
两组后期CC下降，不继续沿同数据高学习率预热路线扩大预算；转向保留GT的额外图像辅助训练。
两进程已终止，GPU0/1恢复12/25MiB；PID3559541/3560835当时为无CUDA资源的defunct条目，
并不表示还在训练，不进行GPU reset或杀其他任务。

SHA256（依次为best / training_report / selected_checkpoint_test_evaluation）：

- 联合：`35ee4b4a88cd3526450a1776db365acc7997380b1c937e30477ff7b4f3d77f22`
  / `dfc462ac0b244ab06ec2f0c7c32ccb70ac11612fc3d9c999f13648936f284f59`
  / `51ab9aef2c28652a04ac257d0d0b65011a749f9f11533e57d4c0f5a99c5c79e8`。
- 预热：`695a6b21e3876a71a9fe53cb805ae285986e9838e624f180628ae8b08bab48ab`
  / `97548f880151f36b7b8519567c5268d60366d063724e664cce58def233d3cda5`
  / `9be5a2e1fd09048394954ebcb940c80743e9ee6f8110e2230d62d06758cbe2b8`。

预热best是重新保存的checkpoint容器，文件SHA与来源不同；不能据容器SHA判断张量发生更新。
公开测试选模偏差仍存在，目标.88仍未达到。
