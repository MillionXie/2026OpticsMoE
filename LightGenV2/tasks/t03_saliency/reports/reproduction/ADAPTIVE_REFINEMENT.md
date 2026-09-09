# SALICON：平台期后的受控精修

## 起点与动机

2026-09-09早期五组在15–30轮后停止改善。截至20:14，control、kernel5已完成100轮，
flip为92轮，两组GRN为76轮；各自最好CC依次为0.85051669、0.85025527、
0.85016260、0.84995152、0.85007239。旧日志和best/last全部保留。
新试验从`moe_alpha40_generalize_kd060_seed42/best_checkpoint.pt`开始，
CC=0.85812016，SHA256=`36333e6ba013cac3a2801bce4babcc2fb5cd36adf02969e019437b40ceebaffd`。
不重置alpha；不是早期alpha重设后的低分起点。

## 三个假设，不增加推理模块

| config后缀 | KD系数 | GT CC损失系数 | 目的 |
|---|---|---|---|
| keepkd | 固定0.6 | 1.5 | 小步长、维持教师约束的对照 |
| releasekd | 第1轮0.6→第10轮0，之后0 | 1.5 | 检验教师是否限制学生 |
| releasekd_cc | 同上 | 3.0 | 释放教师后加强真实显著图相关性目标 |

第二组相对第一组仅改变KD；第三组相对第二组仅改变GT CC系数。不保证这些方法有效。
三组结构相同：冻结Qwen视觉输入端、原3×3电子残差、光Router Top2、同尺度融合、
alpha≥0.4、20%–30%零级分量，原光学ROI与传播次数不变，不加GRN/Transformer/attention。
原有KL/SIM/NSS项、相位及Router正则、teacher train缓存和SHA约束不变。
本轮无增强；历史起点曾接受增强训练。

## 训练与自动调整

- 最多30轮；LR：电子3e-6、相位5e-5、Router1e-5、CCD读出5e-6、头1e-5。
- 原分阶段调度保留，warmup=0、polish_start=21；EMA=.995，batch32/test48/workers2。
- 10000张train，5000张官方val作为public test，无独立validation。
- 起点、epoch1、每2轮、最后一轮完整测试；最高CC保存best，未超越则保留起点。
- 自第6轮起，连续3次完整测试未超过控制器历史最好CC至少0.0001，后续LR乘0.5。
  两次降速后再出现同样平台则结束。改善重置等待次数，但不恢复已降低的倍率。
  每轮先重新计算原退火LR，再乘持久倍率，避免意外指数衰减。
- 阈值只用于调速，best保存仍采用严格最大CC，不丢弃小幅改进权重。
- 停止前先保存当轮last和指标，再执行best可视化、完整复评及Router审计。
- `metrics/adaptive_events.json`记录测试、动作和后续倍率；训练历史记录实际LR。
  `training_report.json`记录实际轮数和停止原因。仅保存best/last，不生成周期PT。
- 公共测试集参与选模、调速、早停，存在选择偏差，不称为独立盲测。
  自动控制只降LR/早停，不会自行修改网络或启动任意新试验。

## 操作命令

在已同步Git提交的仓库根目录执行，GPU编号按实时负载选择，不停止他人进程：

```bash
conda activate xml
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
python -m pytest LightGenV2/tasks/t03_saliency/tests -q
# 三个独立终端；GPU编号为示例
CUDA_VISIBLE_DEVICES=1 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_adaptive_keepkd.yaml --phase all
CUDA_VISIBLE_DEVICES=3 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_adaptive_releasekd.yaml --phase all
CUDA_VISIBLE_DEVICES=5 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_adaptive_releasekd_cc.yaml --phase all
```

结果在任务`runs/simulation/moe_alpha40_adaptive_<后缀>_seed42`，源码commit、实际配置、
命令、环境由run记录。勿将命令覆盖到已有run。正式结论读取
`selected_checkpoint_test_evaluation.json`并与0.85812016、同规格头Qwen的0.88968476比较。
本文是训练协议，不是已取得提升的结果声明。

## 2026-09-10：降低硬均衡压力的单变量对照

新增`moe_alpha40_soften_hard_balance.yaml`，对照为`moe_alpha40_hint_control.yaml`。
同一起点CC=0.85812016、相同50轮/学习率/EMA/KD/无增强训练；不启用feature hint。
仅将硬Top-2均衡的初始/最终系数从0.10/0.10降为0.01/0.01，仍为正数；
软均衡0.08、importance 0.02、光学噪声和所有推理结构均不变。
这不是取消均衡，也不是放宽alpha下限。

动机来自只读训练梯度诊断（源码`b3f88069`，源checkpoint SHA
`36333e6ba013cac3a2801bce4babcc2fb5cd36adf02969e019437b40ceebaffd`）：
CPU float32、torch seed42、无增强、训练模式和光学扰动开启，
以`random.Random(1042).sample(range(10000),32)`固定抽取一个训练batch。
其光router任务梯度L2=0.00150178690；诊断中显式按0.50求取的硬均衡梯度L2=0.00785553450，
但**当前profile经继承解析后实际硬均衡系数为0.10，不是公共默认0.50**。
按损失系数线性换算，实际硬均衡梯度L2=0.00157110690，约为任务梯度1.05倍，
两者cos=-0.01415；软均衡+importance梯度L2=0.00148858965。
不得把0.50的敏感性诊断写成当前训练的实际权重或声称实际硬均衡占主导。
梯度按`autograd.grad`分别求取，各参数组拼接后转float64计算L2和cos；没有optimizer step。
这只是一批训练数据上的局部诊断，不证明整个数据集都存在梯度冲突，也不代表性能改善。

```bash
# 使用有足够余量的GPU；目标run目录必须尚不存在。
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_soften_hard_balance.yaml --phase all
```

以完整5000张public-test CC与原控制组比较；仍披露测试选模偏差。
正式候选必须同时查看`selected_checkpoint_test_evaluation.json`中的专家选择占比、
有效专家数和未使用专家，不以性能小涨为由接受明显专家坍缩。
目标CC≥0.87尚未达成；本节是受控试验协议。
