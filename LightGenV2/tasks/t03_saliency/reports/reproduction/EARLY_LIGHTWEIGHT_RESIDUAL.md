# 较早起点与论文依据的轻量电子残差试验

## 论文依据及借鉴边界

- Liu et al., **A ConvNet for the 2020s**, CVPR 2022：[论文](https://arxiv.org/abs/2201.03545)、
  [作者代码](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py)。
  只借鉴大核depthwise卷积思路；原作者使用7×7，本轮在既有14×14 token网格上用5×5，
  保留192宽度和192→384→192 MLP。不导入整套ConvNeXt或预训练权重。
- Woo et al., **ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders**, CVPR 2023：
  [论文](https://arxiv.org/abs/2301.00808)、
  [作者GRN代码](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py)、
  [插入位置](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py)。
  按GRN公式独立实现token版本，放在电子MLP的GELU后、投影回192前。
  不引入FCMAE预训练，也不宣称复现整套ConvNeXt V2。

以上论文不是本任务的性能保证，5×5及GRN是否改善SALICON须由对照实验判断。
GRN尝试改善电子通道响应利用率；这不是光Router专家均衡，也不表示我们已证明电子通道坍缩。

## 结构、公式与参数预算

```text
E级输入 [B,196,192]
 → 原局部混合：3×3或5×5 depthwise＋192→192＋内部残差
 → LN → Linear192→384 → GELU
 → 可选GRN384 → 原dropout → Linear384→192＋内部残差
 → 和对应光分支按同尺度(1-alpha)E+alpha O融合
```

GRN按单张图的各通道在196个位置上的L2响应，除以通道响应均值，计算
`y=x+gamma*x*relative_response+beta`。gamma/beta初始为0，保持初始恒等变换。
FP32统计只沿单样本空间轴归约，不跨batch，不构造token×token相似度矩阵，没有Q/K/V或attention。
仅支持没有padding的196 token合同，其他大小直接报错。新归一化在现有电子残差内部，不增加第三支路。

两级GRN新增`2×384×2=1536`参数；两级3→5卷积新增`2×192×(25−9)=6144`，
组合共7680，约占原有效电子参数788234的0.97%（不含冻结Qwen前端）。参数比例不等于时间/性能贡献。
保持光Router Top2、alpha≥.4、20%–30%零级分量、478有效ROI、224专家、30间隔、10cm传播、
两级融合、CCD读出及最终显著性头。像素位置扰动为0，保留既有光学/读出噪声。
不执行Transformer/attention，不加入VGG；教师只用于训练缓存，不进入光模型推理。

## 早期源与对照

源：`runs/simulation/alpha_comparison_20260908/source_checkpoint.pt`。
SHA256：`92bc99ac7c5e999dc4690f3044b118710d928c0b1bfb5419dde01906d59af0da`。
记录epoch=75、CC=0.84879587，位于alpha限制训练及后续refine/generalize之前。
这是较早的历史选定权重，**不是训练第0轮或随机初始化**。
所有组将alpha统一重新编码为.45、约束[.4,1]；因此真正起点CC以各组warmstart复评为准，
不能直接沿用源权重原alpha下的0.84879587。

配置前缀`moe_alpha40_early_`，run后缀`_seed42`：

| 后缀 | 增强 | 电子卷积 | GRN | 比较目的 |
|---|---|---|---|---|
| control | 无 | 3×3 | 无 | 同源同训练配方对照 |
| flip | 同步水平翻转 | 3×3 | 无 | 与control比较增强 |
| kernel5 | 同步水平翻转 | 5×5 | 无 | 与flip比较感受野 |
| grn | 同步水平翻转 | 3×3 | 有 | 与flip比较GRN |
| kernel5_grn | 同步水平翻转 | 5×5 | 有 | 组合交互效果 |

5×5仅将两个旧3×3居中、外圈补0；GRN只允许四个新增gamma/beta张量。
其余key/shape差异拒绝加载。架构后缀为`_ek5`、`_grn`、`_ek5_grn`，旧权重不静默按新结构解释。
测试真实电子模块初始前向一致性、卷积外圈和GRN梯度、单样本独立性、零输入数值稳定性。

## 训练协议

- SALICON train2014全部10000张训练，val2014全部5000张作为public test；无独立validation。
- 同步翻转概率.5，同时变换图像、GT density、fixation、teacher logits；不裁剪或改亮度。
  镜像教师图是等变训练约束，不冒充镜像图重新执行教师；test始终不增强。
- 全组100epoch，1–5冻结electronic组（含输入编码/残差/alpha），相位/Router/CCD读出/头可训练；
  6–70联合退火，71–100精修。
- LR：电子3e-5、相位1e-3、Router2e-4、读出5e-5、头1e-4；
  相位/Router无weight decay，其余.01；EMA=.995；batch32/test48/workers2。
- GT损失KL1+CC1.5+SIM.25−NSS.1；KD从.6线性降到第70轮的.2并保持。
  既有train-only教师缓存与SHA检查不变，不生成测试集教师图。
- 起点完整复评，epoch1/每5epoch/末轮完整测试，最高public-test CC选best；只保存best/last。
  明确有公共测试集选模偏差，不宣称独立盲测。
- 本轮用early control隔离变量；同时与历史最佳0.85812016比较收益，不能只说超过早期源。

## 命令（所选Git工作树根目录）

```bash
conda activate xml
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
TASK=LightGenV2/tasks/t03_saliency
python -m pytest "$TASK/tests" -q
# 各命令在独立终端运行；GPU编号是示例，先检查显存与负载，不停止其他任务。
CUDA_VISIBLE_DEVICES=2 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_early_control.yaml" --phase all
CUDA_VISIBLE_DEVICES=4 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_early_flip.yaml" --phase all
CUDA_VISIBLE_DEVICES=5 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_early_kernel5.yaml" --phase all
CUDA_VISIBLE_DEVICES=6 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_early_grn.yaml" --phase all
CUDA_VISIBLE_DEVICES=6 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_early_kernel5_grn.yaml" --phase all
```

同卡并行仅在显存/负载允许时使用。源码运行于固定Git worktree，数据/缓存共享，结果统一写回原仓库T03
`runs/simulation`。不覆盖历史run，不移动原数据。
完成后读取`selected_checkpoint_test_evaluation.json`、`metrics/training_history.csv`、`best_visualization/`，
报告CC、SIM/NSS、alpha、专家份额和相位变化。本文是协议，不是已取得提升的结果声明。

## 启动核验

代码提交`04204b9c367c0f2af2e3a25000c3a87e1f9de3a4`经31项服务器CPU测试后发布GitHub。
主工作树有其他任务未提交改动，实际运行于`2026OpticsMoE/.worktrees/t03_early_grn`固定提交；
输出仍在原仓库T03 runs。实际GPU分配为control=1、flip=2、kernel5=3、grn=6、kernel5_grn=6。
五组均已完成5000张起点复评，统一alpha=.45后的CC约`0.716815`（组间差异小于1e-6），
这是低光占比早期权重重新适应强光占比的起点，不能用源权重旧alpha下的0.84880冒充。
后续仍需超过历史best 0.85812016才算实际改进；此次启动记录不代表已完成100epoch。
