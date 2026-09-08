# 同读出头与alpha约束对照（2026-09-08）

## 已有实测证据

- 原光电CC：0.8290598。
- mean_only续训候选：epoch75最佳CC=0.84879587（检查时训练尚未结束）。
- CC损失权重加倍候选：约0.8476，尚未优于原权重。
- 原Qwen头固定权重复评：0.8810325。
- 原Qwen头重新初始化训练30epoch：0.87899109，best epoch30；属于新的单seed训练复现，非多seed统计。
  原始证据：[重训复评JSON](evidence/baseline_retrain_20260908.json)。

这些数值均为同一5000张公开test上的CC，有测试选模偏差。新alpha对照尚无最终成绩。

## 读出头公平性：适配器不能漏算

旧Qwen的230257参数头包含1024维输入的投影。我们的85412参数解码头之前，
还有 `Linear(1024,192)+LayerNorm(192)`，197184参数。两者不能只按23万对8.5万判断全部电子容量。

新增 `aligned_baseline.py`，冻结完整Qwen视觉主干后使用与我们同规格的适配器及完全同类的解码头：

| 部分 | 光电版 | 新Qwen对照 |
|---|---:|---:|
| Linear1024→192 + LN192 | 197184 | 197184 |
| SaliencyDensityDecoder192→224×224 | 85412 | 85412 |
| 上述合计 | 282596 | 282596 |

没有将baseline全部读出参数声称为8.5万。适配器在光电版位于光电主体之前，在Qwen版位于冻结主体之后；
光电版还训练两个残差mixer和光学读出等，因此这也**不代表总可训练参数完全相同**。
旧230257参数强baseline保留为独立参照，新头不能沿用旧0.8810成绩，必须重训。
新对照和alpha两组使用相同数据、batch64、100epoch、每5epoch测试、相同损失及相同适配器/解码头学习率调度。
Qwen新头随机初始化；光电从已有任务权重继续训练，历史训练预算并不相同，论文须披露。

## alpha对照

两组采用同一份不可变source checkpoint，SHA256：
`92bc99ac7c5e999dc4690f3044b118710d928c0b1bfb5419dde01906d59af0da`。
路径 `runs/simulation/alpha_comparison_20260908/source_checkpoint.pt`。

- free：alpha= sigmoid(raw)，无额外下限，范围[0,1]。
- ge040：alpha=0.4+0.6 sigmoid(raw)，两层在训练和推理时均不低于0.4。
- 两组都重新编码为**实际alpha=0.45**，不能直接复制旧raw logit后按新区间解释。
- 除alpha范围和输出目录外，两份配置完全相同；checkpoint架构标签包含区间，避免误加载。

两层仍为同尺度凸融合，有光router Top2、DC20%–30%、无像素位移、mean_only CCD预处理。
不更改光路、ROI、专家大小；不增加attention、Transformer或额外电子分支。

## 三阶段训练

1. epoch1–10：电子组学习率0，固定电子残差和alpha；训练光相位、router、光学读出和解码头。
2. epoch11–70：联合训练，独立组学习率余弦降低；硬Top2均衡权重0.5逐渐降到0.1，软均衡保留。
3. epoch71–100：低学习率精修；不额外做测试集拟合、预测校准或推理增强。

仅保留best/last，每个epoch记录两层实际alpha、阶段、各组学习率。原来两个低alpha续训任务继续完成，
不删除或覆写它们。alpha≥0.4可能牺牲CC，不能保证高比例与性能一定兼得。

## 命令（仓库根目录）

```bash
conda activate xml
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# 以下依次为三种任务；并行时为每个进程分别设置空闲GPU的CUDA_VISIBLE_DEVICES。
python -m LightGenV2.tasks.t03_saliency.aligned_baseline --run-dir LightGenV2/tasks/t03_saliency/runs/simulation/qwen_aligned_head_staged_seed42
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_staged_alpha_free.yaml --phase all
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_staged_alpha_ge040.yaml --phase all
```

运行需要既有SALICON图像/标注、本地Qwen快照和上述source checkpoint。移机时先校验source SHA，
再更改本机数据/模型路径；禁止引用仍在不断覆盖的另一个run的best作为两组的共同初始权重。
正式日志位于 `runs/simulation/alpha_comparison_20260908/`；配置命名对应最终run。
最终查看 `selected_checkpoint_test_evaluation.json`，结合CC、NSS、SIM、两层alpha和专家选择占比选候选。

## 启动核验

三个任务已启动：free在物理GPU0，ge040与新Qwen对照共享空闲A100（物理GPU6）；不停止其他人的进程。
两组alpha候选epoch0均为CC=0.716815，实际alpha均为0.45。相同起点核验通过，
但比低alpha权重的0.8488低很多；这是权重比例直接改变后的分布变化，不是100epoch训练后的结果。
新Qwen对照已进入第一轮反向训练，不能填写旧头的0.8810作为其成绩。
