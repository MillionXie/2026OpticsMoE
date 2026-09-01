# P03 光学部署非理想因素实验

## 1. 要回答的问题

P02 已经在理想数字光学环境中比较唯一四组：NoFT、BP、FA-pretrained、FA-random。P03 不再
训练或选择模型，只冻结 P02 的 best checkpoint，并在推理光路注入部署误差。核心问题是：

1. 在合理的光学非理想因素下，FA-pretrained 是否仍接近 BP？
2. FA-pretrained 是否仍稳定优于同形状的 FA-random？
3. 哪一种物理误差首先破坏性能，四种训练方法的退化斜率是否不同？

这里的“成立”不是要求强噪声下绝对准确率完全不变，而是要求在尚有任务信息的工作区间内，
四组的相对关系和预训练反馈优势仍存在。

## 2. 首轮只研究三类误差

| 因素 | 数字仿真实现 | 首轮强度 | 物理含义 |
|---|---|---|---|
| 相位标定误差 | 每层相位掩模加入一次固定的逐像素高斯偏差 | 0.05、0.15 rad | SLM/相位板标定、量化与器件误差 |
| 横向失配 | 每层掩模按固定随机方向平移，边界不循环回卷 | 1、2 pixel | 约 16、32 μm 横向装调偏差 |
| 探测器噪声 | 每次 CCD 强度读出加入相对该图 RMS 的高斯噪声并截断到非负 | 1%、5% | O/E/O 读出噪声和有限 SNR |

另设 `combined_moderate = 0.15 rad + 2 pixel + 5% detector noise`，检查误差叠加。每次部署
运行中的相位误差和位移保持静态；探测器噪声随 batch 变化。所有方法共享同一个 deployment
seed，因此比较是配对的。

暂不在首轮加入散射、像差、波长漂移、距离误差、像素串扰和相位量化。它们应在确认当前三类
接口和效应量后分批加入，避免一次扫描过多因素而无法解释。

## 3. 两阶段协议

### P03-S：验证集筛选（本轮）

- 冻结 P02 四组 seed 2026 best checkpoint；
- 使用 validation split，不提前消耗最终 test robustness 结论；
- deployment seed 固定为 9101；
- 运行 ideal、六个单因素条件和一个联合条件；
- ideal accuracy 必须与 P02 checkpoint 的 validation accuracy 完全一致，否则实现无效；
- 只看退化曲线和方法排序，不据此修改 checkpoint 或模型。

进入正式复验的最低条件：没有数值错误；误差强度产生可分辨但未全部坍缩到随机水平的曲线；
FA-pretrained 在单因素条件下仍优于 FA-random。联合强条件允许出现排序压缩或性能坍缩，必须
原样报告。

### P03-F：测试集确认（P03-S 通过后）

- 锁定 P03-S 的条件，不依据 test 结果增删强度；
- 使用 P02 training seeds 2026/2027/2028；
- 使用 deployment seeds 9101/9102/9103；
- 报告 accuracy mean/std、相对 ideal 的绝对下降和保留率；
- 逐条件报告配对的 `BP - FA-pretrained` 与 `FA-pretrained - FA-random`。

## 4. 图表结构

最终只需要三类主图，不增加算法组：

1. 三个单因素的 accuracy--severity 曲线，颜色始终对应四种正式方法；
2. 联合扰动下的 accuracy 和相对 ideal 下降柱状图；
3. `BP - FA-pretrained`、`FA-pretrained - FA-random` 的配对差值及置信区间。

补充材料记录每层实际位移、相位误差实际 RMS、部署 seed、checkpoint SHA-256 和全部逐运行数据。

## 5. P03-S1 首轮观察与亚像素补充

P03-S1 seed 2026 验证集结果表明，0.05/0.15 rad 相位误差和 1%/5% 探测器噪声位于有效
工作区间；FA-pretrained 在这四个条件中均保持高于 FA-random。相反，1 pixel 已使 BP、
FA-pretrained 接近随机水平，因此 1/2 pixel 不适合作为最终失配曲线的低/中强度点。这一负
结果保留在原输出中，不删除也不重定义。

在查看 test robustness 之前，增加 validation-only 的 P03-S2 亚像素筛选，配置为
`configs/p03b_deployment_subpixel_screen.yaml`，位移为 0.125/0.25/0.5/0.75 pixel（约
2/4/8/12 μm）。亚像素平移对复相位 phasor 的实部/虚部做双线性采样，再恢复相位，避免直接
插值 0/2π 包裹角产生伪误差。P03-F 将依据 S1/S2 的 validation 曲线冻结失配强度。

## 6. 可复现入口

筛选配置为 `configs/p03_deployment_robustness_screen.yaml`。GPU 推理只能使用
`commands/33_run_p03_deployment_screen.sh`，全部四组结果齐全后由
`commands/34_compare_p03_deployment_screen.sh` 汇总。输出位于
`runs/p03_deployment_robustness_screen/`，不提交 checkpoint 或运行产物到 Git。

## 7. 筛选结果与正式冻结

P03-S1 的四组 validation accuracy（NoFT/BP/FA-pretrained/FA-random）为：

| 条件 | NoFT | BP | FA-pretrained | FA-random |
|---|---:|---:|---:|---:|
| ideal | 50.42% | 73.40% | 72.68% | 62.96% |
| phase 0.05 rad | 50.00% | 72.94% | 72.00% | 62.80% |
| phase 0.15 rad | 42.54% | 69.88% | 67.64% | 61.20% |
| detector 1% RMS | 50.42% | 73.32% | 72.54% | 63.20% |
| detector 5% RMS | 50.22% | 73.08% | 72.04% | 62.98% |
| shift 1 pixel | 16.72% | 14.84% | 13.68% | 23.04% |
| shift 2 pixel | 13.04% | 16.86% | 15.40% | 19.58% |

P03-S2 显示 0.125 pixel 时 BP/FA-pretrained/FA-random 为 62.20%/60.04%/56.22%，排序仍
成立；0.25 pixel 时为 29.66%/27.42%/37.36%，说明约 2--4 μm 之间存在明显失配拐点。
0.5 pixel 后三种可训练方法均接近随机水平。

在查看 test robustness 前冻结 P03-F：phase 0.05/0.15 rad、shift
0.0625/0.125/0.25 pixel、detector 1%/5%，以及 phase 0.15 rad + shift 0.0625 pixel +
detector 5% 的联合工作点。正式配置使用三个 training seeds 和三个 deployment seeds；0.25 pixel
明确标记为失效边界，不用于支撑“关系成立”的主结论。配置为
`configs/p03_deployment_robustness_formal.yaml`，只能由 commands 35/36 运行和汇总。

## 8. P03-F 正式结果

36 个 checkpoint×deployment-seed 结果全部完成。下表先对每个 training seed 的三个 deployment
seeds 求均值，再报告三个 training seeds 的 mean ± sample std，避免把九个相关运行直接当成九个
独立模型：

| test 条件 | NoFT | BP | FA-pretrained | FA-random |
|---|---:|---:|---:|---:|
| ideal | 51.18% ± 0.00 | 72.30% ± 0.54 | 71.44% ± 0.42 | 63.29% ± 0.91 |
| phase 0.05 rad | 50.76% ± 0.00 | 72.05% ± 0.79 | 70.98% ± 0.58 | 63.28% ± 0.93 |
| phase 0.15 rad | 41.66% ± 0.00 | 68.55% ± 0.98 | 66.52% ± 1.23 | 61.83% ± 0.98 |
| shift 0.0625 pixel | 49.04% ± 0.00 | 69.99% ± 0.59 | 68.67% ± 0.40 | 61.61% ± 1.08 |
| shift 0.125 pixel | 41.34% ± 0.00 | 62.00% ± 0.77 | 59.44% ± 0.58 | 56.09% ± 0.71 |
| shift 0.25 pixel（边界） | 22.90% ± 0.00 | 30.91% ± 1.80 | 28.45% ± 1.92 | 35.81% ± 0.48 |
| detector 1% RMS | 51.08% ± 0.00 | 72.33% ± 0.57 | 71.37% ± 0.45 | 63.32% ± 0.95 |
| detector 5% RMS | 50.96% ± 0.00 | 72.25% ± 0.76 | 71.16% ± 0.55 | 63.27% ± 0.89 |
| combined operating | 41.12% ± 0.00 | 65.92% ± 0.58 | 63.92% ± 0.40 | 59.72% ± 1.16 |

关键配对差值（同样先平均 deployment seeds）：

| 条件 | BP - FA-pretrained | FA-pretrained - FA-random |
|---|---:|---:|
| ideal | 0.86 ± 0.26 pp | 8.15 ± 1.32 pp |
| phase 0.15 rad | 2.03 ± 0.31 pp | 4.69 ± 2.20 pp |
| shift 0.0625 pixel | 1.32 ± 0.19 pp | 7.06 ± 1.45 pp |
| shift 0.125 pixel | 2.56 ± 0.19 pp | 3.35 ± 1.17 pp |
| detector 5% RMS | 1.08 ± 0.29 pp | 7.89 ± 1.43 pp |
| combined operating | 2.00 ± 0.23 pp | 4.20 ± 1.56 pp |
| shift 0.25 pixel（边界） | 2.46 ± 0.20 pp | **-7.36 ± 2.38 pp** |

P03-F 支持：在仍有可用任务信息的相位误差、探测器噪声、0.0625/0.125 pixel 失配和联合工作
点内，FA-pretrained 仍接近 BP 且优于 FA-random。它不支持“任意部署误差下都成立”：0.25
pixel 已破坏高性能光学表征并导致相对排序反转。当前探测器结果还只代表经过每层标准化的相对
高斯读出噪声，不能替代绝对光功率、shot noise、量化、饱和和动态范围实验。
