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

## 5. 可复现入口

筛选配置为 `configs/p03_deployment_robustness_screen.yaml`。GPU 推理只能使用
`commands/33_run_p03_deployment_screen.sh`，全部四组结果齐全后由
`commands/34_compare_p03_deployment_screen.sh` 汇总。输出位于
`runs/p03_deployment_robustness_screen/`，不提交 checkpoint 或运行产物到 Git。
