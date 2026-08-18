# 指标定义与读取规则

## 统一规则

- `↑` 表示越大越好，`↓` 表示越小越好。
- 所有当前结果均为单次运行，不含标准差或置信区间。
- 训练 history 是逐 epoch 原值；不得从曲线图片反推数值，应直接读取 CSV/JSON。

## SALICON 显著性

| 字段 | 含义 | 方向 |
|---|---|---:|
| `validation_kld` | 目标 fixation density 到预测 density 的 KL divergence | ↓ |
| `validation_cc` | 预测图与目标 density map 的逐样本 Pearson correlation 后取均值 | ↑ |
| `validation_sim` | 两张归一化 density map 的 histogram intersection | ↑ |
| `validation_nss` | 在 fixation 像素上读取预测显著图 z-score 后取均值 | ↑ |
| `validation_auc_judd` | 以 fixation/non-fixation 像素构造 ROC 的 AUC-Judd | ↑ |
| `validation_mae` | 预测与目标分别除以自身最大值后的像素 MAE | ↓ |

正式 checkpoint 按 `validation_cc` 最大选择，不要再沿用旧 SALICON 文档中按 AUC-Judd 选择的口径。

## ISIC 2016 病灶分割

所有二值指标使用 sigmoid probability `≥0.5` 作为病灶预测。

| 字段 | 含义 | 聚合方式 | 方向 |
|---|---|---|---:|
| `mean_iou` | Jaccard / intersection-over-union | 每张图计算后平均 | ↑ |
| `mean_dice` / `mean_f1` | Dice coefficient | 每张图计算后平均 | ↑ |
| `mae` | probability map 与二值 mask 的平均绝对误差 | 全部像素 | ↓ |
| `pixel_accuracy` | 像素分类正确率 | 全部像素 | ↑ |
| `sensitivity` | TP / (TP + FN) | 全 test 像素汇总 | ↑ |
| `specificity` | TN / (TN + FP) | 全 test 像素汇总 | ↑ |
| `balanced_pixel_accuracy` | 0.5 × (sensitivity + specificity) | 全 test 像素汇总 | ↑ |

`test_predictions.csv` 保存 379 张图的逐样本 IoU、Dice、MAE、sensitivity 和 specificity，可用于后续分布图、失败样本筛选或 bootstrap；`test_metrics.json` 是正式整体结果。

## LSP 人体姿态

| 字段 | 含义 | 方向 |
|---|---|---:|
| `test_pck_at_0.2_torso` | 预测误差不超过 `0.2 × torso scale` 的关节点比例 | ↑ |
| `test_pckh_at_0.5_head` | 预测误差不超过 `0.5 × head scale` 的关节点比例 | ↑ |
| `test_mean_pixel_error` | 224×224 输入坐标系中的平均关键点像素误差 | ↓ |
| `test_normalized_mean_error_torso` | 像素误差除以 torso scale 后的均值 | ↓ |
| `train_router_entropy` | 对 4 expert 概率熵除以 `log(4)` 的单样本归一化均值 | 诊断量 |

这里的 torso scale 是两条 shoulder-to-opposite-hip 对角距离的均值；head scale 是 `2 × distance(neck, head_top)`。

低 router entropy 只说明每个样本的路由概率尖锐，不能单独证明所有样本集中到同一个 expert。判断全局负载是否塌缩还需要 expert selection frequency。

## Caltech101-10 图像检索

测试流程不是 10 类分类 head，而是 prototype retrieval：每类 3 张独立 gallery 图先通过同一个网络得到 64-D L2-normalized embedding，再取均值得到一个 class prototype；200 张 test query 分别与 10 个 prototype 计算相似度并排序。

| 字段 | 含义 | 方向 |
|---|---|---:|
| `test_top1` | 正确类别 prototype 排名第 1 的 query 比例 | ↑ |
| `test_top3` | 正确类别 prototype 位于前 3 的 query 比例 | ↑ |
| `test_mrr` | 正确类别排名倒数 `1/rank` 的 query 均值 | ↑ |
| `ema_test_*` | 使用参数 EMA checkpoint 计算的对应指标 | ↑ |
| `train_top1/top3/mrr` | 每个训练 batch 内随机抽 support 建 prototype、其余样本作 query 的 episodic 指标 | 训练诊断 |
| `total_loss` | supervised contrastive、gallery/prototype CE 与加权 CCD operating-point loss 的和 | ↓ |
| `phase_grad_rms` | phase 参数在反传中收到的 RMS 梯度 | 诊断量 |
| `phase_delta_run_rms_rad` | phase 相对 run 初始化值的 RMS 位移（rad） | 诊断量 |

术语关系：

- `support`：只在训练 episode 内临时抽出的同类参考样本，用于构造 batch prototype；
- `gallery`：评估时固定的参考库，本次为 10 类 × 3 张；
- `query`：待检索样本，本次 test 为 10 类 × 20 张；
- 正确的 gallery 类别排名越靠前，Top-k 和 MRR 越高。

正式 checkpoint 为 `total_loss` 最小的 epoch 54，raw Top-1 `0.88500`，同 epoch EMA Top-1 `0.89000`。虽然 `test_*` 有完整逐轮记录，但它们没有参与该 checkpoint 选择；观察到的 raw Top-1 最大值 `0.89000` 只能作为 monitored-test 诊断。另需同时报告 `phase_learning_rate=0` 和 `phase_delta_run_rms_rad=0`，否则“光学 phase 可训练”的表述会与证据不符。
