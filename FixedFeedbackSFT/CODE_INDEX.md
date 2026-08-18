# 代码与产物索引

## V1：已完成分类实验

根目录：

`experiments/d2nn_cifar100c10_fixed_feedback_20stage400`

| 文件 | 作用 |
|---|---|
| `optics.py` | 自定义 fixed-feedback autograd、传播、OEO、残差 |
| `model.py` | 20-stage optical classifier 与 feedback mode 配置 |
| `training.py` | pretrain、三种微调、NoFT、checkpoint 和诊断 |
| `analysis.py` | 参数漂移、BP endpoint cosine 和聚合 |
| `publication_report.py` | 结果表、图和 source data 生成 |
| `visualization.py` | 相位、光场和训练可视化 |
| `datasets.py` | CIFAR-100/CIFAR-100-C 划分和固定顺序 |
| `configs/main.yaml` | 正式配置 |
| `commands/` | 分组启动脚本 |
| `tests/test_core.py` | forward 一致性、反馈梯度和 buffer 测试 |

正式结果：

`experiments/d2nn_cifar100c10_fixed_feedback_20stage400/results/main`

关键 source data：

- `source_data/aggregate_performance.csv`
- `source_data/aggregate_geometry.csv`
- `source_data/checkpoint_performance.csv`
- `source_data/endpoint_geometry.csv`
- `source_data/training_trajectories.csv`

## V2：对比学习迁移实验

根目录：

`experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400`

| 文件 | 作用 |
|---|---|
| `optics.py` | 与 V1 同定义的 fixed optical connector |
| `model.py` | 20-stage backbone + signed 128D embedding readout |
| `losses.py` | SupCon 与 leave-one-out prototype loss |
| `datasets.py` | CIFAR-100/CIFAR-10 split 与 balanced P x K sampler |
| `training.py` | pretrain、prototype evaluation、三种微调、诊断 |
| `analysis.py` | matched endpoint 与 final/validation-selected 汇总 |
| `configs/main.yaml` | 正式配置 |
| `commands/COMMANDS.md` | 全部运行命令 |
| `tests/test_core.py` | 9 项核心测试 |

服务器正式结果目录：

`experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/runs/main`

关键结果：

- `comparison/aggregate.csv`
- `comparison/task_metrics.csv`
- `comparison/endpoint_geometry.csv`
- `comparison/comparison.json`
- `finetune/*/seed_*/training_history.csv`
- `finetune/*/seed_*/gradient_diagnostics.csv`

专题级只读验证入口：`FixedFeedbackSFT/commands/01_verify_v2_results.sh`。

## 固定反馈核心代码位置

V2 中最关键的实现：

- `_FixedFeedbackOptical.forward/backward`：`optics.py`
- `OpticalOEOStage.set_feedback`：`optics.py`
- `OpticalEmbeddingNetwork.configure_feedback`：`model.py`
- pretrained phase snapshot：`model.py::snapshot_feedback_phases`
- 当前 vs BP 瞬时梯度诊断：`training.py`
- matched endpoint 累计更新比较：`analysis.py`

## 维护规则

- 不覆盖 V1 的 config、checkpoint 或结果；
- V2 正式实验也应保留 resolved config 和 digest；
- 修改 fixed-feedback 数学定义时新建实验/方法名；
- 禁止把 sample-specific error field 存下来反复使用；
- 禁止让 FA 方法的 forward 使用 frozen phase；
- 结果文档必须注明 checkpoint policy。
