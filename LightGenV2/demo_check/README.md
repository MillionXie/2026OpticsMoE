# 历史 demo / 必要对照入口

这里是历史实验与必要 baseline，不是另一个日常主工程。正式任务仍从仓库根 `START_HERE.md` 和 `LightGenV2/tasks/` 进入。原数据、权重、逐样本预测及全部测速保留。

## EuroSAT 历史协议请分开看

| 协议 | 源码入口 | 原结果范围 |
| --- | --- | --- |
| 无电子分类旁路、相位训练 | `pure_optical/run.py`、`pure_optical/evaluate_holdout.py` | 动态MoE/整孔径D2NN，旧2000验证39.85%/29.20%；不是当前任务成绩 |
| 同一冻结CNN、输出概率0.5融合 | `frozen_electronic/run.py`、`frozen_electronic/config.json` | 电子76.05%、MoE融合75.85%、D2NN融合75.90%，旧2000验证；融合没有带来净收益 |
| 同一冻结CNN先编码光场，无输出电子分类旁路 | [shared_frontend](shared_frontend/README.md)；单run独立测试 `shared_frontend/evaluate_single_holdout.py` | 原20轮验证77.20%/73.65%，后续独立空间测试与续训另记，不混成同一个数值 |

原实验说明、限制与完整报告见各协议 README 和 `reports/reproduction/`。本轮没有重新训练、测试或采集，也没有把历史验证精度改称独立测试精度。

## 本轮收录范围

三份此前遗漏的入口已与 `/DATA/DATA1/guest3/demo_reproduction_20260915` 的实际文件逐字节核对。源码身份见 [清单](HISTORICAL_RUNTIME_IDENTITY_20261006.json)。融合配置来自既有提交 `81fb4e79332b4d07601e7218e7e69fa59dc480c2`。CPU测试只校验来源、固定配置、小型指标计算，不执行历史训练或访问正式数据。

历史训练入口仍按原脚本方式运行，部分使用同目录导入与动态导入，不保证任意 `python -m` 调用方式。环境、全部模型/数据闭包及历史打包入口还没有完成整体验收，不能把本页当作新机器部署完成证明。

独立发布包内五份改写源码保留为交付版本，不能用包内源码SHA替换训练源SHA。其余历史分析/绘图/数据准备工具仍待用途核对，本轮未删除任何文件。
