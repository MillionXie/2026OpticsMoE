# 历史 LSP / SALICON / OpenMoji 冻结 backbone 对照

统一源码入口：`python -m LightGenV2.common.frozen_backbone_dense_tasks --help`。
这是 2026-09-23 历史对照的恢复，不是当前光电最终版，也不启动新实验。
执行原 `extract` / `train` 时须显式绑定原配置、数据、冻结模型、缓存和新 run 目录；
不得用整理工作默认重建已清理缓存、重训或覆盖原 run。

## 源码身份及兼容边界

来源提交 `d53d82ab49743fb05022a4b93a31b7d7e37a9066`。
runner 仅将 OpenMoji settings / shared_readout 两个导入重绑至同任务的
`baseline_settings.py` / `baseline_shared_readout.py`，其余计算、训练、EMA 保存、
指标及选模逻辑未改。两个 baseline 专属模块是原提交的完整源码，不替换现用
`settings.py`、`shared_readout.py` 或光电模型。它们复用的 PositionReadout、
SemanticGridDecoder、ConditionedResidual2D 与 main 的定义相同。

服务器 CPU 完整模块导入通过；DeepSeek / CLIP / YOLO+CLIP 的原 final_seed73
PT 严格加载、参数量及有界合成输入输出通过。没有重新读数据集、提取大模型特征、
训练、评精度或测速。这些检查不能代替真实数据固定权重复评。

原报告、PT SHA 和源提交逐项见仓库
`maintenance/storage/DENSE_BASELINE_ASSET_IDENTITY_20261005.json`。
OpenMoji 原 `layered_scene_qwen_shared.yaml` 的七层继承已机械展开为
`tasks/t04_semantic_interaction/configs/baseline_layered_scene_frozen_backbones.yaml`；
相对资产路径仍按同一 configs 目录解析，数值与原继承结果相同。这是原源码配置
入口恢复，原 PT 未内嵌完整配置／命令，不能据此声称每份 PT 的完整启动身份已闭环。
其中 13 份报告含必要最终对照及历史试错，不代表 13 份最终版：OpenMoji 应区分
`final_seed73` 与早期 EMA 保存错误／seed42 对照；LSP 低 PCK 仍属待诊断结果。
SALICON 固定低学习率对照和原分段训练不能混成同一预算。

## 科学口径

- Backbone 完全冻结，只训练各任务自己的 adapter/readout。
- LSP 按最小 TRAIN loss 选择；SALICON 按公开 validation CC 选择。
- OpenMoji 原协议按 TEST changed-cell 选择，属于开发指标，不是独立测试。
- YOLO 的 OpenMoji 文本来自冻结 CLIP，必须标作组合 baseline。
- 历史数据、权重、逐样本结果、测速和必要对照原位保留；身份核验不等于全部
  缓存、模型 snapshot、配置继承及数据资产已完成复现闭环。

当前 lightgen 主方法、ABO 封存版和 OpenMoji 现用实验均未因本次恢复而改变。
