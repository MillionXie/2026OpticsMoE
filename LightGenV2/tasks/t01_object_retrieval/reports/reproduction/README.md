# t01_object_retrieval 复现说明入口

[2026-10-03版本与资产核验](IDENTITIES_20261003.md)：正式DC20、无DC20必要对照、PT/SHA、数据划分及未完成边界。

[Baseline复现说明](BASELINE_METHODS.md)：历史baseline方法说明；本次没有重新训练。

## 冻结前端的当前绑定与历史限制（2026-10-05）

五个已保存 run 的配置、run manifest 和现有 best PT 元数据已只读核对：它们使用
`Qwen/Qwen3-VL-Embedding-2B`，但没有记录精确 frontend revision。当前缓存的
`9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` 及逐文件 SHA 已保存在
[原资产清单](../../../../../maintenance/storage/T01_CONTENT_ASSETS_20261004.json)，
不能反向证明当时运行一定用了它。详见
[五个 run 的绑定审计](../../../../../maintenance/storage/T01_FRONTEND_BINDING_20261005.json)。

主入口新增可选 `--frontend-snapshot` 和 `--frontend-identity`，只用于显式绑定当前已保存
模型：逐文件校验大小/SHA，强制离线加载，要求全新 `--run-dir` 并使用独立 teacher cache。
原 profile 默认行为不变，原 run、权重、指标和测速不改。此命令是后续复现入口，未实际
执行新的评估，不能据此声称历史精度已复现：

```bash
python -m LightGenV2.tasks.t01_object_retrieval.run --profile main_dc20 --phase evaluate \
  --frontend-snapshot /DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --frontend-identity maintenance/storage/T01_CONTENT_ASSETS_20261004.json \
  --checkpoint /DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t01_object_retrieval/runs/simulation/moe_router_scale_dc20_strict_seed42/best_checkpoint.pt \
  --run-dir LightGenV2/tasks/t01_object_retrieval/runs/simulation/explicit_frontend_recheck
```

执行前仍须核验该 PT、原 split manifest、初始化来源及环境。新绑定的结果另存，不覆盖
历史报告；未知的历史 revision 保持明确例外，不倒填。

本目录集中保存baseline及主方法的可复现性证据；本次只建立入口，**尚未进行本任务的新一轮复现**。
当前任务结构和已有结果见 [任务README](../../README.md)。不得因为存在本文件就声称已复现。

后续每个正式结果需在这里记录：

1. baseline定义，冻结/训练的参数，预处理、输出头与主方法差异。
2. 原始数据版本、train/test清单与SHA256、标签生成方法、指标与选模口径。
3. 源码commit、完整命令、依赖环境、模型及checkpoint的SHA256和获取位置。
4. 固定权重复评与从头重训分别报告；标明样本数、seed、运行ID、结果和误差。
5. 速度/能耗的硬件、计时边界、功率积分口径；未测的不得填估计值冒充实测。

原始日志及逐样本结果留在本任务runs，文档只引用。参考 [SALICON复现说明](../../../t03_saliency/reports/reproduction/README.md)。
