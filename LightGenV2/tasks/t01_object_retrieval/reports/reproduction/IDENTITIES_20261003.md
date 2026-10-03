# T01 已有版本及资产核验

2026-10-03。本次只读核验服务器，不重新训练/评估，不删代码、数据或测速。
逐文件证据：[两任务资产审计](../../../../../maintenance/storage/T01_T02_ASSET_AUDIT_20261003.json)。

## 保留的正式版本与必要对照

服务器run根为 `/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t01_object_retrieval/runs/simulation/`。

| run | 意义 | 原报告仿真Top-1 | best epoch |
| --- | --- | ---: | ---: |
| moe_router_scale_dc20_strict_seed42 | 当前DC20主方法 | .9000 | 25 |
| d2nn_matched_dc20_seed42 | 同合同普通D2NN | .8950 | 15 |
| qwen_frozen | 冻结Qwen，不训练光学学生 | .9950 | 不适用 |
| moe_router_scale_seed42 | 无DC20旧主方法，必要历史对照 | .9100 | 20 |
| d2nn_matched_seed42 | 无DC20旧D2NN，必要历史对照 | .9050 | 25 |

这些均为已有仿真报告，不是新实拍。两种训练模型按周期TEST选权重，属于开发/选模偏置指标；
冻结Qwen是固定单次评估。新旧DC20条件不同，不按较高分替换正式版本。
四个训练run的best/last均重新计算SHA，完整值见审计JSON；冻结Qwen没有学生best/last。

## 数据身份

五个run的 `manifests/caltech101_10class_subset.csv` 文件SHA相同：
`87be127ae3099f0a0c5f8f276ed45e990fa012111ac5978b8f58ed9a1eca2ef3`。
各有2625 TRAIN、30 gallery、200 TEST，共2855条；样本ID无重复，全部图片路径存在。
本次不逐原图重算内容SHA，不将“路径存在”冒称原图字节完整审计。

报告中的逻辑记录SHA `c8accd596cd418bbef65834ac75b11cd35ce7cdd4ebe418aa51d942e00701d11`
不是CSV原文件SHA。已按原 `sha256_records` 协议重新计算：恢复int/bool类型，
按 `(split, sku_index, sample_id)` 排序，再对规范JSON逐记录哈希，结果与报告完全一致。
两种身份均保留，不能混用或因二者不同误判数据损坏。

## 源码和复现边界

观察服务器主目录HEAD `3f85510285e5ffdfca28def93eef2eb082b1655c`，本任务源码无已跟踪修改；
run/report/baseline入口的52项静态导入闭包与本轮main逐Git文件相同。
T01 DC20训练 `environment.json` 实际记录训练commit
`f9bcf8261b9e89cce10469b6ab0a972da752cc9c`，不能用今天的盘点HEAD替代训练身份。
本机 `common/__init__.py` 有独有修改，本次没有覆盖或提交该文件。

实际服务器既有T02个人工程中执行T01/T02 CPU合同测试，共39项通过；不是新完整模型复评。
本机Torch DLL加载失败，未改本机环境，不能声称本机科学测试也通过。
命令、配置、原始结果仍在上述run；旧后端 `experiments/` 尚有运行依赖，不删除。

测速、功耗、baseline身份与原遥测全部原位保护；当前精确PT的实验台端到端延迟未核定。
剩余工作是动态/环境依赖及全部测速绑定核验，不因本页就宣称三端完整复现完成。
