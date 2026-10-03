# T03 显著性任务：已发布源码，资产继续原位保护

2026-10-03。正式源码发布 main：`7dfc04b4d9892a64334560d6bb983142584d6031`。
本地 main、GitHub main、服务器 origin/main 已一致；服务器工作 HEAD 与修改未改变。
工作目录仍为用户原 T12 分支，不能把发布说成所有运行目录都已经切换 main。

## 本次完成

- 来源为实际服务器 `/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t03_licensed_figures_20260929`，
  固定提交 `b75a56203b47818810e0a45dceecf9cea4c6aa9f`；任务源码无修改、无相关训练占用。
- 191份 Python/YAML运行文件在候选树逐字节一致、Python编译通过。
  main 原有5份核心差异按旧/新SHA明确审查替换，177份缺失运行文件追加。
  保留当前main的T02/T06共享依赖，不用T03旧分支覆盖其他任务。
- 固定来源CPU合同296项通过；与当前main五份不同共享依赖组合后仍296项通过。
  初次stdin测试启动器的两项多进程测试因不存在的 `<stdin>` 路径失败，修正启动器后全通过；
  没有修改科学源码来绕过测试。CPU测试不等于重新训练或实拍性能复测。
- 保留原报告、baseline、硬件说明与配置历史；补齐正式任务入口与图例打包脚本。
  图例显示脚本不修改预测或评分，仅用于展示，不是模型新版本。
- 既有T13/T16保护清单89项源码和22项教师依赖SHA复核无错；治理工具29项通过。
- Git临时索引生成候选，无新分支/工作树，无整分支merge，无用户脏文件入暂存。

## 已重新核验的资产身份

服务器最终best：`036bc8caedcde4d6dabce276b1a2e4af15d960d88620098b19e1c198849b8bfe`；
last：`4cfa2ac828452e7bdb3e81f8eeea0091faf8e2c4e4509684ebdd4d19aea2eaed`。
原run为 `moe_alpha40_sam_batch8_crosssample_20260913_seed42`。
原数据清单SHA：`8b564715318cdf956aa83861b69195b44e1782d1e695645a44bf05e51d2ff738`。

本地原实拍结果 `LightGenV2/tasks/t03_saliency/runs/hardware/salicon_08625_shs_20260914/full01_retry01/results.json`
SHA：`8104cd297e4b6c468cc35abcddfb0822170193107082b87155f1b88f34ca7996`。
5000图、三轮CCD逐图记录；原仿真CC=.8624925081777596，原实拍CC=.8597739692151547，未微调。
这些是旧正式结果的身份核验，不是本轮新采集。

Qwen aligned staged baseline best SHA：`531c4a330fc05e2c27b037784588716d248a29d1ab8da951d443165dcce552f8`；
旧DC20 MoE best SHA：`fae78751fb4876c176d57ffc081aae806d30c91d7e0ad741902360483a7430b4`。
同权重去光、D2NN、旧Qwen与全部测速仍保留；未逐一完成全资产复核，不能删旧目录。

## 尚未完成

全量资产跨机器闭包、历史baseline实际路径及实拍设备源码/依赖完整核验仍需收尾。
本任务由“待审计”改为“源码已收敛、资产部分核验”，不计为整个工程全部完成。
本轮数据/PT/CCD/收据/测速/功耗文件零删除，外部上传未恢复。
