# 给 AI/新同学的修改约束

## 不允许隐式改变的合同

1. 不得用 test 选 checkpoint、epoch、超参数、方向或 ROI。
2. 不得在没有实测背景帧时声称“背景扣除”。
3. 不得对每张 CCD 单独 min-max；这会抹掉真实能量差。
4. 不得让仿真和实测走不同的 detector normalization。
5. 不得修改 phase 翻转、中心、pitch、distance 后继续使用旧 BMP/manifest。
6. 不得在采集下一层后再更新其上游网络；否则已播放输入失效。
7. 不得把 quick/smoke CCD 混入 formal 数据目录。
8. 不得静默联网下载 Qwen；离线任务模型缺失应立即报错。
9. 不得把旧实验台的 LUT、角点和曝光写成新实验台默认值。
10. 不得只看可视化色图判断一致性；必须保留线性强度指标。

## AI 修改一个新任务时必须回答

- 输入、target、输出 shape 分别是什么？
- 选择哪个参考工程，为什么？
- head、loss、metric 是否与任务类型匹配？
- 哪些参数冻结，哪些训练，各自数量和学习率？
- 数据如何拆 train/development/test，是否按内容或参考图分组防泄漏？
- 光学 active field 如何映射到真实振幅/相位 SLM？
- 哪些鲁棒性来自物理事实，范围依据是什么？
- 实测 CCD 插入哪一层，插入后哪些上游冻结？
- 仿真和实测归一化是否同源实现？
- 最终复现实验需要哪些命令、SHA 和证据文件？

## 任务特定注意事项

- 检索：gallery/query 身份隔离；报告 Top-1、Recall@K、mAP。
- 质量评价：按 reference content 分组拆分；报告 SRCC/PLCC/RMSE，不能随机拆失真图泄漏。
- 稠密任务：保存像素级几何，不能只保留全局 embedding；decoder 必须与 token/grid 对齐。
- 多模态：明确视觉、语言各自在哪些层过光，不得假设 DeepStack 自动参与。
