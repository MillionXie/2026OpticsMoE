# 2026-09-23 ABO 图搜图全量实测

## 结果

固定协议为 800 个 query 对 1600 张模拟 gallery，同 SKU 为正例。

| 数据 | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| 同批输入纯仿真 | 0.825000 | 0.918750 | 0.952500 | 0.8671712970484474 |
| 六层实测 | 0.197500 | 0.318750 | 0.402500 | 0.26716238505468964 |

纯仿真与实测描述符的平均余弦相似度为 `0.44705474376678467`。独立重算与运行报告完全一致，因此低实测指标不是 gallery 映射或指标脚本错误。

## 硬件配置

- 相机实际曝光：`20002 μs`
- 相机增益：`1.0`
- 振幅 SLM 换图等待：`240 ms`
- 全部 800 query 均完成六层光学采集
- 最大饱和像素比例：`0`
- 不做逐图光度归一化
- 模型 SHA256：`c9926cbaaa1ef066d9657a8028dffc130a33915aa9f392551573dc79894192d0`

六阶段平均 PCC：

| 阶段 | PCC |
|---|---:|
| vision router | 0.339786 |
| vision expert | 0.075752 |
| vision global | 0.071747 |
| language router | 0.306688 |
| language expert | 0.051699 |
| language global | 0.058089 |

结论：相机动态范围与设备控制正常，router 仍有约 `0.31–0.34` PCC，但 expert/global 的仿真实测一致性很低，是本轮准确率下降的主要位置。该结果是真实全量实测结果，不能用四样本 smoke 或筛选样本替换。

## 文件

- `report.json`：权威全量报告及800条逐样本预测。
- `features.pt`：800条实测描述符与同批纯仿真描述符。
- `independent_verification.json`：独立重算的仿真/实测指标。
- `run_contract.json`：曝光、等待、模型及六阶段方向合同。

远端原始 CCD 和断点数据：

`E:\code\guest\2026OpticsMoE\ABO_I2I_Lab_DVP_8um\runs\full800_20260923`

远端完整日志：

`E:\code\guest\2026OpticsMoE\ABO_I2I_Lab_DVP_8um\runs\full800_20260923.log`
