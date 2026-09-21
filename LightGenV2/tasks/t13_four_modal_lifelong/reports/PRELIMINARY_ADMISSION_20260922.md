# 四模态单任务准入结果（2026-09-22）

## 结论

四项任务均已在真实角谱 FFT 传播、逐层 OEO 和最终电子 MLP 读出下通过单任务准入。
D2NN 门槛为 65%，四专家 MoE 门槛为 70%。下表使用验证集选择轮次，测试集只在选择完成后评估；
数值为 balanced accuracy，数据平衡时也等于 accuracy。

|任务|D2NN 验证 / 测试|MoE 验证 / 测试|结论|
|---|---:|---:|---|
|EuroSAT RGB / SAR|78.50% / 77.10%|78.05% / 76.80%|均通过|
|CLEVR 图 / 文|74.00% / 70.67%|73.60% / 72.93%|均通过|
|Speech Commands 音 / 文|94.31% / 94.35%|93.83% / 93.83%|均通过|
|Physical Concepts 视频 / 文|98.30% / 99.33%|98.39% / 99.41%|均通过|

这批结果的作用是确认四种输入和统一光学接口可学，不作为全量终身学习的最终表格。
EuroSAT、CLEVR 和 Physical Concepts 在本表仍使用已审计的 preliminary 包；Speech 使用完整去重后的
`mini_speech_commands` 发布包中的全部可用语音，但本表对应的 run 仍保留 preliminary 名称。

## 数据合同

|任务|训练 / 验证 / 测试|类别与正负构造|
|---|---:|---|
|EuroSAT|6000 / 2000 / 2000 张|10 类；每个划分同时含 RGB 与配对 SAR|
|CLEVR|6000 / 750 / 750 个问答|二分类；颜色形状查询存在 / 不存在，正负平衡|
|Speech Commands|12526 / 1686 / 1734 个问答|二分类；语音关键词与文本相符 / 不符，正负平衡|
|Physical Concepts|5784 / 1056 / 1352 个问答|二分类；视频与 possible / impossible 文本相符 / 不符，正负平衡|

## 模型和训练

- 单任务准入使用固定的 518×518 光学几何。MoE 有四个 224×224 专家，router 将探测能量归一化为
  soft-routing 功率，专家入口振幅乘 `sqrt(q)`；D2NN 使用相同孔径和传播次数。
- 输入经过真实 angular-spectrum FFT propagation、相位层和非负 `intensity -> softsign` OEO。
- 最终输出是完整 CCD 强度经 28×28 pooling 后进入任务专属 MLP。固定 CCD 能量区只在训练时提供
  相位辅助损失，不参与最终预测。
- 光学训练后冻结传播层，在训练集 CCD 特征上重新拟合 MLP，用验证集选择 MLP 轮次。表中所有数字
  都来自这个 MLP。
- 单任务共训练 40 轮；表中的选中光学轮次依次为 D2NN 34/30/36/2，MoE 14/16/25/6。

## 可审计 run

服务器根目录：
`/DATA/DATA1/guest3/t12_gate_0e577a828/LightGenV2/tasks/t13_four_modal_lifelong/runs`

- `admission_eurosat_aux_s17_v1_{d2nn,moe}`
- `admission_clevr_aux_calibrated_s17_v2_{d2nn,moe}`
- `admission_speech_aux_s17_v1_{d2nn,moe}`
- `admission_physical_temporal_s17_v1_{d2nn,moe}`

每个目录保存实际配置、Git commit、数据 manifest SHA256、逐轮 history、best/last checkpoint、
校准后的 checkpoint、混淆矩阵和最终 `comparison.json`。

## 全量准备状态

- EuroSAT：完整归档每模态 27,000 个文件；审计 split 中 26,892 个可用配对已全部解码，无抽样。
- Speech Commands：完整 `mini_speech_commands` 发布包共保留 7,973 条去重语音，27 条完全重复波形被记录并移除。
- Physical Concepts：全部 5,000 个 continuity quadruplet 已编码，划分为 3543 / 700 / 757 个互斥组。
- CLEVR：完整 19,021,600,724 字节官方 ZIP 正在并行断点下载；完成后先做 ZIP 全成员校验，再处理
  70,000 个 train 图像和 15,000 个 validation 图像，每张生成 3 个正例与 3 个负例。

完整包使用独立路径和 `all_original_samples=true`；配置中的 `require_full=true` 会拒绝任何抽样包。
正式终身矩阵必须等四个完整 feature package 都通过这个检查后再运行。

## 终身比较

任务顺序固定为 EuroSAT → CLEVR → Speech Commands → Physical Concepts。正在按相同顺序和训练预算运行：

1. sequential D2NN，无 replay；
2. sequential D2NN，每个旧任务相同 replay 预算；
3. ours，固定 16 个专家槽位，按 4→8→12→16 激活，冻结旧专家并使用相同 replay。

每学完一个任务就评估所有已学习任务，输出 validation/test 下三角矩阵、backward transfer 和 forgetting。
没有联合训练 D2NN。
