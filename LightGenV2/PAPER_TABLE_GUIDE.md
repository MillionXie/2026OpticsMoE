# 论文总表填写口径

当前每个已建立任务比较三组系统，但三组并不都具有 Router：

| 方法 | 当前服务器测什么 | RTX 5090 D 测什么 | Router |
| --- | --- | --- | --- |
| Ours | 仿真性能 | 不测大模型指标 | 光学 Router，4 专家 Top-2 |
| Matched D2NN | 仿真性能 | 不填速度、功耗 | 无 Router |
| Frozen Qwen | 暂不运行，字段保持 `null` | 性能、速度、功耗 | 无 Router |

因此，论文表中的一个笼统 `Baseline` 性能格不能无标签地混放两个结果。推荐将性能拆为
`Ours Sim. / Ours Exp. / D2NN Sim. / Frozen-Qwen` 四列；如果暂时不改表头，则在
`Baseline` 单元格中明确分两行写 `D2NN: ...` 和 `Frozen Qwen: 待5090D`。速度与功耗的
`Baseline` 只指 Frozen Qwen，D2NN 按当前要求不测这两项。

## Frozen Qwen 速度

- 设备固定为 RTX 5090 D，`batch=1`。
- 起点是已经形成 hidden states、即将进入第一个原生 Transformer block。
- 终点是任务最终输出：检索排名、关键点、显著图、语义编辑网格或质量分数。
- 不计文件读取、图像/视频解码、tokenizer、patch/token embedding 和第一个 block 前处理。
- 模型只加载一次，显式 warm-up 50 次；一般任务测 200 个 test 样本，OpenMoji 的
  自回归生成则测完整 1,000 条 test。
- 同时保存 CUDA/host 的 mean、median、P5、P95 和逐样本记录，不能只报告最快值。

## Frozen Qwen 功耗

与速度在同一批推理中测 RTX 5090 D 的板卡功率：至少 20 Hz 采样，并报告 idle、active
mean、peak 和扣除 idle 后的 `J/sample`。不得用 575 W TDP 代替实测平均功率，也不得
从其他型号 GPU 外推。

## 数据和选模

除当前明确排除的 ABO 外，已建立任务统一采用：无 validation、每 5 epoch 测一次 test、
按任务主指标选择 `best_checkpoint.pt`，并同时保留 `last_checkpoint.pt`。这是一种
`test used for selection` 协议，不应表述为无偏泛化评估。各任务确切样本数和切分规则见
[`DATA_SPLIT_AND_ROUTER_PROTOCOL.md`](DATA_SPLIT_AND_ROUTER_PROTOCOL.md)。

5090 D 的可执行入口、记录字段和待测 JSON 合同见
[`RTX5090D_QWEN_BASELINE_PROTOCOL.md`](RTX5090D_QWEN_BASELINE_PROTOCOL.md)。
