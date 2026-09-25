# 冻结共享视觉前端：新协议与阶段性证据（2026-09-25）

本文件记录验证集候选和已经按验证选模后仅测试一次的 EuroSAT A；未完成的 B/C/D 不填入正式下三角矩阵。原无视觉前端协议的数字不可拼入本协议。

## 固定电子输入与公平边界

- 唯一视觉前端为 `pretrain_clevr_vision.py` 的 32,128 参数 CNN。它在完整 CLEVR 原图与问句标签上预训练，临时 3,096 参数／24 输出头在光学训练和推理时丢弃。checkpoint SHA256：`d20273e1828fec4e5a7a5545fd892a722f3909f7205a791e0e682b437526aa11`。完整 45,000 条验证问句准确率 86.64% 只说明此前端学到视觉属性，**不是光学成绩**。
- EuroSAT 与 CLEVR 都读取同一冻结 CNN 权重。每个 R/G/B 112×112 方块上方 84 行为该通道的原图缩放，下方 28 行为同一 128 维特征的固定图案。三方块的原图部分合计功率 0.25、特征部分合计 0.25。第四格仍是同地点 SAR 或该样本原始文字问句，功率 0.5。Speech、Physical 无 CNN 改动；Physical 用原始八帧，不算帧差。
- 输入审计 `audit_frozen_vision_input.py` 已在服务器训练集通过：EuroSAT 15,998 个配对地点，CLEVR 140,000 条原图／问句记录；原 SAR 和原问句格逐元素未变；同图正负问句的图像格相同、问句格不同；两任务抽查入射功率都是 1.0。
- MoE、D2NN 光路几何、OEO、CCD 及唯一 `Linear(784,10)` 读出合同未改。前端和编码对两个模型逐元素相同，但视觉 CNN 用 CLEVR 标签预训练，结论只能称作**共享冻结电子视觉前端下的光电架构比较**。

## 验证与已开放的一次测试

| run ID（位于 `runs/simulation/`） | 范围 | 选中轮次 | 完整验证宏平均召回 | 测试宏平均召回 |
|---|---|---:|---:|---:|
| `euro_moe_sharedvision_s17_b015` | EuroSAT A，MoE 4 槽 | 15 | 74.56% | 77.85% |
| `euro_d2nn_sharedvision_s17_b015` | EuroSAT A，D2NN | 4 | 67.96% | 71.77% |
| `clevr_moe8_sharedvision_s17_b015` | 独立 8 槽 CLEVR 可学习性候选 | 5 | 80.15% | 未测试；不能填入终身学习 B |

EuroSAT A 两种模型都使用相同输入和无额外训练技巧的 10 输出头；MoE 比 D2NN 的独立测试宏平均召回高 6.09 个百分点。测试文件分别为 A run 内 `selected_test.json`；按完整验证集选定 checkpoint 后各开放一次。MoE A 的四个活动槽 4/7/10/13 在验证集上平均分得 12.87%/11.85%/16.97%/58.31% 路由功率，所有验证样本的最大权重都是槽 13；**精度优势不证明路由已形成样本级专家分工**。从 A checkpoint 低学习率继续训练的 balance=1/10 候选虽使均值更均衡，仍未满足样本级多槽胜出并保持精度的双重条件，因此正式 B 从原 A 继续。

## 顺序学习当前难点

阶段 B 的 `stage2_moe_replay_sharedvision_s17_b015` 从上述 MoE A 出发，旧四专家相位冻结、开放四个新专家；每轮分别遍历 EuroSAT 全部 15,998 条和 CLEVR 全部 140,000 条训练记录一次。`old_task_loss_weight=1` 不等于两任务累计梯度同权，因为 CLEVR 批次约为 EuroSAT 的 8.75 倍。运行期间曾出现 CLEVR 完整验证升至约 80%、EuroSAT 降至约 61% 的明显遗忘；最终选模尚待完成。下一验证候选提高旧任务损失权重，仍以两任务完整验证宏平均召回均值选模，不碰测试集。D2NN full replay B 是额外对照；正式无 replay 下三角使用独立的 `--replay-mode none` 入口。

所有 run 的 `config.json` 保存源码 commit、源协议和视觉 checkpoint 哈希，`history.json` 保存各轮验证，`result.json`／`selected_test.json` 保存选中结果。当前 B/C/D 仍在推进，任何尚未产生的矩阵单元均为空。

## 已完成的固定 D2NN 纯推理 4×4

`d2nn_fixed4x4_sharedvision_s17_d9d5`：行是独立训练的 D2NN 相位和**该行自己的固定单层 Linear**；列是目标测试集，完全不重训相位或头。EuroSAT/CLEVR 目标均用上述同一冻结视觉前端；Speech 的独立源 checkpoint 因音文输入没有视觉 CNN、输入和光路均未改变，沿用此前的独立训练结果；Physical 源是新无帧差 checkpoint。单位为测试集宏平均召回百分比。

| D2NN 源权重／目标任务 | EuroSAT | CLEVR | Speech | Physical 无帧差 |
|---|---:|---:|---:|---:|
| EuroSAT | 71.77 | 0.00 | 0.00 | 0.00 |
| CLEVR | 10.00 | 76.50 | 50.00 | 50.00 |
| Speech | 10.13 | 50.25 | 67.19 | 50.00 |
| Physical 无帧差 | 10.00 | 50.00 | 51.90 | 75.35 |

16 格均已计算；EuroSAT、CLEVR、Physical 三个新独立 run 的对角格与其 `selected_test.json` 完全一致。0% 与 10% 格受十输出共用标签位置和固定头预测到目标合法类别之外的影响，不能孤立解释为相位完全没有迁移。Physical D2NN 对角格虽为 75.35%，两类召回约为 53.30%／97.40%，存在预测偏向。

## D2NN 顺序对照当前节点

- full replay 额外对照 `stage2_d2nn_replay_sharedvision_s17_b015` 已按验证选中第 4 轮并测试一次：学完 B 后 EuroSAT/CLEVR 为 **63.87%/76.30%**。
- 正式无 replay `stage2_d2nn_noreplay_sharedvision_s17_7a50` 已按验证选中第 4 轮并测试一次：学完 B 后 EuroSAT/CLEVR 为 **10.00%/76.41%**。同一个起点的 A 对角格为 EuroSAT **71.77%**。这条链正在推进 C、D；B 的严重遗忘不能由独立训练模型替代或推断。
