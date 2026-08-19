# P04 部署偏移后适配实验

更新日期：2026-08-20

## 1. 本轮修正的问题

P03 测量的是冻结 checkpoint 在部署误差下的零样本推理性能，没有执行部署后的反向传播。旧实现中的 `phase_override` 来自 `detach()` 后的静态相位，并且会让 FA 模型绕过固定反馈路径，因此不能直接用于继续训练。

P04 新增可微部署路径：前向传播始终使用发生固定横向偏移后的当前相位掩模；BP 使用该部署前向的当前精确 Jacobian；FA-pretrained 的局部相位梯度仍由当前部署算子精确计算，但传向前一光学层的误差信号固定使用部署前最后一次训练得到的理想算子；FA-random 使用固定随机误差连接。

## 2. 公平的四组协议

四组全部从同一个 P02 BP 高性能 endpoint 开始，不能分别从四个不同 P02 endpoint 开始，否则部署前模型性能和电子旁路依赖会混淆部署适配能力。

| 组别 | 部署前向 | 部署反向/更新 |
|---|---|---|
| NoFT | 固定偏移后的当前物理前向 | 不更新，提供即时部署下限 |
| BP-current | 同上 | 当前偏移算子的精确 BP，作为仿真上限 |
| FA-pretrained | 同上 | 局部梯度精确，跨层连接固定为部署前最后算子 |
| FA-random | 同上 | 局部梯度精确，跨层连接固定随机 |

电子读出头和小型电子残差在三种适配组中均使用普通 BP；架构、参数量和光学权重下限保持 P02 A13 不变。

## 3. 首轮验证集筛选

- 共同 source：P02 `bp/seed_2026/best.pt`。
- deployment seed：9101。
- 只使用 validation 汇报首轮结果，不查看 test。
- 适配 10 epochs，所有组使用相同数据顺序、增强、优化器预算和 validation-best 选择规则。
- 全局刚性偏移：所有八层共享同一个 `(dy, dx)`。
- 逐层独立偏移：八层分别采样方向；这是 P03 使用的更严苛装调误差。
- 首轮强度：0.125 和 0.25 pixel。先确定 BP-current 能否恢复，以及 FA-pretrained 是否显著优于随机，再扩展到 0.5/1/2 pixel 和鲁棒训练。

## 4. 必须报告的量

1. 偏移后、适配前的即时准确率；
2. 每 epoch 的恢复曲线和 validation-best；
3. 恢复的绝对百分点，以及找回的偏移损失比例；
4. epoch 0 分层梯度 cosine：FA-pretrained/FA-random 分别相对 BP-current；
5. 全局偏移与逐层独立偏移分开报告；
6. 每层实际 `(dy, dx)`、source checkpoint SHA-256 和完整命令入口。

## 5. 代码与执行动作记录

- `optics.py`：增加可微相位平移；处理未覆盖边界处 `atan2(0,0)`，防止 NaN 梯度；增加“当前部署前向/局部梯度 + 固定跨层连接”的 FA 路径。
- `model.py`：部署状态可以传入每层位移和固定相位误差，而不再依赖 detached phase override。
- `deployment_robustness.py`：保留 P03 冻结推理，统一位移采样，并增加 global/layerwise 几何以及可微部署状态构造器。
- `deployment_adaptation.py`：实现共同 source、四组适配、梯度诊断、恢复曲线、checkpoint 和汇总。
- 配置：`configs/p04_deployment_adaptation_screen.yaml`。
- 单卡入口：`commands/37_run_p04_adaptation_screen.sh`；汇总入口：command 38；GPU 4/5 双卡启动入口：command 39。

## 6. 后续门槛

若 BP-current 在 0.125/0.25 pixel 下不能明显恢复，优先排查可微偏移、学习率和可逆性，不解释 FA 排名。只有 BP-current 恢复成立后，才判断旧算子 FA 是否有效。若两者都能恢复，再固定协议扩展到多 seed、test、数 pixel 偏移，并增加 misalignment-vaccinated/SAT 鲁棒训练；后者是训练策略，不增加第五个反馈方法组。

## 7. 2026-08-20 启动与首批检查

- Git 实现提交：`efa2e93d`；本地只提交本实验目录，没有纳入工作区中原有的 Qwen/.gitignore 改动。
- 服务器同步后单元测试：`21 passed`。
- 真实 checkpoint 两 batch smoke：global 0.125 px 下，初始 validation 57.94%；BP-current 到 61.16%，FA-pretrained 到 61.10%。FA-pretrained 相对 BP-current 的八层梯度 cosine 为 0.974--1.000，且两种方法部署前向一致。
- GPU 4 运行 global 0.125/0.25 px；GPU 5 运行 layerwise 0.125/0.25 px。两卡均通过 command 39 启动。
- 第一个完整 BP-current epoch：global 0.125 px 从 57.94% 恢复到 71.20%；layerwise 0.125 px 从 62.20% 恢复到 71.50%；共同 source ideal 为 73.40%。这证明 P03 的低值来自冻结推理，而不是 BP 无法获得部署后的当前关系。
- 上述为运行中检查点，不替代 10 epoch validation-best 汇总；必须等待四条件四方法全部完成后再判断 FA-pretrained 与 FA-random。

## 8. 0.125 pixel 四组完整结果

共同 P02 BP source 的理想 validation 为 73.40%。下表为同一 source、同一偏移和同一适配预算下的 validation-best：

| 位移几何 | NoFT | BP-current | FA-pretrained | FA-random |
|---|---:|---:|---:|---:|
| global 0.125 px | 57.94% | 73.54% | 73.36% | 69.24% |
| layerwise 0.125 px | 62.20% | 73.48% | 73.42% | 70.54% |

- global：BP 与 FA-pretrained 分别恢复 15.60/15.42 pp，差 0.18 pp；FA-pretrained 比 FA-random 高 4.12 pp。
- layerwise：BP 与 FA-pretrained 分别恢复 11.28/11.22 pp，差 0.06 pp；FA-pretrained 比 FA-random 高 2.88 pp。
- BP 与 FA-pretrained 的最佳点均为 epoch 8；FA-random 的 global/layerwise 最佳点分别为 epoch 9/2。
- 这批结果首次严格证明：部署偏移后 BP-current 能获得最新关系并恢复性能；部署前最后一次光学算子在 0.125 px 下几乎复现 BP，而同形状随机连接明显较差。
- 0.25 px 仍在运行。其即时 NoFT 为 global 24.70%、layerwise 29.66%，用于检验从 P03 失效边界能否通过部署后训练恢复。
