# ABO 图搜图：实测落差后的鲁棒续训

起点为正式83.125%导出权重，SHA256 `c9926cbaaa1ef066d9657a8028dffc130a33915aa9f392551573dc79894192d0`。
不使用实测微调头作为仿真训练起点，不覆盖正式权重。

原 metadata 已包含DC强度比例20%–30%、CCD截断噪声(mean=.005/std=.01)、增益.95–1.05、相位旁路。
最近 `sku_retrieval_only` 等续训只在10% batch开噪声，且显式关闭router噪声，没有像素配准扰动。
因此不能说原模型“从未做鲁棒训练”，但最后优化阶段的覆盖较弱。

新候选保持六次10cm/532nm传播、17um逻辑尺寸、478有效场、四专家Top2和原电子结构。
alpha下限.35，上限沿用.8；重参数化保留起点四个实际alpha，不直接压低alpha。
冻结Qwen前端，联合训练光学相位、原电子残差/读出和alpha，无教师损失，无新推理分支。

- `physical_robust35`：70% batch加噪声，DC20%–35%，gain .8–1.2，CCD相对均值噪声mean=.015/std=.03、截断[-.05,.12]；router也保留噪声。
- 加每样本入射振幅相对固定相位的±1逻辑像素平移，以及CCD±1逻辑像素平移，边界补零而非环绕。
- `physical_robust35_no_shift`：完全相同日程、噪声、alpha，仅关闭配准扰动，判断配准训练是否牺牲干净指标。
- 相位学习率乘2，router乘.1，EMA.99；每组20epoch×100step，eval每5epoch。只存best/last。

原训练图库1600张用于梯度，query800张仅周期评估/选best；属于已声明的test-selected闭集协议，非无偏独立测试。
保留原始权重作为参考；候选即使鲁棒条件下更好，若干净R1明显下降也不自动替换。
专家资格门槛沿用min share5%、max pair80%、至少3种Top2组合；正常/同权重去光均复评。
上述噪声是训练假设，不是已经测量确定的硬件参数；最终提升仍须新mask实拍验证。

服务器：`/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/physical_robust35_20260926`及`physical_robust35_no_shift_20260926`。
日志为对应目录同级`.log`，状态/执行合同在目录内`status.json`/`execution.json`。
代码在隔离工作树`.worktrees/t07_robust_20260926`，不是修改其他AI运行中的工作树。
