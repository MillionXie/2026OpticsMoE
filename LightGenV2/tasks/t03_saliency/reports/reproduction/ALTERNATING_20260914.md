# SALICON：先训练光，再训练电（2026-09-14）

## 目的与边界

前一组`moe_alpha40_optical_lr_priority_20260913_seed42`只压低电子LR，20轮没有超过起点，
没有验证“新光场+充分适配的电子后端”。本次保持所有推理结构、85412参数头及数据不变，
从独立核验的036bc8ca来源重新进行完整交替，不使用前组的best冒充已更新光学权重。
推理无新增模块/attention，Qwen前端一直冻结。目标CC≥.87，不保证达到。

## 三阶段合同

|epoch|更新范围|冻结范围|初始学习率|
|---|---|---|---|
|1–10|四专家、全局相位、光router|所有电子适配/残差/融合alpha/CCD读出/最终头|相位.004、router.001|
|11–25|原有全部电子组，包括最终读出头|全部光相位和router|电子1e-5、CCD/头2e-5、空间FFN5e-5|
|26–30|原有光电组联合|Qwen前端|上述各组基础LR×.1|

每阶段内部余弦衰减至起始LR的.2倍。LR是AdamW原始参数坐标，不是弧度值。
物理相位仍为2π sigmoid(raw)，没有改变相位参数化。
第一阶段不对冻结电子做SAM虚拟扰动，而以同一个GT+空间CC蒸馏loss做一次AdamW更新；
第二、三阶段恢复电子SAM rho=.05。绝不回退到不同loss的旧训练入口。
每阶段交接继续使用LIVE权重，即使此时CC低于初始化也不恢复best。
重置优化器momentum，并把EMA同步到live，防止冻结光相位仍在EMA中缓慢漂移。
固定光学参数≠光学前向no_grad：电子输入仍通过固定衍射过程获得梯度。
逐轮断言冻结参数逐位不变，逐组记录raw更新RMS；若非法变化或非有限数直接报错。

## 其余不变

光router Top2、四专家、同尺度融合、alpha∈[.4,1]，478有效ROI、224专家、17μm、传播10cm。
训练零级20–30%、原有噪声与跨样本专家均衡不变。本轮不叠加MixUp/额外辅助头/图像增强。
GT始终保留；训练图教师空间CC蒸馏系数维持2，沿用已固定的教师缓存。
SALICON train10000/public-test5000，不设validation；初始、第1、每5轮、末轮测试并选最高CC。
因此存在public-test选模偏差，不能称独立泛化评估。标准eval关闭随机光噪声，不是硬件成绩。
只保存best/last PT；逐轮相位诊断是JSON，不增加周期PT。

## 操作与产物

仓库根运行，先检查空闲GPU；同一助手默认1张、最多2张，不停止他人作业。

```bash
python -m pytest LightGenV2/tasks/t03_saliency/tests -q
python -m LightGenV2.tasks.t03_saliency.alternating_preflight
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --phase all --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_alternating_20260914.yaml
```

正式目录：`runs/simulation/moe_alpha40_alternating_20260914_seed42`。
真实8张train图预检：`runs/smoke/alternating_20260914/report.json`，只验证更新、冻结和执行路径，
其训练CC不是新测试性能。重复预检拒绝覆盖既有report。
正式`metrics/training_history.csv`记录stage、每组LR、冻结核验、每轮raw更新RMS、alpha与CC；
`metrics/alternating_phase_progress.json`记录每张相位相对本run初始的圆周RMS和相量距离。
第10/25/30轮自动将当时live相位图保存到`stage_diagnostics/epoch_NNN_stage/`，
附当时last的SHA和live标识；不保留额外PT、不把这张图冒充test best。
`best_visualization/`始终展示所选best，不能把best退回初始化时的图当成末轮mask。
`last_checkpoint.pt`保留live末轮和EMA状态，解释图片前须明确选的是哪个权重。
模型执行/训练命令/环境/源码由run_manifest与launch_record记录，resolved_config固定实际配置。

起点：`moe_alpha40_sam_batch8_crosssample_20260913_seed42/best_checkpoint.pt`；
SHA256 `036bc8caedcde4d6dabce276b1a2e4af15d960d88620098b19e1c198849b8bfe`。
其独立CC=.86249251，去光=.84229470；新权重的去光必须另测，不复用这个数。

mask结构性只是辅助观察：相位包裹、相干干涉及多样本优化本来就可能产生复杂相位。
不强行加入“看起来像图案”的正则。判断训练有效应同时看物理相位变化、固定输入CCD特征变化、
路由对输入的区分、完整测试与同权重去光。第一阶段短暂掉分不自动回滚，否则无法验证后端适配。
