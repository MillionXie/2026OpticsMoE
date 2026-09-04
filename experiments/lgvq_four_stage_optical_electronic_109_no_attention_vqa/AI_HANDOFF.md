# 给后续 AI/开发者的工程合同

## 目标

把一份已经完成仿真训练的无 attention 光电视频质量模型部署到当前实验台，并按
`LAB_DEPLOYMENT.md` 顺序用实测 CCD 替换仿真传播、逐层微调。允许修改硬件路径、
曝光、LUT、CCD 四点与 batch size；不得未经新实验批准改变模型推理拓扑。

## 推理拓扑不可改项

- 输入恰好 4 帧 RGB，张量 `[B,4,3,224,224]`。
- 无 Qwen、Transformer、attention、mixer、电子 router、冻结外部 backbone。
- 电子前端是 14 个确定性质量通道加 5 层 Conv2D。
- 光学 router 只出现在 stage 1 与 stage 3，均为固定四区域能量的 Top-2。
- 光学专家尺寸 109×109、2×2 布局；有效场 478×478；数值 FFT canvas 518×518。
- 波长 532 nm、逻辑像素 17 μm、传播 10 cm、k 空间限带开启。
- 每层融合先分别做逐样本 RMS 尺度匹配，再计算
  `(1-alpha)E_norm + alpha O_norm`；alpha 限制在 `[0.50,0.90]`。
- 最终读出只允许普通卷积、归一化、全连接；输出 `[Spatial, Temporal]`。

## SLM/CCD 几何不可混淆项

- 518 是仿真 FFT 的数值画布，不是 SLM 有效图尺寸。
- 振幅 SLM：478×478 uint8 逻辑振幅，17 μm→17 μm 原生 1:1，居中嵌入
  1024×1024；255 为亮/透光。
- 相位 SLM：478×478 逻辑相位按物理像素中心最近邻映射 17 μm→8 μm，得到
  1016×1016 有效栅格；当前默认居中于 1920×1200 的 `(980,590)`。
- 当前默认相位导出前 `flip_vertical=true`、`flip_horizontal=false`；硬件软件不得再次翻转。
- 上述中心与翻转的唯一可编辑来源是 `experiments/lab_lgvq/LAB_CONFIG.yaml` 的
  `phase_slm`；换实验台只改这一处，`hardware_bridge` 会在每个 pass 自动读取。
- 相位 BMP 只编码线性的 `[0,2π)` uint8 目标；它假定目标 8 μm 相位 SLM 已在 532 nm
  下完成灰度到相位响应校正。本包不提供可跨器件复用的相位响应 LUT。
- CCD 用四个逻辑角点生成单应性，输出 canonical 478×478。不要在下游再翻转。

## 六 pass 状态机

固定次序为：

```text
stage1_router -> stage1_expert -> stage2_global
              -> stage3_router -> stage3_expert -> stage4_global
```

stage 1/3 的 router pass 必须先采集，因为其实测四区能量决定紧随其后的 expert
振幅图。每完成一个融合 stage 才能微调并产生下一 checkpoint。代码必须拒绝缺失、
越序、哈希不匹配或来自另一 frame cache/checkpoint 的采集。

## 微调边界

已经采集的光学相位和生成其振幅的上游计算必须冻结，否则已有 CCD 不再对应当前
网络。允许训练该 CCD 之后的读出、融合、电子支路和尚未采集的下游相位。当前按 test
间隔选模是实验负责人明确选择，必须在报告保留 `test_used_for_selection=true`。

## 仿真结果解释边界

“optics off”是同一正常光电 checkpoint 在推理时旁路全部光学分支，不是单独训练的
纯电子模型。它只回答已训练模型对光支路的依赖程度。stage 3 router 当前选择份额集中
到一基编号的专家 E2/E4（等价于零基索引 1/3），这是需如实报告的路由塌缩现象，
不能删除该诊断或把它写成均衡路由。

## 重要入口

- 模型：`modeling.py`
- 训练/评估：`training.py`, `run.py`
- 六次硬件合同：`hardware_contract.py`
- 导出/校验/逐层微调：`hardware_bridge.py`
- 仿真—实测一致性：`agreement_evaluate.py`（严格同名配对；PCC/SSIM/gain-aligned NMAE）
- 唯一硬件配置：`experiments/lab_lgvq/LAB_CONFIG.yaml`
- 完整操作：`LAB_DEPLOYMENT.md`
- 仿真证据：`SIMULATION_REPORT.md` 与 `evidence/recommended/`
