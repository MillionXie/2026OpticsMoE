# LGVQ 实验室包索引与边界

本包用于把已经完成仿真训练的四帧光电模型部署到当前 Meadowlark 1024×1024
振幅 SLM、手动 1920×1200 相位 SLM 和 TUCam CCD，并在每个光电融合阶段后做本地微调。

## 唯一操作入口

1. 在解压根目录运行 `python VERIFY_BUNDLE.py`。
2. 从根目录 `README_FIRST.md` 开始，严格按其 0–9 节顺序执行。
3. 实验人员只编辑 `experiments/lab_lgvq/LAB_CONFIG.yaml`。

不要把项目内 `RUN_COMMANDS.md` 当作实验室命令。该文件只是服务器从零训练的溯源记录，
需要包外的原始 LGVQ 视频和训练期软目标。

## 已包含

- 正式 epoch-75 checkpoint 及固定 SHA256；
- 2250 train + 558 test 的四帧 uint8 缓存和样本清单；
- 六张初始物理相位 BMP、逻辑相位 PNG、相位可视化和重建清单；
- 六个 pass、每个 8 个样本的 1024×1024 振幅 BMP 与理论 CCD 预览；
- 双 SLM `k=1` 标定、10 cm P1/P4/P9 菲涅尔标定图；
- Meadowlark/TUCam 驱动、ROI/曝光/LUT 标定、采集及完整性校验代码；
- 六次曝光、四次微调的状态机和每阶段命令；
- 仿真指标、alpha、光 router、同 checkpoint 去光消融及相位训练诊断。
- 仿真配置小型档案，用于核对训练 provenance 并让随包测试完整运行；唯一选中的正式
  配置是 `configs/release/formal_alpha50_kd300_center100.yaml`，实验室只使用
  `configs/deployment/lab_hardware_finetune.yaml`，不要改用其他 release 变体。

## 刻意不包含

- 原始 LGVQ 视频；
- 训练期教师网络或软目标（部署和 CCD 微调均不需要）；
- 某一台实验台专属的线性化 LUT。初始使用随包原厂 LUT，在目标光路完成 128 点标定并
  通过验证后，才能切换到新 LUT；
- 可跨器件复用的相位 SLM 响应 LUT；物理相位 BMP 假定目标器件已在 532 nm 下校正；
- Holoeye、DVP 或其他旧硬件路径。

## 四帧与六次曝光

模型有四个融合 stage，但两个光 router 也需要真实传播，所以固定为六个 pass：

```text
stage1_router -> stage1_expert -> stage2_global
              -> stage3_router -> stage3_expert -> stage4_global
```

前三个 pass 的一张振幅 BMP 已同时编码四个视频帧，并非顺序播放四帧；stage 3 之后四帧
信息已经桥接到一条序列光场。后续 pass 必须由上一阶段微调后的 checkpoint 重新导出，
因此包内全量缓存用于逐阶段生成，不能预先把六个全量振幅目录一次性固定下来。

## 关键事实

- 推理中没有 Qwen、Transformer、attention、mixer 或电子 router；
- 两个 router 均由 CCD 四区域能量产生概率并固定取 Top-2；
- 专家为 109×109、2×2；有效场 478×478，518 仅为 FFT 数值画布；
- 四层 alpha 为尺度匹配后的凸组合系数，不等于原始光功率占比；
- “optics off” 是同一正常光电 checkpoint 的推理旁路，不是单独训练的纯电子模型。

随包 JSON/CSV 中少量 `input_dir`、`output_dir`、`checkpoint` 绝对路径仅是生成时的
provenance 记录；执行时以解压后的相对路径、`README_FIRST.md` 变量和 SHA256 为准，
不要照抄生成机器的盘符或服务器路径。
