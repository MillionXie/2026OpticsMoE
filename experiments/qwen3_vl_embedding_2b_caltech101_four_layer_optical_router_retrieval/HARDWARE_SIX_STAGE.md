# 六阶段光学 Router 实测闭环

本文件只描述新工程的严格硬件状态机。旧 warmstart5 的四阶段
`hardware_bridge` 不认识两次 Router 曝光，不能直接用于本实验。

## 1. 固定合同

- 顺序固定为：`vision_router → vision_expert → vision_global →
  language_router → language_expert → language_global`。
- Router 输入是居中的 224×224 振幅；Expert/Global 仍是原来的 478×478
  MoE4 振幅。
- 六张可训练相位都先形成逻辑 478×478 mask；Router 自身的 224×224 相位嵌在
  该逻辑面阵中央。导出保持旧合同：17 μm 逻辑采样按物理坐标最近邻映射到
  8 μm、1920×1200 相位 SLM，中心由配置给出（当前 `(980,590)`），并保留原来的
  vertical flip。程序会把实际 native bounds 和 phase SHA 写入合同，越界直接中止。
- 六个阶段的 CCD 都必须由实验室 homography/方向合同处理成 canonical、单通道
  uint8、478×478。Router 评分和四个 feature measured-CCD loader 均不再翻转；
  release 配置中的 downstream vertical/horizontal flip 必须都是 `false`。
  `score_router` 也不扣背景。
- 四个计分区严格读取 release config；当前可达性修正版为 `[164,223)` 和
  `[255,314)` 的笛卡尔积。专家顺序是左上、右上、左下、右下，不能在实验室
  手工改坐标。
- Expert 布局仍为 224×224，pitch=254，即两方向均保留 30 像素间隔。
- 所有导出、CCD、routing、checkpoint 均由 manifest 和 SHA 串联。目标目录
  已存在任何文件时程序会中止，避免旧数据混入。

## 2. 初始化和一次性六相位导出

在服务器仓库根目录执行：

```text
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.hardware_bridge \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/optical_power_topk2.yaml \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/runs/optical_power_topk2/ema_best_train_loss_checkpoint.pt \
  --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/hardware_sessions/router_run1 \
  --phase initialize
```

初始化会生成固定数据集 `manifest.csv`、`six_stage_state.json`，并从同一个
checkpoint 一次性导出全部六张相位：

```text
hardware_sessions/router_run1/00_phase_bundle/phase_to_play/
  vision_router.bmp
  vision_expert.bmp
  vision_global.bmp
  language_router.bmp
  language_expert.bmp
  language_global.bmp
```

这六张用于证明“同 checkpoint、同几何”的初始相位合同。逐层微调以后，尚未
采集的下游相位可能改变；每次 `--phase export` 会在对应阶段目录重新导出当前
checkpoint 的正确相位，实验时应播放阶段目录中的版本。

## 3. 每个阶段的三类操作

### 3.1 服务器导出

以 Vision Router 为例：

```text
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.hardware_bridge \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/configs/release/optical_power_topk2.yaml \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/runs/optical_power_topk2/ema_best_train_loss_checkpoint.pt \
  --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/hardware_sessions/router_run1 \
  --stage vision_router \
  --phase export \
  --inference-batch-size 10
```

该命令真实地从 Qwen Vision 边界导出逐样本 224×224 中心 Router 输入，不需要
用户手工构造 `central_router_inputs`。阶段目录含：

- `compact_amplitude/`：传输用逻辑 PNG；
- `amplitude_to_play/`：Meadowlark 1024×1024 BMP；
- `compact_phase/` 与 `phase_to_play/`：该阶段当前相位；
- `compact_amplitude_manifest.csv` 和 `transport_spec.json`；
- 空的 `ccd_captured/`。

### 3.2 实验室播放和采集

把整个阶段目录复制到实验室工程。确认实验室唯一配置文件中的 LUT、曝光、等待、
相机 homography 均正确，然后在实验室工程根目录执行：

```text
python -m experiments.hardware_sdk.workflows.acquire_folder \
  --config experiments/lab_qwen/generated/formal_hardware.yaml \
  --stage-dir <复制到实验室的阶段目录>
```

确认 `phase_to_play/` 中唯一 BMP 已加载到相位 SLM。采集程序必须保存 homography
之后的 `canonical_model_xy`，不能保存相机原方向，也不能再增加 downstream flip。
采集结束后，把完整的 `ccd_captured/` 和 `acquisition_logs/` 一起上传回服务器原
阶段目录。不要改文件名、不要混入预览图，也不要在上传前再次翻转 CCD。服务器
会核对 acquisition log 中的 homography 状态、phase SHA、amplitude SHA 和 CCD SHA；
只有图片而没有正式采集日志会被拒绝。

### 3.3 服务器验收

所有阶段先执行：

```text
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.hardware_bridge \
  --config <同一 optical_power_topk2.yaml> \
  --checkpoint <当前 checkpoint> \
  --session-dir <同一 router_run1> \
  --stage <当前 stage> \
  --phase validate_capture
```

Router 阶段验收后还必须评分：

```text
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.hardware_bridge \
  --config <同一 optical_power_topk2.yaml> \
  --checkpoint <当前 checkpoint> \
  --session-dir <同一 router_run1> \
  --stage vision_router \
  --phase score_router
```

程序按 session manifest 的顺序生成 `routing/routing.csv`，并严格验证概率、
deterministic top-k、selected mask、振幅权重以及 `sum(weight²)=1`。该 routing 会在
后续模型 forward 中显式注入 Vision Router，Vision Global 也复用同一 route。

## 4. 完整执行顺序

下面的 `<CURRENT.pt>` 始终表示 `six_stage_state.json` 中的
`current_checkpoint`。每次 feature fine-tune 后必须改用新 checkpoint。

1. `vision_router export → 实验室采集 → validate_capture → score_router`。
2. 用同一 `<CURRENT.pt>` 执行 `vision_expert export`。程序读取实测 Vision
   routing，生成原 MoE4 2×2 Expert 振幅。
3. 实验室采集 `vision_expert`，然后 `validate_capture`。
4. 微调并选择 development 最优 epoch：

```text
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.hardware_bridge \
  --config <同一 optical_power_topk2.yaml> \
  --checkpoint <CURRENT.pt> \
  --session-dir <同一 router_run1> \
  --stage vision_expert \
  --phase finetune \
  --epochs 100 \
  --selection-policy development \
  --early-stopping-patience 15
```

   新 checkpoint 为 `checkpoints/after_vision_expert.pt`。
5. 用 `after_vision_expert.pt` 执行 `vision_global export`。程序注入实测 Vision
   routing 和 Vision Expert CCD，完成电子 readout、Mixer Block 1 与融合，再用同一
   route 生成 Global 振幅。
6. 采集、验收并微调 `vision_global`，得到 `after_vision_global.pt`。
7. 用 `after_vision_global.pt` 执行 `language_router export`。该输入已经经过两张
   实测 Vision CCD、电子融合和冻结 Qwen merger。
8. 采集、验收、`score_router --stage language_router`。
9. 用当前 checkpoint 导出、采集、验收并微调 `language_expert`。
10. 用 `after_language_expert.pt` 导出 `language_global`；程序复用实测 Language
    route，并注入此前全部实测 CCD。
11. 采集、验收并微调 `language_global`。最终 checkpoint 是
    `checkpoints/after_language_global.pt`；sealed test 不参与 epoch 选择，仅在选出
    development 最优 checkpoint 后由旧四层评估合同执行一次。

四个 feature stage 的 `finetune` 参数冻结、development 选模、loss、最终 test
评估均直接复用已审计的旧四层 measured-CCD trainer；新桥只替换模型加载、六阶段
目录映射和实测 routing 注入。

## 5. 查看状态和排错

```text
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.hardware_bridge \
  --config <同一 optical_power_topk2.yaml> \
  --checkpoint <CURRENT.pt> \
  --session-dir <同一 router_run1> \
  --phase status
```

常见中止均是保护性行为：

- `identity mismatch`：config、current checkpoint 或数据 manifest SHA 变了；
- `must be new or empty`：目标目录有旧文件，应建立新 session，不要混跑；
- `CCD/manifest mismatch`：缺图、多图、改名或混入预览图；
- `set_measured_routing`：代码与 checkpoint 工程版本不一致；
- `sum(weight^2)` 或 deterministic top-k 不符：routing CSV 被改过或使用了不同配置；
- 下游 export 缺 dependency：上一阶段尚未完成 capture/routing 合同。

## 6. Router CCD 质量门与指标口径

`score_router` 会在写入并封存 routing 之前逐帧检查饱和比例、p99、p99-p01
动态范围以及 Top-k 分界概率 margin。阈值位于 release 配置的
`router_experiment.optical.hardware_quality`，并封存在六阶段 session 的 measurement
合同中；修改阈值后必须新建 session。全均匀帧和全饱和帧无条件拒绝。

若任一帧失败，程序不会生成 `routing.csv`，而会生成
`routing/routing_quality_failures.csv` 和 `routing/routing_quality_report.json` 后中止。
此时该批 CCD 已由前一步 `validate_capture` 封存，不能在原 session 中替换。请保留失败
报告，修正曝光、LUT、ROI 或光路后新建 session，并从导出、采集、验收重新执行；不允许
只删除 `routing/` 后用替换过的 CCD 混入旧 session。

未提供实测暗场时程序绝不臆造 background。此时四区原始码值之和除以全帧原始
码值之和只能称为 `raw_capture_fraction`，用于诊断光分布，不能写成物理捕获效率；
仿真中无相机 offset 的 `capture_fraction` 则保留原有物理含义。
