# Temporal SRCC 0.8044：给接收方 AI 的移植合同

## 1. 唯一模型身份

- Run：`multivideo16x4_rank_s163`，不是 Temporal-36，也不是单视频 16 帧模型。
- 每个光场并行放置 **16 个互相独立的视频**，每个视频使用 4 帧；一次前向输出 16 个连续 Temporal MOS。
- test：固定 558 个视频，共 35 个光场；最后一场 14 个有效槽位和 2 个 padding，padding 不计指标。
- 主权重：`weights/best_checkpoint.pt`。
- checkpoint SHA256：`5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c`。
- 完整仿真：SRCC 0.8043868643、KRCC 0.5968244684、PLCC 0.8180329374、RMSE 7.9910755、MAE 5.9920588。

`release.json` 是样本、标签、槽位和参考预测的唯一清单；`SHA256.json` 是包内文件完整性清单。不要把本权重和别的配置、36 帧缓存或旧相位混装。

## 2. 包内能够独立复现什么

包内 `inputs/field_0000.pt` 至 `field_0034.pt` 已经是对应固定 Temporal prompt 和视频的冻结 Qwen 前端缓存，因此复算这 558 条 test **不需要联网、原视频或加载完整 Qwen**：

```bash
python simulate.py --device cuda --fields 0
```

命令会先检查全部文件 SHA，再严格加载 state_dict，重新执行完整光电仿真，输出 `simulation_reproduction.json`，并逐样本核对包内参考预测。`--fields 1` 只用于快速冒烟测试，不能报告正式指标。不同 GPU/PyTorch 的 FFT 与归约核并不保证逐位一致；脚本允许最多 0.01 MOS 的逐样本数值误差，同时分别限制 SRCC、KRCC、PLCC、RMSE、MAE 的偏差，不能只凭“看起来接近”放行。

这并不代表包内包含了重新训练所需的一切。若要换数据、换 prompt 或重新训练，接收方仍需原视频/标签、Qwen3-VL-2B-Instruct 前端以及训练缓存生成代码；不能拿这 35 个 test 缓存训练。

## 3. 推理计算图与不能改变的合同

固定推理保留文本输入；这里只是把昂贵且冻结的 Qwen 前端结果预先缓存。主模型不运行 Qwen Transformer block/attention。

六次光传播顺序固定为：

1. `vision_router`
2. `vision_expert`
3. `vision_global`
4. `language_router`
5. `language_expert`
6. `language_global`

两次 router 都是光学 Top-2，均为 4 专家。四层同尺度 RMS 融合的 alpha 约为 `0.5611, 0.5600, 0.5679, 0.5695`，均表示归一化后光分支占比；电子占比为 `1-alpha`。训练配置允许 alpha `[0.5, 0.9]`。仿真评估使用 20% 名义未调制功率。相位使用连续精度仿真，没有 8-bit 直通量化训练。

模型几何固定为 532 nm、传播 10 cm、模型像素 17 um、逻辑有效面 478x478。`runtime/` 内建模代码和 `phases/*.npy` 与 PT 是同一版本；改代码后必须重新做严格加载和仿真复算。

## 4. 移植到另一套实验设备时允许修改什么

允许修改的是设备适配层，不是模型：

- 振幅 SLM、相位 SLM、CCD 的 SDK/驱动调用；
- 17 um 逻辑面到实际 SLM 像素的物理尺寸保持映射、居中和方向；
- CCD 四点透视、镜像方向、固定 ROI；
- LUT、曝光、换图稳定等待、旧帧丢弃；
- 每层固定的 CCD DN 到模型强度比例，需独立标定，不能逐图 min-max；
- 把六层采集组织成接收方已有的硬件工作流。

禁止在移植时悄悄改变：16 视频 x 4 帧槽位顺序、六层顺序、Top-2、专家数、478 逻辑几何、相位方向、文本缓存、归一化、alpha、读出头或 checkpoint。若确实修改其中任何一项，它就是新模型，不能继续声称复现 0.8044。

本包自带的 `run.py` 已实现一套 SHS/Holoeye 六层流程，可作为设备适配参考：

```bash
python run.py --help
python run.py init --session pilot01 --fields 1 --config LAB.local.json
python run.py prepare --session pilot01 --stage vision_router --device cuda --config LAB.local.json
```

之后按 `COMMAND_SHS.md` 逐层采集。若接收方设备不同，优先只替换 `runtime/.../lab_bench.py` 的 raster/capture 边界和硬件配置，不复制或重写模型 forward。

## 5. 文件地图

- `weights/best_checkpoint.pt`：唯一正式权重，包含相位、router、融合和电子读出参数。
- `runtime/`：与权重精确匹配的最小推理/光学代码，不依赖仓库其他目录。
- `inputs/`：完整 558 test 的 35 个冻结 Qwen 前端场缓存。
- `phases/`：六层 canonical 478x478 浮点相位；实际 BMP 应由设备适配层按物理尺寸与方向生成。
- `simulate.py`：无硬件的完整仿真复现入口。
- `run.py`、`COMMAND_SHS.md`：逐层硬件部署参考。
- `release.json`：正式指标、样本、标签、padding 和逐样本仿真预测。
- `SHA256.json`：包内完整性校验。

先运行 `simulate.py --fields 1` 确认环境和权重可加载，再运行 `--fields 0` 获得正式复现证据；两步通过以后再动硬件适配代码。
