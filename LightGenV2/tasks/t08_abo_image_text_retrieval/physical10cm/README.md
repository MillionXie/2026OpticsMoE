# 文搜图采用版：实拍回放与读出工具

本包归入当时服务器实际运行的七份工具，来源和逐项 SHA 见仓库
`maintenance/storage/T08_PHYSICAL_TOOLS_IMPORT_20261004.json`。仅将局部导入改为
包内导入、将反向检索实现明确绑定 `reverse_runtime`，计算函数和历史科学合同不改。
命令从仓库根以 `python -m LightGenV2.tasks.t08_abo_image_text_retrieval.physical10cm.<模块>`
调用；各模块 `--help` 列出显式数据/配置/封存权重/CCD目录参数。

- `export_full_next_stage`：按全数据逐层导出，重放此前各层真实CCD；TRAIN固定每类8图。
- `cache_physical_readout`：从六层真实CCD重建末端读出前特征。
- `finetune_readout`：只改原LN+Linear读出，600 TRAIN拟合/200 TRAIN留出选模，封存后TEST一次评价。
- `evaluate_full_physical`：固定主体或指定SHA的适配PT，重放100标题和2400 TEST图。
- `audit_physical_router`：逐样本记录原真实路由。
- `export_router_pilot`、`export_full_vision_router`：历史导出和诊断对照，非新硬件验收。

这是 **2026-09-25旧文搜图合同**，保留其每图正值p99.5幅度缩放，不能冒称后来
ABO/OpenMoji的保零bounded tanh合同；不要套给其他主体或已有新版CCD。
主体 `cc977b83...`、用户采用10轮适配读出 `89e25360...` 的 .85 版本及200轮
读出 `d4d1f6c4...` 的 .99–1.00 对照分开保存，不相互覆盖、不混用测速PT。

这些是原运行工具，原输出写入/接续行为保留；仅在明确授权的新实验目录下运行，
不能用整理工作触发重采、重评或覆盖已封存报告。本次只进行了源码和有界合成CPU检查。

## 采用版DVP逐层采集入口

`capture_full_stage` 现直接调用本包 `bench` 和
`LightGenV2.hardware_common.dvp_legacy.Camera`，不再导入外部图搜图模型或Torch。
振幅与相位控制仍使用AST/源码一致的共享Holoeye、PhaseHDMI实现。
其原DVP相机不是后来SHS高速相机，不能互换相机类或套用400µs/GainX4合同。
来源、逐源码SHA和审核边界见 `maintenance/storage/T08_CAPTURE_SOURCE_IMPORT_20261004.json`。

以下只是已有资产的采集入口说明，本次整理未执行：

```bash
python -m LightGenV2.tasks.t08_abo_image_text_retrieval.physical10cm.capture_full_stage \
  --project /absolute/path/to/existing/ABO_T2I_10cm_alpha040_20260925 \
  --machine-config /absolute/path/to/private/t08_dvp_machine.json \
  --stage vision_router --run-name full_test
```

外部JSON必须显式提供 `camera_dll`、`phase_sdk`、`phase_lut`、`amplitude_sdk`、
`amplitude_bin` 五个实际文件/目录路径；相对路径按此JSON所在目录解释。机器SDK、
LUT和现有数据不进Git，不从示例猜测路径。默认仍为原10,000µs、gain=1.0、wait240ms、
六帧取最后帧、固定原ROI及各层方向；本次未改变原1%饱和守卫或添加新归一化。
TEST前三层各2400图、后三层各2500项（含100标题）；TRAIN为800/900项。
新入口仍保留原断点接续行为，只能用于明确授权的运行目录，不能为整理重拍。

11项纯CPU合同检查和原采集函数/Bench生命周期AST比较通过，未加载SDK或打开设备。
实验室原脚本保持原样；机器二进制身份、真实设备回归和Git部署切换尚未完成。
私人连接/服务器接力脚本不公开归入包，不能将其存在当作主线部署已完成的证据。
