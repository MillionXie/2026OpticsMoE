# ABO 文搜图 10 cm、α≈0.40 版本

这是 2026-09-25 新训练的 **ABO text-to-image（文搜图）仿真模型**，不是 α≈0.15、Hit@1=0.91 的版本，也不是旧的 15 cm 模型。

- 唯一权重：`best_checkpoint.pt`，SHA256 `cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae`。
- 光学传播：10 cm；波长 532 nm；模型逻辑像素 17 μm，相位 SLM 8 μm。
- 光 Router Top-2、Vision 两层、Language 两层；四处同尺度融合 α 均约 0.40。
- 最佳 epoch 12；仿真 TEST Hit@1=0.86、Hit@5=0.98、Hit@10=1.00。同权重关闭光支路的 Hit@1=0.73。测试集用于选 epoch，指标有选模偏差。

`config.yaml`、`architecture.json` 与上述权重来自同一服务器运行。`optical_moe.py` 已更正为训练实际使用的 `t08_text_to_image_20260920` 工作树源码，SHA256 `ebf026e5aa2eaf6dc32ef7b5c5f18c5ac2eaea8515fe2d5a3431427d67e97869`；最初误随包传送的主树旧图搜文源码不可使用。`final_report.json` 和 `run_manifest.json` 记录指标与来源。`best_phase_overview.png` 只用于查看训练相位，不是可直接加载到相位 SLM 的 BMP。

`export_phase_bmp.py` 从正确 checkpoint 导出六张 1920×1200 BMP 到 `phase_bmp_provisional/`；采用 17→8 μm 物理中心采样、9 月 24 日光路已用过的方向/灰度配置。文件夹名称保留了早期“provisional”历史名，不代表正式采集会因低 PCC 被拦截。

## 部署状态与边界

选中模型已部署到实验电脑，并完成六层完整 TEST 实测。相位 BMP 由本 checkpoint 导出，采用已知的 8 μm 灰度反向/方向；CCD 采用 2026-09-24 的四角 ROI 和阶段方向，保存线性 uint8（只做几何透视/方向处理）。低仿真-实测 PCC 不拦截；超过 1% 的明显饱和或文件损坏会停机。六层与 100 个标题的实拍均已完成，`physical_test_report.json` 的实测 Hit@1 为 **0.79**，仿真为 **0.86**，相差 0.07。完整实测 Hit@5=0.96、Hit@10=0.97、MRR≈0.85335。

部署到实验电脑的独立目录：`E:\code\guest\2026OpticsMoE\ABO_T2I_10cm_alpha040_20260925`。原有 ABO 图搜图和 15 cm 图搜文目录不作覆盖。

正式采集的逐层日志、CCD 与紧凑输入在该目录的 `full_test/01_vision_router` 至 `full_test/06_language_global`；服务器逐层导出与评价位于 `LightGenV2/tasks/t08_abo_image_text_retrieval/runs/physical/alpha040_10cm_20260925`。接力脚本见本包的 `export_full_next_stage.py`、`capture_full_stage.py`、`sync_full_stage.py`、`continue_full_test.py`，只用 `best_checkpoint.pt` 对应的固定 TEST split，不按实测结果挑选/删改测试图片。

结果文件：`physical_test_report.json`（总指标）、`physical_test_predictions.csv`（100 个标题的逐查询排名）、`physical_capture_qa.json`（六层文件数、曝光、p99 与饱和审计）、`vision_router_routing_summary.json` 与 `language_router_routing_summary.json`（实测专家选择分布）。Vision Router 的选择计数为 1139/1201/1075/1385；Language Router 为 2500/2437/63/0（每样本 Top-2，总计分别为 4800 和 5000 次选择）。后者明显集中，但不能仅凭此断言 0.07 差距的唯一原因。
