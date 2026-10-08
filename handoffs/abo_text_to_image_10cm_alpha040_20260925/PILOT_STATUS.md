# 10 cm α≈0.40 实测进度（2026-09-25）

本文件记录正式采集前的试采。用户已明确要求沿用既有方向与 ROI：低仿真-实测 PCC 作为实验结果记录，不作为停止采集门槛。完整 TEST 六层实测现已启动；最终 Hit@1 只在六层及 100 个标题查询均实拍、完成电子推理后报告。

## 已核验

- 师弟电脑的交互桌面中，相位 SLM、振幅 SLM、小相机 SDK 均能连接；最新 CCD 四角采用 TL(995,172)、TR(4310,172)、BR(4303,3440)、BL(980,3435)。相位 LUT 为 `19x12_8bit_linearVoltage`。
- 10 ms 全白灰度在 ROI 内有约 7.08% 像素饱和，20 ms 有约 73.24%；不能拿全白测试的曝光直接认定各网络层合适。四个 Vision Router 真实输入在 10 ms 均未饱和，p99 分别为 37、23、29、21/255；3 ms 的 p99 仅为 12、8、9、7/255。
- 固定振幅时，平相位重复 PCC=0.9987，训练相位重复 PCC=0.9971；平相位与训练相位 PCC≈0.909。相位 SLM 的切换确实产生可重复的光学变化，并非完全不响应。
- 8 种相位灰度/翻转 × 8 种 CCD 方向的四样本扫描，最佳平均 CCD-仿真 PCC≈0.213（`none_normal` + `rot270`）；按预定四个 Router 探测窗比较，最佳 Top-2 仅 2/4 样本一致。历史 T07 第一层准入阈值为 0.30，未通过。

这些现象表明本次瓶颈不是“设备 SDK 完全没有加载相位”，而是**实测 CCD 与 10 cm 模型的 Router 能量分布尚未对齐**。此前因此暂停完整测试是错误的：这正是实际性能需要量化的差距。试采不计入最终测试；正式采集使用固定 TEST split 和原始线性 uint8 CCD。

## 可复查文件

- `pilot_evidence/health_report.json`：灰度与曝光短扫。
- `pilot_evidence/router_10ms_report.json`：四张 10 ms Router 试采。
- `pilot_evidence/router_routes_report.json`：相位/方向搜索及 Top-2 对比。
- `pilot_evidence/phase_response_report.json`：相位平/训练 A-B-A-B 重复性。
- `pilot_evidence/actual_router_00_preview.png` 与 `pilot_evidence/sim_router_00_preview.png`：第一张的显示用对比增强预览；源 CCD PNG 仍保持线性 uint8，不对网络输入偷偷增强。

正式采集目录在师弟电脑 `E:\code\guest\2026OpticsMoE\ABO_T2I_10cm_alpha040_20260925\full_test`；各层的 `capture.log`、`capture_journal.jsonl` 与 `ccd_captured` 分开保存。服务器的逐层导出与最终评价位于 `LightGenV2/tasks/t08_abo_image_text_retrieval/runs/physical/alpha040_10cm_20260925`。只有 `physical_test_report.json` 生成后才发布当前光路的实测 Hit@1。
