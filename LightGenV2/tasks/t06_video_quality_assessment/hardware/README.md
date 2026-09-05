# T06 实验室六阶段

每个视频在当前方案中有 36 帧，6×6 排在固定 478×478 有效光场中。一次完整推理按
以下顺序采集六个 CCD 平面：

1. `vision_router`
2. `vision_expert`
3. `vision_global`
4. `language_router`
5. `language_expert`
6. `language_global`

前四个融合节点分别在完成 vision expert、vision global、language expert、language
global 后微调。通用 LUT、曝光、ROI、单应性和双 SLM 对齐继续使用
`experiments/hardware_sdk` 与 `experiments/lab_lgvq`；本目录只负责 T06 的顺序。

## 示例

先确认 canonical checkpoint 在当前机器存在，然后从仓库根执行：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment.hardware export-pass `
  --optical-pass vision_router `
  --session-dir LightGenV2\tasks\t06_video_quality_assessment\runs\hardware\temporal36_run1 `
  --all-data
```

播放生成的振幅 BMP、手动保持对应相位 BMP，再用统一硬件采集程序填充该 pass 的
`ccd_captured`。之后：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment.hardware validate-capture `
  --optical-pass vision_router `
  --session-dir LightGenV2\tasks\t06_video_quality_assessment\runs\hardware\temporal36_run1
```

完成所有 pass 后，按旧正式指南中同样的四阶段顺序微调。新 wrapper 不改变数值合同，
只统一目录和入口。完整设备操作仍以打包进 ZIP 的 `README_FIRST.md` 为准。
